import os
import sys
import json
import streamlit as st
import weaviate
from weaviate.classes.init import Auth
from typing import Dict, List, Any
from dotenv import load_dotenv  # Added dotenv import

# LangChain imports
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_weaviate import WeaviateVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain

# Setup environment variables
def setup_environment():
    """
    Load environment variables from .env file and check for required variables.
    Returns True if all required variables are set, False otherwise.
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Check for required environment variables
    required_vars = ["WEAVIATE_URL", "WEAVIATE_API_KEY", "GEMINI_API_KEY"]
    
    missing_vars = []
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file and ensure all required variables are set.")
        return False
    
    print("✓ All required environment variables are set")
    return True

# Initialize Weaviate client
def init_weaviate_client():
    """Initialize and return a Weaviate client."""
    try:
        weaviate_url = os.environ.get("WEAVIATE_URL")
        weaviate_api_key = os.environ.get("WEAVIATE_API_KEY")
        
        # Connect to Weaviate Cloud with additional configuration to handle gRPC issues
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=Auth.api_key(weaviate_api_key),
            additional_config=weaviate.classes.init.AdditionalConfig(
                timeout=weaviate.classes.init.Timeout(init=60),  # Increase init timeout
            ),
            skip_init_checks=True  # Skip gRPC health checks if they fail
        )
        
        # Try to check readiness, but continue if it fails
        try:
            is_ready = client.is_ready()
            print("Weaviate ready:", is_ready)
        except Exception as readiness_error:
            print(f"Warning: Could not check if Weaviate is ready: {readiness_error}")
            print("Continuing with connection despite readiness check failure...")
        
        return client
        
    except Exception as e:
        print(f"❌ Failed to connect to Weaviate: {e}")
        return None

# Create or get collection
def ensure_collection_exists(client, collection_name="RFPChunk"):
    """Create the collection if it doesn't exist."""
    try:
        collections = list(client.collections.list_all())
        collection_names = []
        
        # Handle collections that might be returned as strings or objects
        for c in collections:
            if isinstance(c, str):
                collection_names.append(c)
            elif hasattr(c, 'name'):
                collection_names.append(c.name)
            else:
                print(f"Warning: Could not determine name for collection: {c}")
        
        print(f"Found collections: {collection_names}")
        
        if collection_name not in collection_names:
            print(f"Creating {collection_name} collection...")
            # Create the collection
            collection = client.collections.create(
                name=collection_name,
                description="RFP document chunks for semantic search",
                properties=[
                    {
                        "name": "text",
                        "dataType": "text",
                        "description": "The text content of the document"
                    },
                    {
                        "name": "source_pdf",
                        "dataType": "text",
                        "description": "Source PDF filename"
                    },
                    {
                        "name": "page_number",
                        "dataType": "number",
                        "description": "Page number in the source document"
                    },
                    {
                        "name": "chunk_index",
                        "dataType": "number",
                        "description": "Index of the chunk within the document"
                    },
                ],
                vectorizer_config=weaviate.config.Configure.Vectorizer.none()
            )
            print(f"Collection created: {collection.name}")
            return collection
        else:
            print(f"{collection_name} collection already exists")
            return client.collections.get(collection_name)
            
    except Exception as e:
        print(f"Error checking/creating collection: {e}")
        return None

# Initialize LangChain components
def init_langchain(client, collection_name="RFPChunk"):
    """Initialize LangChain components including vector store and LLM."""
    # Set up embeddings using Gemini
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.environ["GEMINI_API_KEY"],
        task_type="retrieval_query"  # For search queries
    )
    
    # Make sure collection exists
    collection = ensure_collection_exists(client, collection_name)
    if not collection:
        print("Failed to ensure collection exists")
        return None, None
    
    # Initialize the vector store with LangChain Weaviate integration
    try:
        vector_store = WeaviateVectorStore(
            client=client,
            index_name=collection_name,
            text_key="text",
            embedding=embeddings,
            attributes=["source_pdf", "page_number", "chunk_index"]
        )
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        return None, None
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.environ["GEMINI_API_KEY"],
        temperature=0.3,
        top_p=0.85,
        top_k=40
    )
    
    return vector_store, llm

# Function to search similar documents
def search_similar_documents(vector_store, query, k=5):
    """Search for similar documents in the vector store."""
    try:
        # Ensure k is at least the requested number, to get enough documents
        docs = vector_store.similarity_search(query, k=max(k, 10))
        if not docs:
            return []
        
        results = []
        for i, doc in enumerate(docs):
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        # Return exactly k documents (or fewer if not enough were found)
        return results[:k]
    except Exception as e:
        print(f"Error searching documents: {e}")
        return []

# Function to generate RFI document
def generate_rfi_document(llm, requirements, reference_docs=None):
    """Generate an RFI document based on requirements and reference documents."""
    # Get current date
    from datetime import datetime
    current_date = datetime.now().strftime("%B %d, %Y")
    
    # Build context from reference documents if available
    context = ""
    if reference_docs and len(reference_docs) > 0:
        context = "Here are some example documents for reference:\n\n"
        # Use up to 5 reference documents
        for i, doc in enumerate(reference_docs[:5], start=1):
            context += f"Reference {i}:\n{doc['content']}\n\n"
    
    # Create the prompt
    prompt = PromptTemplate.from_template(
        """You are an expert in creating Request for Information (RFI) documents for industrial and technical projects.
        
        {context}
        
        Using the following requirements, create a comprehensive and professionally formatted RFI document:
        
        Requirements: {requirements}
        
        Your RFI should include:
        1. Introduction and background
        2. Project objectives 
        3. Scope of work
        4. Required vendor information
        5. Evaluation criteria
        6. Submission guidelines
        7. Timeline
        
        Format the document with proper markdown formatting:
        - Use # for the main title
        - Use ## for section headers
        - Use bullet points (* or -) for lists
        - Use appropriate formatting for emphasis where needed
        
        Use placeholders like [Company Name], etc. for specific details that would need to be filled in later.
        The current date is {current_date}.
        """
    )
    
    # Create a runnable sequence instead of LLMChain (updated from deprecated method)
    chain = prompt | llm
    
    # Use invoke instead of run (updated from deprecated method)
    response = chain.invoke({"requirements": requirements, "context": context, "current_date": current_date})
    
    # Extract text from response
    if hasattr(response, 'content'):
        result = response.content
    else:
        result = str(response)
    
    # Ensure proper markdown formatting
    if not result.startswith('#'):
        result = f"# Request for Information (RFI)\n\n**Date: {current_date}**\n\n{result}"
    
    return result

# Function to add documents to the vector store
def add_documents_to_vectorstore(vector_store, documents: List[Dict[str, Any]]):
    """Add documents to the vector store."""
    # Convert to LangChain Document format
    langchain_docs = []
    for i, doc in enumerate(documents):
        content = doc.get("text", "")
        # Create metadata dictionary, ensuring proper types
        metadata = {
            "source_pdf": doc.get("source_pdf", "unknown"),
            "page_number": int(doc.get("page_number", 0)),
            "chunk_index": int(doc.get("chunk_index", i))
        }
        langchain_docs.append(Document(page_content=content, metadata=metadata))
    
    # Add to vector store
    try:
        vector_store.add_documents(langchain_docs)
        print(f"Added {len(langchain_docs)} documents to vector store")
        return True
    except Exception as e:
        print(f"Error adding documents to vector store: {e}")
        return False

# Import documents from JSON file
def import_documents_from_json(vector_store, json_path):
    """Import documents from a JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        documents = []
        for item in data:
            doc = {
                "text": item.get("content", ""),
                "source_pdf": item.get("source", "unknown"),
                "page_number": item.get("page", 0),
                "chunk_index": item.get("chunk_index", 0)
            }
            documents.append(doc)
        
        success = add_documents_to_vectorstore(vector_store, documents)
        return success, len(documents)
    except Exception as e:
        print(f"Error importing documents from JSON: {e}")
        return False, 0

# Import documents from PDF files in a directory
def scan_pdf_directory(directory_path):
    """Scan a directory for PDF files with metadata JSON files."""
    pdf_files = []
    metadata_files = []
    
    try:
        for file in os.listdir(directory_path):
            if file.endswith('.pdf'):
                pdf_path = os.path.join(directory_path, file)
                pdf_files.append(pdf_path)
            elif file.endswith('_metadata.json'):
                metadata_path = os.path.join(directory_path, file)
                metadata_files.append(metadata_path)
                
        return pdf_files, metadata_files
    except Exception as e:
        print(f"Error scanning directory: {e}")
        return [], []

# Sample data for initial testing
sample_docs = [
    {
        "text": "This RFP seeks proposals for a new AI-powered document management system. Evaluation criteria include technical approach (50%), cost (30%), and vendor experience (20%).",
        "source_pdf": "IT_Systems_RFP.pdf",
        "page_number": 1,
        "chunk_index": 0
    },
    {
        "text": "The project timeline should not exceed 6 months with clear milestones defined. The solution must be cloud-based and integrate with our existing systems.",
        "source_pdf": "IT_Systems_RFP.pdf", 
        "page_number": 2,
        "chunk_index": 1
    },
    {
        "text": "The vendor qualification requirements include at least 5 years of experience in document management solutions and at least 3 reference clients in a similar industry.",
        "source_pdf": "Vendor_Requirements.pdf",
        "page_number": 1,
        "chunk_index": 2
    },
    {
        "text": "This RFI is for the replacement of our aging control system with a modern PLC-based solution. The vendor must have experience with Allen Bradley PLCs and integration with SCADA systems.",
        "source_pdf": "Control_System_RFI.pdf",
        "page_number": 1,
        "chunk_index": 3
    },
    {
        "text": "The selection criteria for this RFP will be based on: Technical capabilities (35%), Project management approach (25%), Price (20%), and Vendor experience (20%).",
        "source_pdf": "Selection_Criteria.pdf",
        "page_number": 1,
        "chunk_index": 4
    }
]

# Streamlit Application
def main():
    st.set_page_config(page_title="RFP/RFI Generator", page_icon="📄", layout="wide")
    
    # Header
    st.title("RFP/RFI Document Generator")
    st.markdown("Generate professional Request for Information documents using AI and similar document references.")
    
    # Initialize state variables
    if 'client' not in st.session_state:
        st.session_state.client = None
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'reference_docs' not in st.session_state:
        st.session_state.reference_docs = []
    
    # Sidebar for setup and configuration
    with st.sidebar:
        st.header("Setup")
        
        if st.button("Initialize System"):
            with st.spinner("Setting up environment..."):
                setup_environment()
                
            with st.spinner("Connecting to Weaviate..."):
                client = init_weaviate_client()
                if client:
                    st.session_state.client = client
                    st.success("Connected to Weaviate!")
                else:
                    st.error("Failed to connect to Weaviate. Check console for details.")
                    return
            
            with st.spinner("Initializing LangChain components..."):
                vector_store, llm = init_langchain(client)
                if vector_store and llm:
                    st.session_state.vector_store = vector_store
                    st.session_state.llm = llm
                    st.success("LangChain components initialized!")
                else:
                    st.error("Failed to initialize LangChain components. Check console for details.")
                    return
                
            with st.spinner("Adding sample documents..."):
                if add_documents_to_vectorstore(vector_store, sample_docs):
                    st.success("Sample documents added!")
                else:
                    st.warning("Could not add sample documents.")
            
            st.session_state.initialized = True
        
        if st.session_state.initialized:
            st.success("System is ready! Use the main panel to generate RFI documents.")
        
        # Document upload section
        if st.session_state.initialized:
            st.header("Add New Documents")
            
            with st.form("upload_form"):
                doc_text = st.text_area("Document Text", height=150)
                source_pdf = st.text_input("Source Document Name", "Custom_Document.pdf")
                page_number = st.number_input("Page Number", min_value=1, value=1)
                
                submitted = st.form_submit_button("Add Document")
                if submitted:
                    if doc_text:
                        new_doc = {
                            "text": doc_text,
                            "source_pdf": source_pdf,
                            "page_number": page_number,
                            "chunk_index": 0
                        }
                        
                        if add_documents_to_vectorstore(st.session_state.vector_store, [new_doc]):
                            st.success("Document added successfully!")
                        else:
                            st.error("Failed to add document.")
                    else:
                        st.warning("Please enter document text.")
    
    # Main panel
    if not st.session_state.initialized:
        st.info("Please initialize the system using the sidebar.")
        return
    
    # RFI Generation
    st.header("Generate RFI Document")
    
    user_input = st.text_area(
        "Describe what kind of RFI you need:",
        height=150,
        placeholder="Example: We need an RFI for replacing the Allen Bradley PLC system in our manufacturing facility with a modern control system."
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        search_first = st.checkbox("Search for reference documents first", value=True, 
                                 help="This will search for similar documents before generating the RFI")
    with col2:
        num_references = st.slider("Number of reference documents to use", 1, 10, 5)
    
    if st.button("Generate RFI", type="primary"):
        if not user_input:
            st.warning("Please describe what kind of RFI you need.")
            return
        
        # Search for reference documents if enabled
        if search_first:
            with st.spinner("Searching for similar documents..."):
                reference_docs = search_similar_documents(
                    st.session_state.vector_store, 
                    user_input, 
                    k=num_references
                )
                st.session_state.reference_docs = reference_docs
                
                # Display reference documents
                if reference_docs:
                    with st.expander("Reference Documents", expanded=False):
                        for i, doc in enumerate(reference_docs):
                            st.markdown(f"### Reference Document {i+1}")
                            st.text_area(f"Content", doc['content'], height=100, key=f"ref_doc_{i}")
                            st.markdown(f"**Source:** {doc['metadata'].get('source_pdf', 'Unknown')}, " +
                                       f"**Page:** {doc['metadata'].get('page_number', 0)}")
                            st.markdown("---")
                else:
                    st.info("No similar documents found for reference.")
        
        # Generate the RFI
        with st.spinner("Generating RFI document..."):
            # Get reference docs if they exist
            reference_docs = st.session_state.reference_docs if search_first else []
            
            # Generate the document
            rfi_document = generate_rfi_document(
                st.session_state.llm,
                user_input,
                reference_docs
            )
            
            # Display the generated document
            with st.expander("Generated RFI Document", expanded=True):
                st.markdown(rfi_document)
                
                # Download button
                st.download_button(
                    label="Download RFI Document",
                    data=rfi_document,
                    file_name="generated_rfi.md",
                    mime="text/markdown"
                )
    
    # Search section
    st.header("Search Existing Documents")
    search_query = st.text_input("Search Query", placeholder="Enter keywords to search for similar documents")
    
    if st.button("Search"):
        if not search_query:
            st.warning("Please enter a search query.")
            return
            
        with st.spinner("Searching documents..."):
            results = search_similar_documents(st.session_state.vector_store, search_query)
            
            if results:
                st.subheader(f"Found {len(results)} relevant documents")
                for i, doc in enumerate(results):
                    with st.expander(f"Document {i+1} - {doc['metadata'].get('source_pdf', 'Unknown')}", expanded=i==0):
                        st.text_area(f"Content", doc['content'], height=150, key=f"search_doc_{i}")
                        st.markdown(f"**Source:** {doc['metadata'].get('source_pdf', 'Unknown')}, " +
                                   f"**Page:** {doc['metadata'].get('page_number', 0)}")
            else:
                st.info("No relevant documents found.")

# Run the Streamlit app
if __name__ == "__main__":
    main()