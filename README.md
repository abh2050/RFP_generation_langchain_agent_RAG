![](https://www.cloudways.com/blog/wp-content/uploads/Request-for-Proposal-OG.jpg)
# RFP/RFI Document Generator

A Streamlit-based web application for generating professional Request for Information (RFI) and Request for Proposal (RFP) documents using AI-powered semantic search and document generation. The application leverages Weaviate for vector storage, LangChain for orchestration, and Google's Gemini API for embeddings and text generation.

## Features

- **Document Search**: Perform semantic searches to find relevant RFP/RFI documents based on user queries.
- **RFI Generation**: Generate professionally formatted RFI documents using user-defined requirements and optional reference documents.
- **Document Management**: Add new documents to the vector store manually or via JSON import.
- **Sample Data**: Includes sample RFP/RFI document chunks for testing and demonstration.
- **Streamlit Interface**: User-friendly web interface for interacting with the system.

## Prerequisites

- Python 3.8 or higher
- A Weaviate Cloud account with API access
- A Google Cloud account with access to the Gemini API
- A `.env` file with required environment variables (see [Configuration](#configuration))

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/abh2050/RFP_generation_langchain_agent_RAG.git
   cd rfp-rfi-generator
   ```

2. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Install the required Python packages listed in the project:
   ```bash
   pip install weaviate-client==4.5.0 langchain==0.1.3 langchain-core==0.1.17 langchain-community>=0.0.10 langchain-weaviate==0.0.1 langchain-google-genai==0.0.6 openai>=1.0.0 google-generativeai>=0.3.0 python-dotenv==1.0.0 streamlit
   ```

4. **Set Up Environment Variables**:
   Create a `.env` file in the project root and add the following variables:
   ```plaintext
   WEAVIATE_URL=your_weaviate_cluster_url
   WEAVIATE_API_KEY=your_weaviate_api_key
   GEMINI_API_KEY=your_google_gemini_api_key
   ```
   Replace the placeholders with your actual credentials. For example:
   ```plaintext
   WEAVIATE_URL=xxxxxxxxxxxxxxxxxxxx
   WEAVIATE_API_KEY=xxxxxxxxxxxxxxxxx
   GEMINI_API_KEY=your_gemini_api_key
   ```

## Configuration

- **Weaviate Setup**:
  - Ensure you have a Weaviate Cloud account and a cluster set up.
  - The default collection name is `RFPChunk`. The application automatically creates this collection if it doesn't exist, with properties for `text`, `source_pdf`, `page_number`, and `chunk_index`.
  - Update the `WEAVIATE_URL` and `WEAVIATE_API_KEY` in your `.env` file to match your Weaviate cluster.

- **Google Gemini API**:
  - Obtain an API key for Google's Gemini model through your Google Cloud account.
  - Add the `GEMINI_API_KEY` to your `.env` file.

- **Collection Schema**:
  The Weaviate collection (`RFPChunk`) is configured with the following properties:
  - `text` (Text): The content of the document chunk.
  - `source_pdf` (Text): The source PDF filename.
  - `page_number` (Int): The page number of the chunk.
  - `chunk_index` (Int): The index of the chunk within the document.
  - `location` (Text): Optional, not used in the provided code but present in the schema.

## Usage

1. **Run the Application**:
   Start the Streamlit app by running:
   ```bash
   streamlit run app.py
   ```
   Replace `app.py` with the name of your Python script if different.

2. **Initialize the System**:
   - Open the Streamlit app in your browser (usually at `http://localhost:8501`).
   - In the sidebar, click **Initialize System** to:
     - Load environment variables.
     - Connect to the Weaviate cluster.
     - Initialize LangChain components (vector store and LLM).
     - Add sample documents to the vector store.

3. **Generate an RFI**:
   - In the main panel, enter a description of the RFI you need in the text area (e.g., "We need an RFI for replacing the Allen Bradley PLC system").
   - Choose whether to search for reference documents and how many to use (default: 5).
   - Click **Generate RFI** to create the document.
   - View the generated RFI in markdown format, with an option to download it as a `.md` file.

4. **Search Documents**:
   - Enter a search query in the **Search Existing Documents** section.
   - Click **Search** to retrieve up to 5 relevant documents based on semantic similarity.
   - Expand each result to view the content and metadata (source PDF, page number).

5. **Add Documents**:
   - In the sidebar, use the **Add New Documents** form to manually add document text, source PDF name, and page number.
   - Submit the form to add the document to the vector store.

6. **Import Documents** (Optional):
   - Use the `import_documents_from_json` function to import documents from a JSON file. The JSON should have the following structure:
     ```json
     [
         {
             "content": "Document text",
             "source": "filename.pdf",
             "page": 1,
             "chunk_index": 0
         },
         ...
     ]
     ```
   - Call the function programmatically or extend the Streamlit interface to support file uploads.

## Project Structure

- `app.py`: The main application script containing the Streamlit interface and core logic.
- `.env`: Environment variable file (not tracked in version control).
- `sample_docs`: Sample RFP/RFI document chunks included in the code for initial testing.

## Sample Documents

The application includes sample document chunks for testing, such as:
- RFP for an AI-powered document management system.
- Requirements for a cloud-based solution with a 6-month timeline.
- Vendor qualification criteria.
- RFI for a PLC-based control system replacement.
- RFP selection criteria.

These are automatically added to the vector store during initialization.

## Notes

- **Weaviate Connection**: If you encounter gRPC or readiness check issues, the code includes fallbacks to skip these checks and continue operation.
- **Vector Store**: Uses Google's Gemini embeddings (`embedding-001`) for semantic search, stored in Weaviate without a native vectorizer.
- **LLM**: Uses Google's Gemini model (`gemini-1.5-flash`) for RFI generation, with configurable temperature, top-p, and top-k parameters.
- **Error Handling**: The application includes robust error handling for Weaviate connections, document imports, and generation tasks.
- **Scalability**: For large-scale use, consider adding multi-tenancy support or optimizing the vector store for larger datasets.

## Troubleshooting

- **Weaviate Connection Errors**:
  - Verify your `WEAVIATE_URL` and `WEAVIATE_API_KEY` are correct.
  - Check the Weaviate Cloud dashboard to ensure your cluster is active.
  - Increase the timeout in `init_weaviate_client` if needed.

- **Gemini API Errors**:
  - Ensure your `GEMINI_API_KEY` is valid and has access to the required models.
  - Check your Google Cloud quota for API usage.

- **Missing Environment Variables**:
  - The `setup_environment` function checks for required variables. Ensure your `.env` file is correctly configured.

- **Streamlit Issues**:
  - Run `streamlit run app.py` from the project root.
  - Ensure all dependencies are installed correctly.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for bug reports, feature requests, or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

This README provides a comprehensive guide to setting up, running, and extending the RFP/RFI Document Generator. Let me know if you need adjustments or additional sections!
