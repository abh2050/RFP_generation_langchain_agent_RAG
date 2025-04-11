import os
import random
import re  # Add this import at the top level
import time
from datetime import datetime, timedelta
from faker import Faker
from fpdf import FPDF
import google.generativeai as genai
import json

fake = Faker()

# Gemini API Configuration
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Configuration settings for Gemini
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Access the Gemini model with configuration settings
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Industrial-focused mappings
industry_mappings = {
    "Manufacturing": {
        "disciplines": ["Mechanical Engineering", "Electrical Engineering", "Process Engineering", "Industrial Automation"],
        "regulations": ["ISO 9001", "OSHA Regulations", "ASME Standards"],
        "project_types": ["Facility Upgrade", "Automation Retrofit", "Emergency Response"]
    },
    "Oil & Gas": {
        "disciplines": ["Piping & Pipeline", "Structural Engineering", "Instrumentation & Controls", "Process Engineering"],
        "regulations": ["API Standards", "OSHA Regulations", "EPA Requirements"],
        "project_types": ["Plant Expansion", "Safety Compliance", "Decommissioning"]
    },
    "Chemical Processing": {
        "disciplines": ["Process Engineering", "Mechanical Engineering", "Environmental Engineering", "Piping & Pipeline"],
        "regulations": ["ISO 14001", "EPA Requirements", "NFPA Codes"],
        "project_types": ["Capacity Enhancement", "Safety Compliance", "Modernization"]
    }
}

location_types = ["Industrial Park", "Refinery Zone", "Factory Complex"]

def select_random_subset(items, min_count=2, max_count=None):
    if max_count is None:
        max_count = len(items)
    count = random.randint(min_count, min(max_count, len(items)))
    return random.sample(items, count)

def generate_project_name(industry, project_type, location):
    prefixes = {
        "Manufacturing": ["Tech", "Forge", "Indust"],
        "Oil & Gas": ["Petro", "Drill", "Refine"],
        "Chemical Processing": ["Chem", "Synth", "React"]
    }
    prefix = random.choice(prefixes.get(industry, ["Project"]))
    city = location.split(",")[0].split()[-1]
    return f"{prefix} {city} {project_type}"

def generate_project_backstory(industry, project_type, location_type):
    backstories = {
        "Manufacturing": {
            "Facility Upgrade": f"Upgrade an aging assembly line in a {location_type} to boost production efficiency.",
            "Emergency Response": f"Repair a {location_type} facility damaged by an industrial accident."
        },
        "Oil & Gas": {
            "Plant Expansion": f"Expand a {location_type} refinery to increase crude oil processing capacity.",
            "Safety Compliance": f"Retrofit a {location_type} platform to meet new safety standards post-incident."
        },
        "Chemical Processing": {
            "Safety Compliance": f"Overhaul a {location_type} plant to comply with stricter emissions regulations."
        }
    }
    return random.choice(backstories.get(industry, {}).get(project_type, [f"Generic {project_type} in a {location_type}."]))

def generate_varied_sow(selected_disciplines, complexity_level, industry, regulations, backstory):
    prompt = f"""
Generate a detailed Scope of Work for an industrial {industry} project. Context: {backstory}
Disciplines: {', '.join(selected_disciplines)}.
Complexity: {complexity_level}/5 (Level 1 = basic upgrades, Level 5 = complex engineering).
Regulations: {', '.join(regulations)} (apply only where relevant).

For each discipline, provide 2-3 detailed, realistic tasks that fit the context and industry.
Include specific technical details (e.g., dimensions, materials, standards, deliverables) to enrich the tasks.
Keep each task 2-3 sentences long, focused on actionable steps.
Example for Oil & Gas:
- Piping: Design a 10-km, 16-inch high-pressure gas pipeline reroute adhering to API 650 standards, including stress analysis and material specs for 1000 psi operation.
- Structural: Reinforce platform supports with 50-ton steel beams to withstand 120 mph winds, delivering shop drawings and load calculations.

End with 1-2 cross-disciplinary tasks (2-3 sentences each) tying teams to the project goal.
Add a brief note (1 sentence) on complexity impact.
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"SOW generation failed: {e}")
        return "Scope of Work unavailable."

def generate_rfq(project_name, project_type, sow_text, location_type, complexity_level, industry, backstory, location):
    submission_days = random.randint(14, 30) if "Emergency" in project_type else random.randint(20, 40)
    inquiry_days = random.randint(7, submission_days - 5)
    project_start_days = random.randint(20, 60)
    project_duration_months = random.randint(3, 12) if complexity_level <= 2 else random.randint(6, 18)
    
    today = datetime.now()
    submission_date = (today + timedelta(days=submission_days)).strftime('%B %d, %Y')
    inquiry_date = (today + timedelta(days=inquiry_days)).strftime('%B %d, %Y')
    project_start = (today + timedelta(days=project_start_days)).strftime('%B %d, %Y')
    
    prompt = f"""
Generate a concise RFQ for a {project_type} project named '{project_name}' in a {location_type} at {location}.
Industry: {industry}. Context: {backstory}.
Complexity: {complexity_level}/5.
Scope of Work: {sow_text}

Include:
1. Qualifications: 3+ years in {industry}, proven regulatory compliance.
2. Proposal: Technical designs (1-2 pages) and cost breakdown.
3. Criteria: Technical (50%), Cost (30%), Experience (20%).
4. Dates: Release {today.strftime('%B %d, %Y')}, Questions {inquiry_date}, Due {submission_date}, Start {project_start}, Duration {project_duration_months} months.
5. Contract: {'Fixed Price' if complexity_level <= 2 else 'Time & Materials'}.
Email: procurement@{industry.lower().replace(' ', '')}.com. Keep it 1 page.
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"RFQ generation failed: {e}")
        return "RFQ unavailable."

def clean_text(text):
    return "\n".join(line.strip() for line in text.split("\n") if line.strip()).replace('" "', "")

def remove_markdown(text):
    """Remove Markdown formatting like **bold** from text."""
    # Replace **text** with text (removes bold markdown)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    # Replace ## text with text (removes heading markdown)
    text = re.sub(r'##\s*(.*?)(?:\n|$)', r'\1\n', text)
    return text

def sanitize_text(text):
    """Sanitize text to replace unsupported characters for FPDF."""
    # First remove any markdown formatting
    text = remove_markdown(text)
    # Then handle character encoding
    return text.encode("latin-1", "replace").decode("latin-1")

def generate_contact_info(company_name):
    """Generate random contact information based on company name."""
    name = fake.name()
    title = random.choice(["Procurement Manager", "Project Director", "RFP Coordinator", 
                          "Engineering Manager", "Technical Director", "Contracts Administrator"])
    phone = fake.phone_number()
    
    # Create company email domain from company name
    domain = company_name.lower().replace(" ", "").replace(",", "").replace(".", "") + ".com"
    first_name = name.split()[0].lower()
    email = f"{first_name}@{domain}"
    
    return {
        "name": name,
        "title": title,
        "phone": phone,
        "email": email
    }

def generate_pdf(company, project_name, project_type, location, location_type, industry, 
                 project_value, selected_disciplines, applicable_regulations, sow_text, 
                 rfq_text, output_file, complexity):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)  # Increased margin to avoid tight spacing
    pdf.add_page()
    
    # Generate contact information
    contact_info = generate_contact_info(company)
    
    # Header
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 7, sanitize_text("REQUEST FOR PROPOSAL (RFP)"), ln=True, align='C')
    pdf.set_font("Arial", '', 9)
    pdf.cell(0, 5, sanitize_text(company), ln=True, align='C')
    
    # Project Overview
    pdf.ln(2)  # Add consistent spacing
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(0, 6, sanitize_text("PROJECT OVERVIEW"), ln=True)
    pdf.set_font("Arial", '', 8)
    pdf.multi_cell(0, 4, sanitize_text(f"Name: {project_name}\nType: {project_type}\nLocation: {location} ({location_type})\nIndustry: {industry}\nValue: ${project_value:,.0f}\nComplexity: {complexity}/5\nDate: {datetime.now().strftime('%B %d, %Y')}"))
    pdf.multi_cell(0, 4, sanitize_text(f"Disciplines: {', '.join(selected_disciplines)}\nRegulations: {', '.join(applicable_regulations)}"))
    
    # Scope of Work
    pdf.ln(3)  # Consistent spacing
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(0, 6, sanitize_text("SCOPE OF WORK"), ln=True)
    pdf.set_font("Arial", '', 8)
    # Process and split the sow_text to properly format each section and subsection
    sow_lines = sow_text.split('\n')
    for line in sow_lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if line seems to be a header (all caps or starts with "I.", "II.", etc.)
        if line.isupper() or re.match(r'^[IVX]+\.', line) or ":" in line[:20]:
            pdf.ln(2)
            pdf.set_font("Arial", 'B', 8)  # Make headers bold
            pdf.multi_cell(0, 4, sanitize_text(line))
            pdf.set_font("Arial", '', 8)  # Reset to normal
        else:
            pdf.multi_cell(0, 4, sanitize_text(line))
    
    # RFQ
    pdf.add_page()
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(0, 6, sanitize_text("REQUEST FOR QUOTATION"), ln=True)
    pdf.set_font("Arial", '', 8)
    
    # Process and split the rfq_text to properly format each section
    rfq_lines = rfq_text.split('\n')
    for line in rfq_lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if line seems to be a header
        if line.isupper() or re.match(r'^\d+\.', line) or ":" in line[:20]:
            pdf.ln(2)
            pdf.set_font("Arial", 'B', 8)  # Make headers bold
            pdf.multi_cell(0, 4, sanitize_text(line))
            pdf.set_font("Arial", '', 8)  # Reset to normal
        else:
            pdf.multi_cell(0, 4, sanitize_text(line))
    
    # Contact Section
    pdf.ln(4)  # Slightly more space before contact section
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(0, 6, sanitize_text("CONTACT"), ln=True)
    pdf.set_font("Arial", '', 8)
    contact_text = f"{contact_info['name']}, {contact_info['title']}\n"
    contact_text += f"Phone: {contact_info['phone']}\n"
    contact_text += f"Email: {contact_info['email']}"
    pdf.multi_cell(0, 4, sanitize_text(contact_text))
    
    # Timeline Section
    pdf.ln(4)
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(0, 6, sanitize_text("TIMELINE"), ln=True)
    pdf.set_font("Arial", '', 8)
    pdf.multi_cell(0, 4, sanitize_text("Include key dates such as submission deadlines, inquiry deadlines, and project start dates."))
    
    # Footer
    pdf.set_y(-20)  # Move higher to avoid being too close to bottom
    pdf.set_font("Arial", 'I', 7)
    pdf.cell(0, 4, sanitize_text(f"Ref: {datetime.now().strftime('%Y%m%d')}-{project_name[:5].upper()} | CONFIDENTIAL"), ln=True, align='C')
    
    pdf.output(output_file)
    print(f"Generated: {output_file}")

def generate_rfp_dataset(num_rfps=100, output_dir="rfp_dataset", delay_seconds=60):
    """
    Generate a set of RFP PDFs with metadata files.
    
    Args:
        num_rfps: Number of RFPs to generate
        output_dir: Directory to save the generated files
        delay_seconds: Delay between API calls in seconds to avoid rate limiting
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Starting generation of {num_rfps} RFP documents with {delay_seconds}s delay between each...")
    
    for i in range(1, num_rfps + 1):
        print(f"Generating RFP {i} of {num_rfps}...")
        
        industry = random.choice(list(industry_mappings.keys()))
        mapping = industry_mappings[industry]
        company_name = fake.company()
        project_type = random.choice(mapping["project_types"])
        location_type = random.choice(location_types)
        location = f"{fake.city()}, {fake.state_abbr()}"
        project_name = generate_project_name(industry, project_type, location)
        backstory = generate_project_backstory(industry, project_type, location_type)
        
        complexity = random.randint(1, 3)  # Simpler for brevity
        project_value = random.randint(500_000, 10_000_000) if complexity <= 2 else random.randint(5_000_000, 20_000_000)
        
        selected_disciplines = select_random_subset(mapping["disciplines"], min_count=2, max_count=3)
        applicable_regulations = select_random_subset(mapping["regulations"], min_count=1, max_count=2)
        
        # Generate SOW and RFQ with API calls
        sow = generate_varied_sow(selected_disciplines, complexity, industry, applicable_regulations, backstory)
        rfq = generate_rfq(project_name, project_type, sow, location_type, complexity, industry, backstory, location)
        
        # Generate filenames with leading zeros for better sorting
        output_name = f"{output_dir}/rfp_{i:03d}_{project_name.lower().replace(' ', '_')}.pdf"
        metadata_file = f"{output_dir}/rfp_{i:03d}_{project_name.lower().replace(' ', '_')}_metadata.json"
        
        # Generate the PDF
        generate_pdf(company_name, project_name, project_type, location, location_type, industry, 
                     project_value, selected_disciplines, applicable_regulations, sow, rfq, output_name, complexity)
        
        # Generate metadata for RAG
        metadata = {
            "project_name": project_name,
            "company": company_name,
            "industry": industry,
            "complexity": complexity,
            "project_type": project_type,
            "location": location,
            "location_type": location_type,
            "project_value": project_value,
            "disciplines": selected_disciplines,
            "regulations": applicable_regulations,
            "creation_date": datetime.now().strftime('%Y-%m-%d'),
        }
        
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Generated: {output_name} and {metadata_file}")
        
        # Wait before the next generation to avoid rate limiting
        # Skip delay after the last one
        if i < num_rfps:
            print(f"Waiting {delay_seconds} seconds before generating next RFP...")
            time.sleep(delay_seconds)
    
    print(f"Successfully generated {num_rfps} RFPs with metadata in '{output_dir}'")

if __name__ == "__main__":
    generate_rfp_dataset(num_rfps=100, output_dir="rfp_dataset", delay_seconds=30)