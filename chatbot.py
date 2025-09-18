import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import shutil
import atexit
from datetime import datetime
import logging
import json
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import re

# Configure logging
logging.basicConfig(
    filename="yotta_chatbot.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize session state for usage tracking
if "openai_usage" not in st.session_state:
    st.session_state.openai_usage = {
        "embedding_tokens": 0,
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "total_requests": 0
    }

# Constants - Dynamic file path detection
def get_excel_file_path():
    """Dynamically find the Excel file path."""
    possible_paths = [
        os.getenv("EXCEL_FILE_PATH"),
        "Certificate_Details_Chatbot_with_Dummy_Summary (1).xlsx",
        "./Certificate_Details_Chatbot_with_Dummy_Summary (1).xlsx",
        os.path.join(os.getcwd(), "Certificate_Details_Chatbot_with_Dummy_Summary (1).xlsx"),
        r"C:\Users\rajpu\Desktop\AI-YOTTA-CHATBOT\Certificate_Details_Chatbot_with_Dummy_Summary (1).xlsx"
    ]
    
    for path in possible_paths:
        if path and os.path.exists(path):
            return path
    
    return None

EXCEL_FILE_PATH = get_excel_file_path()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHROMA_PERSIST_DIR = "./chroma_db"
DATA_DIR = "./data"

# Cleanup function to remove generated directories
def cleanup_directories():
    """Remove generated directories when the app stops."""
    try:
        if os.path.exists(CHROMA_PERSIST_DIR):
            shutil.rmtree(CHROMA_PERSIST_DIR)
            logging.info("Removed chroma_db directory")
        if os.path.exists(DATA_DIR):
            shutil.rmtree(DATA_DIR)
            logging.info("Removed data directory")
    except Exception as e:
        logging.error(f"Error during cleanup: {str(e)}")

# Register cleanup function
atexit.register(cleanup_directories)

def get_comprehensive_knowledge_base():
    """Return comprehensive built-in knowledge base with download links."""
    return {
        "addresses": """
YOTTA Data Services Private Limited Office Locations:
- Head Office (HO): 5th Floor, Scorpio Building, Hiranandani Gardens, Powai, Mumbai, Maharashtra 400 076, India
- Datacentre (NM1): Edinberg Building, Survey No. 30, Persipina Developers Pvt Ltd, Bhokarpada Village, Panvel, Raigad, Maharashtra 410 206, India
- Pi Data centre (AP1): C/O Pi Data centre's Pvt Ltd., Survey No. 49/P, Plot No.12, IT Park, Mangalagiri, Guntur District, Andhra Pradesh 522 503, India
- Datacentre (D1): Plot No. 7, Sector Knowledge Park V, Greater Noida, Gautam Buddha Nagar, Uttar Pradesh 201 306, India
- Datacentre (TB2): Unit No 204, C Wing, 2nd Floor, Reliable Tech Park, Gut No 31, Village - Elthan, Airoli, Navi Mumbai, Maharashtra 400 708, India
- GDC Data Centre: 12th Floor, 1202, Signature Building, 13B, Zone 1, GIFT Road, GIFT City, Gandhinagar, Gujarat 382355, India
""",
        "policies": {
            "data_retention": {
                "summary": "YOTTA's data retention policy covers data classification, retention periods, and secure disposal procedures.",
                "contact": "GRC team",
                "download_link": "https://internal.yotta.com/policies/data-retention-policy.pdf"
            },
            "sustainability": {
                "summary": "YOTTA demonstrates sustainability commitment through LEED certifications and ISO 50001 energy management.",
                "contact": "GRC team", 
                "download_link": "https://internal.yotta.com/policies/sustainability-policy.pdf"
            },
            "travel": {
                "summary": "YOTTA's travel policy covers employee business travel, expense reimbursement, and approval processes.",
                "contact": "HR team",
                "download_link": "https://internal.yotta.com/hr/travel-policy.pdf"
            },
            "leave": {
                "summary": "YOTTA's leave policy covers employee leave entitlements, approval processes, and leave management procedures.",
                "contact": "HR team",
                "download_link": "https://internal.yotta.com/hr/leave-policy.pdf"
            },
            "maintenance": {
                "summary": "YOTTA maintains comprehensive data center maintenance procedures including preventive maintenance and emergency protocols.",
                "contact": "Operations team",
                "download_link": "https://internal.yotta.com/operations/maintenance-procedures.pdf"
            },
            "information_security": {
                "summary": "YOTTA has robust information security policies aligned with ISO 27001, ISO 27017, ISO 27018, and ISO 27701 standards.",
                "contact": "GRC team",
                "download_link": "https://internal.yotta.com/policies/information-security-policy.pdf"
            },
            "cobc_anti_bribery": {
                "summary": "YOTTA has established Code of Business Conduct and Anti-Bribery policies for ethical business practices.",
                "contact": "Legal team",
                "download_link": "https://internal.yotta.com/legal/cobc-anti-bribery-policy.pdf"
            },
            "document_sharing": {
                "summary": "YOTTA has document sharing guidelines for customer and prospect interactions with distribution guidelines.",
                "contact": "GRC team",
                "download_link": "https://internal.yotta.com/policies/document-sharing-guidelines.pdf"
            }
        },
        "compliance": {
            "gdpr": {
                "summary": "YOTTA is GDPR compliant with ISO 27701 certification for privacy information management.",
                "contact": "GRC team",
                "download_link": "https://internal.yotta.com/compliance/gdpr-compliance-report.pdf"
            },
            "cert_in": {
                "summary": "YOTTA maintains Cert-In empanelment status for government compliance requirements.",
                "contact": "GRC team",
                "download_link": "Contact GRC team for current empanelment status"
            },
            "it_act": {
                "summary": "YOTTA complies with Indian IT Act requirements including data localization as per RBI frameworks.",
                "contact": "GRC team",
                "download_link": "https://internal.yotta.com/compliance/it-act-compliance.pdf"
            },
            "vendor_assessment": {
                "summary": "YOTTA has comprehensive vendor compliance assessment processes covering security and operational requirements.",
                "contact": "GRC team",
                "download_link": "https://internal.yotta.com/vendor/vendor-assessment-guidelines.pdf"
            },
            "supplier_compliance": {
                "summary": "YOTTA maintains supplier compliance standards aligned with security certifications and operational requirements.",
                "contact": "GRC team",
                "download_link": "https://internal.yotta.com/vendor/supplier-compliance-standards.pdf"
            }
        }
    }

def load_knowledge_base():
    """Load comprehensive knowledge base."""
    return get_comprehensive_knowledge_base()

# Load environment variables
load_dotenv()

def validate_env_vars():
    """Validate required environment variables and file paths."""
    if not os.getenv("OPENAI_API_KEY"):
        logging.error("OPENAI_API_KEY not found in .env file.")
        st.error("❌ OPENAI_API_KEY not found. Please set it in the .env file.")
        st.stop()
    
    if not EXCEL_FILE_PATH:
        logging.error("Excel file not found in any of the expected locations.")
        st.error(f"""
        ❌ **Excel file not found!** 
        
        **Expected locations checked:**
        - Current directory: `Certificate_Details_Chatbot_with_Dummy_Summary (1).xlsx`
        - Your path: `C:\\Users\\rajpu\\Desktop\\AI-YOTTA-CHATBOT\\Certificate_Details_Chatbot_with_Dummy_Summary (1).xlsx`
        - Environment variable: `EXCEL_FILE_PATH`
        
        **Solutions:**
        1. **Copy the Excel file** to the same folder as your Python script
        2. **Or** update the path in the code 
        3. **Or** set `EXCEL_FILE_PATH` in your `.env` file
        
        **Current working directory:** `{os.getcwd()}`
        """)
        st.stop()
    
    st.success(f"✅ Excel file found at: `{EXCEL_FILE_PATH}`")
    return True

class OpenAIUsageTracker:
    """Track OpenAI API usage."""
    
    def __init__(self):
        self.session_state = st.session_state.openai_usage
    
    def track_embedding_usage(self, text_length):
        """Estimate embedding token usage."""
        # Rough estimation: ~4 characters per token
        estimated_tokens = text_length // 4
        self.session_state["embedding_tokens"] += estimated_tokens
        self.session_state["total_requests"] += 1
    
    def track_completion_usage(self, prompt_text, response_text):
        """Estimate completion token usage."""
        prompt_tokens = len(prompt_text) // 4
        completion_tokens = len(response_text) // 4
        self.session_state["prompt_tokens"] += prompt_tokens
        self.session_state["completion_tokens"] += completion_tokens
        self.session_state["total_requests"] += 1

usage_tracker = OpenAIUsageTracker()

@st.cache_resource
def initialize_llm():
    """Initialize the ChatOpenAI model."""
    try:
        return ChatOpenAI(model=LLM_MODEL, temperature=0.1, api_key=os.getenv("OPENAI_API_KEY"))
    except Exception as e:
        logging.error(f"Failed to initialize LLM: {str(e)}")
        st.error(f"❌ Failed to initialize language model: {str(e)}")
        st.stop()

@st.cache_resource
def initialize_embeddings():
    """Initialize the OpenAI embeddings model."""
    try:
        return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=os.getenv("OPENAI_API_KEY"))
    except Exception as e:
        logging.error(f"Failed to initialize embeddings: {str(e)}")
        st.error(f"❌ Failed to initialize embeddings: {str(e)}")
        st.stop()

@st.cache_data
def load_and_prepare_data():
    """Load data from Excel and prepare LangChain documents."""
    documents = []
    df = None
    
    try:
        if EXCEL_FILE_PATH and os.path.exists(EXCEL_FILE_PATH):
            st.info(f"📁 Loading data from: `{os.path.basename(EXCEL_FILE_PATH)}`")
            
            try:
                if EXCEL_FILE_PATH.endswith(".xlsx"):
                    df = pd.read_excel(EXCEL_FILE_PATH, engine="openpyxl")
                elif EXCEL_FILE_PATH.endswith(".csv"):
                    df = pd.read_csv(EXCEL_FILE_PATH)
                
                if df is not None and not df.empty:
                    st.success(f"✅ Successfully loaded {len(df)} rows of certificate data")
                    
                    # Process certificate data
                    processed_count = 0
                    for index, row in df.iterrows():
                        if pd.isna(row.get("Certificate_Name")) or str(row.get("Certificate_Name")).strip() == "":
                            continue

                        certificate_name = str(row.get("Certificate_Name", "")).strip()
                        summary = str(row.get("Summary", "")) if pd.notna(row.get("Summary")) else ""
                        
                        # Create content
                        content_parts = [f"Certificate Name: {certificate_name}"]
                        if summary.strip():
                            content_parts.append(f"Summary: {summary.strip()}")
                        
                        # Add other fields
                        for field in ["Scope", "Certificate_Category", "Certification_Body"]:
                            value = str(row.get(field, "")) if pd.notna(row.get(field)) else ""
                            if value.strip():
                                content_parts.append(f"{field.replace('_', ' ')}: {value.strip()}")
                        
                        content = "\n".join(content_parts)
                        
                        # Get download link
                        download_link = ""
                        if pd.notna(row.get("Certificate_Download_Link")):
                            download_link = str(row.get("Certificate_Download_Link")).strip()
                        elif pd.notna(row.get("Dummy_Download_Link")):
                            download_link = str(row.get("Dummy_Download_Link")).strip()
                        
                        metadata = {
                            "certificate_name": certificate_name,
                            "summary": summary,
                            "download_link": download_link,
                            "type": "certificate"
                        }
                        
                        documents.append(Document(page_content=content, metadata=metadata))
                        processed_count += 1
                    
                    st.success(f"✅ Successfully processed {processed_count} certificates from Excel file")
                else:
                    st.error("❌ Excel file is empty or couldn't be read properly")
                    return [], None
                    
            except Exception as e:
                st.error(f"❌ Error reading Excel file: {str(e)}")
                logging.error(f"Excel reading error: {str(e)}")
                return [], None
        
        else:
            st.error("❌ Excel file not found or path is None")
            return [], None
    
    except Exception as e:
        logging.error(f"Error loading certificate data: {str(e)}")
        st.error(f"❌ Critical error loading certificate data: {str(e)}")
        return [], None

    # Load comprehensive knowledge base
    knowledge_base = load_knowledge_base()
    
    # Add address information with multiple variations for better matching
    address_content = f"""
Office Locations: {knowledge_base['addresses']}
YOTTA office locations: {knowledge_base['addresses']}
Where is YOTTA located: {knowledge_base['addresses']}
YOTTA addresses: {knowledge_base['addresses']}
"""
    documents.append(Document(
        page_content=address_content,
        metadata={"type": "address", "category": "office_locations"}
    ))
    
    # Add comprehensive policy information with download links
    for policy_type, policy_info in knowledge_base["policies"].items():
        policy_content = f"""
Policy: {policy_type.replace('_', ' ').title()}
Summary: {policy_info['summary']}
Contact: {policy_info['contact']}
Download Link: {policy_info['download_link']}
Policy Type: {policy_type}
Document search assistance: {policy_info['summary']}
"""
        documents.append(Document(
            page_content=policy_content,
            metadata={"type": "policy", "policy_type": policy_type, "summary": policy_info['summary'], "download_link": policy_info['download_link']}
        ))
    
    # Add comprehensive compliance information with download links
    for compliance_type, compliance_info in knowledge_base["compliance"].items():
        compliance_content = f"""
Compliance: {compliance_type.replace('_', ' ').title()}
Summary: {compliance_info['summary']}
Contact: {compliance_info['contact']}
Download Link: {compliance_info['download_link']}
Compliance information assistance: {compliance_info['summary']}
Vendor compliance: {compliance_info['summary']}
Supplier assessment: {compliance_info['summary']}
"""
        documents.append(Document(
            page_content=compliance_content,
            metadata={"type": "compliance", "compliance_type": compliance_type, "summary": compliance_info['summary'], "download_link": compliance_info['download_link']}
        ))
    
    # Add contact information with simple format
    contact_content = f"""
GRC Team: Handles compliance, certifications, and policy-related queries
Operations Team: Manages data center operations and maintenance procedures  
HR Team: Handles employee policies, travel policies, and HR procedures
Contact Information: Contact respective teams through internal channels
"""
    documents.append(Document(
        page_content=contact_content,
        metadata={"type": "contact"}
    ))

    return documents, df

@st.cache_resource
def initialize_vectorstore(_documents):
    """Initialize Chroma vectorstore with documents."""
    try:
        embeddings = initialize_embeddings()
        
        # Track embedding usage
        total_text_length = sum(len(doc.page_content) for doc in _documents)
        usage_tracker.track_embedding_usage(total_text_length)
        
        vectorstore = Chroma.from_documents(
            documents=_documents,
            embedding=embeddings,
            collection_name="yotta_knowledge",
            persist_directory=CHROMA_PERSIST_DIR
        )
        return vectorstore
    except Exception as e:
        logging.error(f"Failed to initialize vectorstore: {str(e)}")
        st.error(f"❌ Failed to initialize vectorstore: {str(e)}")
        st.stop()

def setup_rag_chain(vectorstore):
    """Setup RAG chain with improved prompt."""
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # Use MMR for better diversity
        search_kwargs={"k": 15, "fetch_k": 20}  # Increase retrieval
    )
    
    prompt_template = """
    You are YOTTA's knowledgeable assistant. ALWAYS respond in the simple Summary + Download format.

    STRICT FORMAT RULES - Use this format for ALL queries:
    Summary: [brief summary of the information/policy/compliance status]
    Download: [document link if available, otherwise "Contact [relevant team] for document access"]

    EXAMPLES:
    - Certificate query: "Summary: ISO 27001 covers information security management. Download: [link]"
    - Policy query: "Summary: Data retention policy covers data classification and disposal. Download: Contact GRC team for document access"  
    - Compliance query: "Summary: YOTTA is GDPR compliant with ISO 27701 certification. Download: Contact GRC team for compliance documents"
    - Vendor/Supplier query: "Summary: YOTTA has robust security policies for vendor assessment. Download: Contact GRC team for vendor compliance documents"

    IMPORTANT:
    - Keep summaries brief (1-2 sentences max)
    - Always include Download link
    - For policies/compliance: use "Contact [team] for document access" if no direct link
    - NEVER provide long explanations or lists

    Question: {question}
    
    Context:
    {context}
    
    Answer:
    """
    
    prompt = PromptTemplate.from_template(prompt_template)
    llm = initialize_llm()

    def format_docs(docs):
        formatted = []
        for doc in docs:
            doc_type = doc.metadata.get('type', 'unknown')
            formatted.append(f"[{doc_type.upper()}] {doc.page_content}\n---")
        return "\n".join(formatted)

    def rag_chain_func(question):
        """Complete RAG chain function."""
        try:
            # Retrieve documents
            docs = retriever.invoke(question)
            context = format_docs(docs)
            
            # Create the prompt
            formatted_prompt = prompt.format(context=context, question=question)
            
            # Get response from LLM
            response = llm.invoke(formatted_prompt)
            
            # Track usage
            usage_tracker.track_completion_usage(formatted_prompt, response.content)
            
            return response.content
            
        except Exception as e:
            logging.error(f"Error in RAG chain: {str(e)}")
            return "I encountered an issue processing your request. Please try rephrasing your question or contact the GRC team for assistance."

    return rag_chain_func

def is_greeting_only(query):
    """Check if the query is ONLY a greeting without any specific request."""
    query_lower = query.lower().strip()
    
    # Simple greeting patterns that don't ask for anything
    simple_greetings = [
        "hello", "hi", "hey", "good morning", "good afternoon", 
        "good evening", "hi there", "hello there"
    ]
    
    # Check if it's ONLY a greeting (no other content)
    words = query_lower.split()
    if len(words) <= 2 and any(greeting in query_lower for greeting in simple_greetings):
        return True
    
    return False

def handle_specific_queries(user_input, rag_chain, data_df):
    """Handle queries with improved logic and comprehensive fallbacks."""
    query_lower = user_input.lower()
    
    # Only respond with greeting if it's JUST a greeting
    if is_greeting_only(user_input):
        return "Hello! 👋 I'm YOTTA's certificate assistant. I can help you find information about YOTTA's certificates, policies, compliance, and company information. What can I help you with today?"
    
    # Handle list all certificates request
    if ("list" in query_lower and ("all" in query_lower or "certificates" in query_lower)) or \
       ("show" in query_lower and "certificates" in query_lower) or \
       ("what certificates" in query_lower):
        if data_df is not None and not data_df.empty:
            certs = data_df["Certificate_Name"].dropna().unique()
            if len(certs) > 0:
                cert_list = "\n".join([f"• **{cert}**" for cert in sorted(certs)])
                return f"📜 **Complete List of YOTTA Certificates:**\n\n{cert_list}\n\n*For detailed information about any certificate, just ask!*"
    
    # Handle specific certificate queries (SOC, PCI-DSS, HIPAA, CMM, etc.)
    if any(cert in query_lower for cert in ["soc", "pci-dss", "pci dss", "hipaa", "cmm", "iso"]):
        # Check what certificates YOTTA actually has from the list
        if data_df is not None and not data_df.empty:
            found_certs = []
            cert_names = data_df["Certificate_Name"].dropna().str.lower()
            
            # Check for specific certificates mentioned in the query
            if "soc" in query_lower:
                soc_certs = cert_names[cert_names.str.contains("soc", na=False)].tolist()
                if soc_certs:
                    found_certs.extend([name.title() for name in soc_certs])
            
            if any(pci in query_lower for pci in ["pci-dss", "pci dss"]):
                pci_certs = cert_names[cert_names.str.contains("pci", na=False)].tolist()
                if pci_certs:
                    found_certs.extend([name.title() for name in pci_certs])
            
            if "iso" in query_lower:
                iso_certs = cert_names[cert_names.str.contains("iso", na=False)].tolist()
                if iso_certs:
                    found_certs.extend([name.title() for name in iso_certs])
            
            if found_certs:
                cert_response = "Summary: YOTTA has the following certificates: " + ", ".join(set(found_certs)) + "."
                if "hipaa" in query_lower or "cmm" in query_lower:
                    cert_response += " HIPAA and CMM certifications are not currently held."
                cert_response += "\nDownload: Contact GRC team for certificate documents and details"
                return cert_response
    
    # Use RAG chain for all other queries
    try:
        response = rag_chain(user_input)
        
        # If response is too short or generic, provide helpful fallback
        if not response or len(response.strip()) < 20:
            return "Summary: Information not available in current knowledge base.\nDownload: Contact GRC team for specific details or check internal portal"
        
        return response.strip()
        
    except Exception as e:
        logging.error(f"Error in RAG processing: {str(e)}")
        return "Summary: Unable to process request due to technical issue.\nDownload: Contact GRC team for assistance"

def format_response_with_clickable_link(response):
    """Format response to make download links clickable."""
    if "Download:" in response:
        lines = response.split('\n')
        formatted_lines = []
        
        for line in lines:
            if line.startswith("Download:") and "http" in line:
                # Extract URL and make it clickable
                url_match = re.search(r'https?://[^\s\]]+', line)
                if url_match:
                    url = url_match.group(0)
                    formatted_lines.append(f"Download: [{url}]({url})")
                else:
                    formatted_lines.append(line)
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    return response

def display_usage_metrics():
    """Display OpenAI usage metrics in sidebar."""
    usage = st.session_state.openai_usage
    
    st.markdown("### 📊 OpenAI Usage")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("🔤 Embedding Tokens", f"{usage['embedding_tokens']:,}")
        st.metric("📝 Prompt Tokens", f"{usage['prompt_tokens']:,}")
    
    with col2:
        st.metric("🤖 Completion Tokens", f"{usage['completion_tokens']:,}")
        st.metric("📞 Total Requests", usage['total_requests'])
    
    # Estimated cost (rough approximation)
    embedding_cost = usage['embedding_tokens'] * 0.0001 / 1000  # $0.0001 per 1K tokens
    completion_cost = (usage['prompt_tokens'] + usage['completion_tokens']) * 0.002 / 1000  # $0.002 per 1K tokens for gpt-4o-mini
    total_cost = embedding_cost + completion_cost
    
    st.metric("💰 Estimated Cost", f"${total_cost:.4f}")

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="YOTTA Knowledge Assistant", 
        page_icon="🏢", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 15px; color: white; text-align: center; 
        margin-bottom: 2rem; box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .chat-message {
        padding: 1rem; border-radius: 10px; margin-bottom: 1rem; border-left: 4px solid #ffffff;
    }
    .user-message { background-color: #000000; }
    .bot-message { background-color: #000000; }
    </style>
    """, unsafe_allow_html=True)

    # Validate environment
    validate_env_vars()

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏢 YOTTA Knowledge Assistant</h1>
        <p style="font-size: 1.2em; margin-bottom: 0;">
            Your intelligent assistant for certificates, compliance, policies, and company information
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load data and initialize system
    with st.spinner("🔄 Initializing YOTTA Knowledge Base..."):
        documents, data_df = load_and_prepare_data()
        vectorstore = initialize_vectorstore(documents)
        rag_chain = setup_rag_chain(vectorstore)

    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 What I Can Help With")
        st.markdown("""
        **📜 Certificates:** Certificate details, summaries, download links
        **🏢 Company Info:** Office locations, team contacts  
        **📋 Policies:** Data retention, sustainability, travel, leave policies
        **🛡️ Compliance:** GDPR, Cert-In, IT Act compliance status
        **🤝 Vendor Queries:** Security policies, compliance info
        """)
        
        # Usage metrics
        display_usage_metrics()
        
        # System status
        st.markdown("### ⚙️ System Status")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 Documents", len(documents))
        with col2:
            cert_count = len(data_df) if data_df is not None else 0
            st.metric("📜 Certificates", cert_count)
            
        st.success(f"✅ {LLM_MODEL}")
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Main content area
    st.markdown("### 💬 Chat Interface")
    
    # Display chat messages
    for message in st.session_state.messages:
        css_class = "user-message" if message["role"] == "user" else "bot-message"
        icon = "👤" if message["role"] == "user" else "🤖"
        
        content = message["content"]
        if message["role"] == "assistant":
            content = format_response_with_clickable_link(content)
        
        st.markdown(f"""
        <div class="chat-message {css_class}">
            <strong>{icon} {"You" if message["role"] == "user" else "YOTTA Assistant"}:</strong><br>
            {content}<br>
            <small style="color: #666;">⏰ {message["timestamp"]}</small>
        </div>
        """, unsafe_allow_html=True)

    # Chat input
    user_input = st.chat_input("Ask about certificates, policies, compliance, or anything YOTTA-related...")
    
    if user_input:
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({
            "role": "user", 
            "content": user_input, 
            "timestamp": timestamp
        })
        
        with st.spinner("🔍 Searching YOTTA knowledge base..."):
            response = handle_specific_queries(user_input, rag_chain, data_df)
            
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response, 
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🏢 <strong>YOTTA Knowledge Assistant</strong> | Powered by AI | 
        For technical support, contact IT Operations</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()