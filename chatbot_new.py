import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import shutil
import atexit
from datetime import datetime
import logging
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_core.callbacks import StdOutCallbackHandler
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

# Initialize conversation memory in session state
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )

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
        return ChatOpenAI(model=LLM_MODEL, temperature=0.7, api_key=os.getenv("OPENAI_API_KEY"))
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

def add_metadata(doc, doc_type):
    """Add metadata to document."""
    doc.metadata["doc_type"] = doc_type
    return doc

@st.cache_data
def load_and_prepare_data():
    """Load data from Excel and prepare LangChain documents with each row as one vector."""
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
                    
                    # Process certificate data - each row becomes one complete document
                    processed_count = 0
                    for index, row in df.iterrows():
                        if pd.isna(row.get("Certificate_Name")) or str(row.get("Certificate_Name")).strip() == "":
                            continue

                        # Create comprehensive content for each certificate (one row = one vector)
                        content_parts = []
                        
                        # Add certificate name
                        certificate_name = str(row.get("Certificate_Name", "")).strip()
                        content_parts.append(f"Certificate Name: {certificate_name}")
                        
                        # Add all available fields from the row
                        for column in df.columns:
                            if column != "Certificate_Name" and pd.notna(row.get(column)):
                                value = str(row.get(column, "")).strip()
                                if value:
                                    field_name = column.replace('_', ' ').title()
                                    content_parts.append(f"{field_name}: {value}")
                        
                        # Join all content for this certificate
                        content = "\n".join(content_parts)
                        
                        # Get download link
                        download_link = ""
                        if pd.notna(row.get("Certificate_Download_Link")):
                            download_link = str(row.get("Certificate_Download_Link")).strip()
                        elif pd.notna(row.get("Dummy_Download_Link")):
                            download_link = str(row.get("Dummy_Download_Link")).strip()
                        
                        # Create metadata
                        metadata = {
                            "certificate_name": certificate_name,
                            "download_link": download_link,
                            "doc_type": "certificate",
                            "source": "excel_data"
                        }
                        
                        # Add summary to metadata if available
                        if pd.notna(row.get("Summary")):
                            metadata["summary"] = str(row.get("Summary")).strip()
                        
                        # Create document and add metadata
                        doc = Document(page_content=content, metadata=metadata)
                        doc = add_metadata(doc, "certificate")
                        documents.append(doc)
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

    return documents, df

@st.cache_resource
def initialize_vectorstore_and_chain(_documents):
    """Initialize Chroma vectorstore and ConversationalRetrievalChain."""
    try:
        embeddings = initialize_embeddings()
        llm = initialize_llm()
        
        # Track embedding usage
        total_text_length = sum(len(doc.page_content) for doc in _documents)
        usage_tracker.track_embedding_usage(total_text_length)
        
        # Text splitter for chunking if needed
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(_documents)
        
        st.info(f"📊 Created {len(chunks)} chunks from {len(_documents)} documents")
        
        # Create vectorstore
        if os.path.exists(CHROMA_PERSIST_DIR):
            try:
                # Try to delete existing collection
                temp_store = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
                temp_store.delete_collection()
            except:
                # If deletion fails, remove directory
                shutil.rmtree(CHROMA_PERSIST_DIR)
        
        vectorstore = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings, 
            persist_directory=CHROMA_PERSIST_DIR
        )
        
        # Create custom prompt template
        template = """
        You are a helpful AI assistant for YOTTA Data Services that answers questions about certificates, policies, and company information.
        Answer the question using only the provided context and take as much context as possible.
        Keep your answers concise, accurate, and professional.
        
        If the context does not contain the answer, politely reply:
        "I don't have information about that in my knowledge base. Please contact the relevant team for more details."
        
        Context:
        {context}
        
        Chat History:
        {chat_history}
        
        Question: {question}
        
        Answer:
        """
        
        QA_PROMPT = PromptTemplate.from_template(template)
        
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8, "fetch_k": 20}
        )
        
        # Create ConversationalRetrievalChain
        qa_chain = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(model=LLM_MODEL, temperature=0),
            retriever=retriever,
            memory=st.session_state.memory,
            callbacks=[StdOutCallbackHandler()],
            combine_docs_chain_kwargs={"prompt": QA_PROMPT}
        )
        
        return vectorstore, qa_chain
        
    except Exception as e:
        logging.error(f"Failed to initialize vectorstore and chain: {str(e)}")
        st.error(f"❌ Failed to initialize vectorstore and chain: {str(e)}")
        st.stop()

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

def handle_specific_queries(user_input, qa_chain, data_df):
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
    
    # Use ConversationalRetrievalChain for all other queries
    try:
        result = qa_chain.invoke({"question": user_input})
        response = result["answer"]
        
        # Track usage
        usage_tracker.track_completion_usage(user_input, response)
        
        # If response is too short or generic, provide helpful fallback
        if not response or len(response.strip()) < 20:
            return "I don't have specific information about that in my knowledge base. Please contact the relevant team for more details."
        
        return response.strip()
        
    except Exception as e:
        logging.error(f"Error in QA chain processing: {str(e)}")
        return "I encountered an issue processing your request. Please try rephrasing your question or contact the relevant team for assistance."

def format_response_with_clickable_link(response):
    """Format response to make download links clickable."""
    # Look for URLs in the response and make them clickable
    url_pattern = r'https?://[^\s\])]+'
    urls = re.findall(url_pattern, response)
    
    for url in urls:
        clickable_url = f"[{url}]({url})"
        response = response.replace(url, clickable_url)
    
    return response

def display_usage_metrics():
    """Display OpenAI usage metrics in sidebar."""
    usage = st.session_state.openai_usage
    
    st.markdown("### 📊 OpenAI Usage")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("📤 Embedding Tokens", f"{usage['embedding_tokens']:,}")
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
        vectorstore, qa_chain = initialize_vectorstore_and_chain(documents)

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
            # Reset memory
            st.session_state.memory = ConversationBufferMemory(
                memory_key="chat_history", 
                return_messages=True
            )
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
            response = handle_specific_queries(user_input, qa_chain, data_df)
            
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