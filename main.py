import streamlit as st
import sqlite3
import pandas as pd
import os
import datetime
import pytz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from deep_translator import GoogleTranslator
from gtts import gTTS
from io import BytesIO
from tavily import TavilyClient
import getpass
from dotenv import load_dotenv
import PyPDF2
import shutil

# Configure Streamlit page
st.set_page_config(
    page_title="University Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Get API keys
try:
    if "GOOGLE_API_KEY" in st.secrets:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    elif "GOOGLE_API_KEY" in os.environ:
        GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
    else:
        st.error("Google Gemini API key not found. Add it to '.streamlit/secrets.toml' or set it as an env variable.")
        st.stop()
except Exception as e:
    st.error(f"Error accessing Google API key: {str(e)}. Check your setup.")
    st.stop()

try:
    if "TAVILY_API_KEY" in st.secrets:
        TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
    elif "TAVILY_API_KEY" in os.environ:
        TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
    else:
        TAVILY_API_KEY = getpass.getpass("Enter TAVILY_API_KEY: ")
        os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
except Exception as e:
    st.error(f"Error accessing Tavily API key: {str(e)}. Check your setup.")
    st.stop()

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize Tavily client
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# Supported languages
LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de"
}

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect("data/chatbot.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS universities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            university TEXT,
            program TEXT,
            tuition INTEGER,
            location TEXT,
            visa_service TEXT
        )
    """)
    conn.commit()
    conn.close()

# Scrape website and save to /scraped_data
def scrape_website(url, keep_old_data):
    try:
        if not url.startswith("http"):
            st.error("Invalid URL. Please include 'http://' or 'https://'.")
            return False
        
        os.makedirs("scraped_data", exist_ok=True)
        if not keep_old_data:
            for file in os.listdir("scraped_data"):
                os.remove(os.path.join("scraped_data", file))
            st.write("Cleared old scraped data.")
        
        with st.spinner(f"Scraping {url}..."):
            crawl_results = tavily_client.crawl(url=url, max_depth=3, extract_depth="advanced")
        
        if not crawl_results["results"]:
            st.error("No data scraped. Check URL or Tavily API limits.")
            return False
        
        for i, result in enumerate(crawl_results["results"]):
            content = result.get("raw_content", "")
            if not content.strip():
                st.warning(f"Page {i+1} has no content. Skipping.")
                continue
            file_path = f"scraped_data/page_{i+1}_{url.replace('https://', '').replace('/', '_')}.txt"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            st.write(f"Saved {len(content)} characters to {file_path}")
        
        st.success(f"Scraped {len(crawl_results['results'])} pages from {url}")
        return True
    except Exception as e:
        st.error(f"Scraping failed: {str(e)}")
        return False

# Load file into appropriate storage
def load_file(file):
    try:
        file_name = file.name
        file_ext = file_name.split('.')[-1].lower()
        
        if file_ext in ['csv', 'xlsx']:
            conn = sqlite3.connect("data/chatbot.db")
            if file_ext == 'csv':
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            
            expected_columns = ['university', 'program', 'tuition', 'location', 'visa_service']
            if not all(col in df.columns for col in expected_columns):
                st.error(f"File {file_name} must have columns: {', '.join(expected_columns)}")
                return False
            
            df.to_sql('universities', conn, if_exists='append', index=False)
            conn.close()
            st.success(f"Loaded {len(df)} rows from {file_name} into universities table.")
        
        elif file_ext == 'sql':
            conn = sqlite3.connect("data/chatbot.db")
            sql_content = file.read().decode('utf-8')
            try:
                conn.executescript(sql_content)
                conn.commit()
                st.success(f"Executed SQL from {file_name}")
            except Exception as e:
                st.error(f"Error executing SQL from {file_name}: {str(e)}")
                return False
            conn.close()
        
        elif file_ext == 'pdf':
            os.makedirs("scraped_data", exist_ok=True)
            timestamp = datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y%m%d_%H%M')
            text_file = f"scraped_data/pdf_upload_{timestamp}.txt"
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            if not text.strip():
                st.error(f"No text extracted from {file_name}")
                return False
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(text)
            st.success(f"Saved PDF text to {text_file}")
        
        else:
            st.error(f"Unsupported file type: {file_ext}")
            return False
        
        return True
    except Exception as e:
        st.error(f"Error loading {file_name}: {str(e)}")
        return False

# Delete file
def delete_file(file_name):
    try:
        file_path = None
        if file_name.startswith('pdf_upload_'):
            file_path = f"scraped_data/{file_name}"
        else:
            file_path = f"data/{file_name}"
        
        if os.path.exists(file_path):
            os.remove(file_path)
            st.success(f"Deleted {file_name}")
            return True
        else:
            st.error(f"File {file_name} not found.")
            return False
    except Exception as e:
        st.error(f"Error deleting {file_name}: {str(e)}")
        return False

# Load and index data for FAISS
def load_and_index_data():
    all_texts = []
    
    # Load university_info.txt
    sample_file = "university_info.txt"
    if os.path.exists(sample_file):
        with open(sample_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                all_texts.append(content)
                st.write(f"Loaded {len(content)} characters from {sample_file}")
    
    # Load scraped_data
    scraped_dir = "scraped_data"
    if os.path.exists(scraped_dir):
        for file_name in os.listdir(scraped_dir):
            if file_name.endswith(".txt"):
                with open(os.path.join(scraped_dir, file_name), "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        all_texts.append(content)
                        st.write(f"Loaded {len(content)} characters from {file_name}")
    
    # Load SQLite universities table
    conn = sqlite3.connect("data/chatbot.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM universities")
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    for row in rows:
        text = " ".join([f"{col}: {val}" for col, val in zip(columns, row)])
        all_texts.append(text)
        st.write(f"Loaded row from universities: {text[:100]}...")
    conn.close()
    
    # Load data folder
    data_dir = "data"
    if os.path.exists(data_dir):
        for file_name in os.listdir(data_dir):
            if file_name.endswith(('.csv', 'xlsx')):
                file_path = os.path.join(data_dir, file_name)
                if file_name.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                for _, row in df.iterrows():
                    text = " ".join([f"{col}: {row[col]}" for col in df.columns])
                    all_texts.append(text)
                    st.write(f"Loaded row from {file_name}: {text[:100]}...")
            elif file_name.endswith('.sql'):
                with open(os.path.join(data_dir, file_name), "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        all_texts.append(content)
                        st.write(f"Loaded SQL from {file_name}: {content[:100]}...")
    
    if not all_texts:
        st.error("No valid data found.")
        return None
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for text in all_texts:
        chunks.extend(text_splitter.split_text(text))
    st.write(f"Created {len(chunks)} chunks for indexing.")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    try:
        if os.path.exists("vectorstore"):
            shutil.rmtree("vectorstore")
            st.write("Cleared old FAISS index.")
        vectorstore = FAISS.from_texts(chunks, embeddings)
        os.makedirs("vectorstore", exist_ok=True)
        vectorstore.save_local("vectorstore")
        st.write("Created new FAISS index.")
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"Failed to create FAISS index: {str(e)}")
        return None

# Generate SQL query using Gemini
def generate_sql_query(query, lang_code):
    if lang_code != "en":
        query = GoogleTranslator(source=lang_code, target="en").translate(query)
    
    conn = sqlite3.connect("data/chatbot.db")
    cursor = conn.cursor()
    
    cursor.execute("PRAGMA table_info(universities);")
    uni_schema = cursor.fetchall()
    uni_schema_str = "\n".join([f"{col[1]} ({col[2]})" for col in uni_schema])
    
    conn.close()
    
    prompt = f"""
You are an expert SQL assistant for a university and visa chatbot. Convert the user's natural language query into a valid SQLite query based on the following table schema:

**universities table**:
{uni_schema_str}

Rules:
1. Generate only the SQL query (no explanations).
2. Use table name exactly as provided (universities).
3. If the query is unrelated to the table, return: 'No relevant data in database.'
4. Handle errors gracefully (e.g., return 'Invalid query' if unparseable).
5. For ambiguous queries, prioritize the universities table.

User query: {query}
SQL query:
"""
    try:
        response = model.generate_content(prompt)
        sql_query = response.text.strip()
        if not sql_query.startswith('SELECT') and 'No relevant data' not in sql_query:
            return 'Invalid query', []
        return sql_query, generate_follow_ups(query)
    except Exception as e:
        return f"Error generating SQL: {str(e)}", []

# Execute SQL query
def execute_sql_query(sql_query):
    try:
        conn = sqlite3.connect("data/chatbot.db")
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()
        
        if not results:
            return "No data found.", []
        
        output = "\n".join([f"{', '.join([f'{col}: {val}' for col, val in zip(columns, row)])}" for row in results])
        return output, []
    except Exception as e:
        return f"Error executing SQL: {str(e)}", []

# Generate follow-up questions
def generate_follow_ups(query):
    prompt = f"""
Based on the following user query about a university or visa service, suggest 3 relevant follow-up questions that could help the user get more information. Provide only the questions as a list.

Query: {query}
Follow-up questions:
"""
    try:
        response = model.generate_content(prompt)
        questions = [line.strip() for line in response.text.split('\n') if line.strip()]
        return questions[:3]
    except Exception:
        return ["What other programs are available?", "What are the tuition fees?", "What visa services are offered?"]

# RAG pipeline for non-SQL queries
def rag_query(query, retriever, lang_code):
    if retriever is None:
        return "Error: Knowledge base not loaded.", []
    
    if lang_code != "en":
        query = GoogleTranslator(source=lang_code, target="en").translate(query)
    
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs])
    
    prompt = f"""
You are a helpful assistant for university and visa information. Answer concisely using only this context. If the answer isn’t in the context, say 'I don’t have enough information.'

Context: {context}

Question: {query}
Answer:
"""
    try:
        response = model.generate_content(prompt)
        if lang_code != "en":
            response_text = GoogleTranslator(source="en", target=lang_code).translate(response.text)
        else:
            response_text = response.text
        follow_ups = generate_follow_ups(query)
        if lang_code != "en":
            follow_ups = [GoogleTranslator(source="en", target=lang_code).translate(q) for q in follow_ups]
        return response_text, follow_ups
    except Exception as e:
        return f"Error generating response: {str(e)}", []

# Text-to-Speech
def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp
    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        return None

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""
if 'language' not in st.session_state:
    st.session_state.language = "English"
if 'feedback' not in st.session_state:
    st.session_state.feedback = {}
if 'admin_authenticated' not in st.session_state:
    st.session_state.admin_authenticated = False

# Custom CSS
st.markdown("""
    <style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #E3F2FD;
    }
    .bot-message {
        background-color: #F5F5F5;
    }
    .timestamp {
        color: gray;
        font-size: 0.8em;
        margin-top: 8px;
    }
    .stButton>button {
        background-color: #000000;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        border: none;
    }
    .stTextInput>div>div>input {
        border-radius: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Create data folder and initialize database
os.makedirs("data", exist_ok=True)
init_db()

# Write sample CSV
sample_csv = """university,program,tuition,location,visa_service
MIT,Computer Science,60000,Cambridge,VisaMonk
Stanford,Business,75000,Palo Alto,VisaMonk
Harvard,Law,65000,Cambridge,VisaMonk"""
with open("data/sample.csv", "w", encoding="utf-8") as f:
    f.write(sample_csv)
st.write("Created sample.csv with fake university data.")

# Sidebar
with st.sidebar:
    st.write("👋 Welcome to University Chatbot!")
    
    # Language selection
    st.session_state.language = st.selectbox("Select Language", list(LANGUAGES.keys()))
    
    # Admin authentication
    st.subheader("Admin Access")
    admin_password = st.text_input("Enter admin password:", type="password")
    if st.button("Login"):
        if admin_password == "admin123":
            st.session_state.admin_authenticated = True
            st.success("Admin access granted!")
        else:
            st.session_state.admin_authenticated = False
            st.error("Incorrect password.")
    
    # Admin features
    if st.session_state.admin_authenticated:
        st.subheader("Admin: Scrape Website")
        scrape_url = st.text_input("Enter URL to scrape (e.g., https://www.visamonk.ai/):", value="https://www.visamonk.ai/")
        data_option = st.radio("Data Handling:", ("Keep old scraped data", "Replace with new data"))
        keep_old_data = data_option == "Keep old scraped data"
        if st.button("Scrape Now"):
            with st.spinner("Scraping website..."):
                if scrape_website(scrape_url, keep_old_data):
                    st.rerun()
        
        st.subheader("Admin: Manage Files")
        uploaded_file = st.file_uploader("Upload CSV, XLSX, SQL, or PDF", type=['csv', 'xlsx', 'sql', 'pdf'])
        if uploaded_file and st.button("Upload File"):
            if load_file(uploaded_file):
                st.rerun()
        
        st.subheader("Delete Files")
        all_files = [f for f in os.listdir("scraped_data") if f.endswith('.txt')] + \
                    [f for f in os.listdir("data") if f.endswith(('.csv', 'xlsx', 'sql', 'pdf'))]
        if all_files:
            file_to_delete = st.selectbox("Select file to delete:", all_files)
            if st.button("Delete File"):
                if delete_file(file_to_delete):
                    st.rerun()
        else:
            st.write("No files to delete.")
        
        if st.button("Re-Index Data"):
            with st.spinner("Re-indexing data..."):
                retriever = load_and_index_data()
                if retriever:
                    st.success("Data re-indexed successfully!")
                else:
                    st.error("Failed to re-index data.")
    else:
        st.write("Admins only: Enter password to access admin features.")
    
    # Clear chat history
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.feedback = {}
        st.rerun()
    
    # Example questions
    st.subheader("Example Questions")
    examples = [
        "What is the tuition at MIT?",
        "What programs does Stanford offer?",
        "What is visamonk.ai?"
    ]
    for q in examples:
        if st.button(q):
            st.session_state.current_question = q
            st.rerun()

# Main interface
st.title("🎓 University Chatbot")
chat_container = st.container()

# Chat history with TTS and feedback
for i, (user_msg, bot_msg, timestamp, follow_ups) in enumerate(st.session_state.chat_history):
    with chat_container:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {user_msg}</div>', unsafe_allow_html=True)
            st.caption(timestamp)
        with col2:
            st.markdown(f'<div class="chat-message bot-message"><strong>Assistant:</strong> {bot_msg}</div>', unsafe_allow_html=True)
            st.caption(timestamp)
            if st.button("Play Response", key=f"play_{i}"):
                audio_fp = text_to_speech(bot_msg)
                if audio_fp:
                    st.audio(audio_fp, format='audio/mp3')
            feedback_col1, feedback_col2 = st.columns(2)
            with feedback_col1:
                if st.button("👍", key=f"up_{i}"):
                    st.session_state.feedback[i] = "positive"
                    st.success("Thank you for your feedback!")
            with feedback_col2:
                if st.button("👎", key=f"down_{i}"):
                    st.session_state.feedback[i] = "negative"
                    st.success("Thank you for your feedback!")
            if i in st.session_state.feedback:
                st.write(f"You {'liked' if st.session_state.feedback[i] == 'positive' else 'disliked'} this response")
            if follow_ups:
                st.write("**Follow-up Questions:**")
                for j, follow_up in enumerate(follow_ups):
                    if st.button(follow_up, key=f"follow_up_{i}_{j}"):
                        st.session_state.current_question = follow_up
                        st.rerun()

# Input area
input_col1, input_col2 = st.columns([4, 1])
with input_col1:
    user_input = st.text_input("Ask your question:", value=st.session_state.current_question, key="input", placeholder="e.g., What is the tuition at MIT?")
with input_col2:
    send_button = st.button("Send 📤")

# Handle input
if send_button and user_input:
    retriever = load_and_index_data()
    if retriever:
        sql_query, sql_follow_ups = generate_sql_query(user_input, LANGUAGES[st.session_state.language])
        if sql_query.startswith('SELECT'):
            response, follow_ups = execute_sql_query(sql_query)
        else:
            response, follow_ups = rag_query(user_input, retriever, LANGUAGES[st.session_state.language])
    else:
        response, follow_ups = "Error: Unable to load knowledge base.", []
    
    current_time = datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M')
    st.session_state.chat_history.append((user_input, response, current_time, follow_ups))
    st.session_state.current_question = ""
    st.rerun()