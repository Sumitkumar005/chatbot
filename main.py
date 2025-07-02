import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from deep_translator import GoogleTranslator
from gtts import gTTS
from io import BytesIO
import os
import datetime
import pytz
from tavily import TavilyClient
import getpass
from dotenv import load_dotenv
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

# Scrape website and save to /scraped_data
def scrape_website(url, keep_old_data):
    try:
        # Validate URL
        if not url.startswith("http"):
            st.error("Invalid URL. Please include 'http://' or 'https://'.")
            return False
        
        # Create scraped_data folder
        os.makedirs("scraped_data", exist_ok=True)
        
        # Clear old scraped data if not keeping
        if not keep_old_data:
            for file in os.listdir("scraped_data"):
                os.remove(os.path.join("scraped_data", file))
            st.write("Cleared old scraped data.")
        
        # Perform crawl
        with st.spinner(f"Scraping {url}..."):
            crawl_results = tavily_client.crawl(url=url, max_depth=3, extract_depth="advanced")
        
        if not crawl_results["results"]:
            st.error("No data scraped. Check URL or Tavily API limits.")
            return False
        
        # Save each page's content
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

# Load and index data from university_info.txt and scraped_data
def load_and_index_data():
    all_texts = []
    
    # Load sample university data
    sample_file = "university_info.txt"
    if os.path.exists(sample_file):
        with open(sample_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                all_texts.append(content)
                st.write(f"Loaded {len(content)} characters from {sample_file}")
            else:
                st.warning(f"{sample_file} is empty.")
    else:
        st.warning(f"{sample_file} not found.")
    
    # Load scraped data
    scraped_dir = "scraped_data"
    scraped_count = 0
    if os.path.exists(scraped_dir):
        for file_name in os.listdir(scraped_dir):
            if file_name.endswith(".txt"):
                file_path = os.path.join(scraped_dir, file_name)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            all_texts.append(content)
                            st.write(f"Loaded {len(content)} characters from {file_path}")
                            scraped_count += 1
                        else:
                            st.warning(f"{file_path} is empty.")
                except Exception as e:
                    st.warning(f"Failed to load {file_path}: {str(e)}")
    else:
        st.warning("No scraped_data folder found.")
    
    if not all_texts:
        st.error("No valid data found in university_info.txt or scraped_data.")
        return None
    
    # Split texts
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for text in all_texts:
        chunks.extend(text_splitter.split_text(text))
    st.write(f"Created {len(chunks)} chunks for indexing.")
    
    # Create new FAISS index
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    try:
        # Clear old FAISS index
        if os.path.exists("vectorstore"):
            shutil.rmtree("vectorstore")
            st.write("Cleared old FAISS index.")
        
        # Create new index
        vectorstore = FAISS.from_texts(chunks, embeddings)
        os.makedirs("vectorstore", exist_ok=True)
        vectorstore.save_local("vectorstore")
        st.write("Created new FAISS index.")
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"Failed to create FAISS index: {str(e)}")
        return None

# Generate follow-up questions
def generate_follow_ups(query):
    prompt = (
        "Based on the following user query about a university or visa service, suggest 3 relevant follow-up questions "
        "that could help the user get more information. Provide only the questions as a list.\n\n"
        f"Query: {query}\n"
        "Follow-up questions:"
    )
    try:
        response = model.generate_content(prompt)
        questions = [line.strip() for line in response.text.split('\n') if line.strip()]
        return questions[:3]
    except Exception:
        return ["What services does it offer?", "What are the success rates?", "How does it work?"]

# RAG pipeline with translation and follow-ups
def rag_query(query, retriever, lang_code):
    if retriever is None:
        return "Error: Knowledge base not loaded.", []
    
    if lang_code != "en":
        query = GoogleTranslator(source=lang_code, target="en").translate(query)
    
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs])
    
    prompt = (
        "You are a helpful assistant for university and visa information. "
        "Answer concisely using only this context. If the answer isn’t in the context, say 'I don’t have enough information.'\n\n"
        f"Context: {context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )
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

# Sidebar
with st.sidebar:
    st.write("👋 Welcome to University Chatbot!")
    
    # Language selection
    st.session_state.language = st.selectbox("Select Language", list(LANGUAGES.keys()))
    
    # Admin authentication
    st.subheader("Admin Access")
    admin_password = st.text_input("Enter admin password:", type="password")
    if st.button("Login"):
        if admin_password == "admin123":  # Hardcoded for simplicity; use secrets in production
            st.session_state.admin_authenticated = True
            st.success("Admin access granted!")
        else:
            st.session_state.admin_authenticated = False
            st.error("Incorrect password.")
    
    # Admin scraping interface
    if st.session_state.admin_authenticated:
        st.subheader("Admin: Scrape Website")
        scrape_url = st.text_input("Enter URL to scrape (e.g., https://www.visamonk.ai/):", value="https://www.visamonk.ai/")
        data_option = st.radio("Data Handling:", ("Keep old scraped data", "Replace with new data"))
        keep_old_data = data_option == "Keep old scraped data"
        if st.button("Scrape Now"):
            with st.spinner("Scraping website..."):
                if scrape_website(scrape_url, keep_old_data):
                    st.rerun()  # Re-run to re-index
        
        # Re-index button
        if st.button("Re-Index Data"):
            with st.spinner("Re-indexing data..."):
                retriever = load_and_index_data()
                if retriever:
                    st.success("Data re-indexed successfully!")
                else:
                    st.error("Failed to re-index data.")
    else:
        st.write("Admins only: Enter password to access scraping.")
    
    # Clear chat history
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.feedback = {}
        st.rerun()
    
    # Example questions
    st.subheader("Example Questions")
    examples = [
        "What is visamonk.ai?",
        "What programs does MIT offer?",
        "What are the tuition fees?"
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
    user_input = st.text_input("Ask your question:", value=st.session_state.current_question, key="input", placeholder="e.g., What is visamonk.ai?")
with input_col2:
    send_button = st.button("Send 📤")

# Handle input
if send_button and user_input:
    retriever = load_and_index_data()
    if retriever:
        response, follow_ups = rag_query(user_input, retriever, LANGUAGES[st.session_state.language])
    else:
        response, follow_ups = "Error: Unable to load knowledge base.", []
    
    current_time = datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M')
    st.session_state.chat_history.append((user_input, response, current_time, follow_ups))
    st.session_state.current_question = ""
    st.rerun()