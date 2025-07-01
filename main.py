import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import os
import datetime
import pytz

# Configure Streamlit page
st.set_page_config(
    page_title="University Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get API key from Streamlit secrets or environment variable
try:
    if "GOOGLE_API_KEY" in st.secrets:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    elif "GOOGLE_API_KEY" in os.environ:
        GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
    else:
        st.error(
            "Google Gemini API key not found. "
            "Please create a '.streamlit/secrets.toml' file with 'GOOGLE_API_KEY = \"your-key\"' "
            "or set the environment variable 'GOOGLE_API_KEY'."
        )
        st.stop()
except Exception as e:
    st.error(f"Error accessing API key: {str(e)}. Please check your setup.")
    st.stop()

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Load and process the knowledge base
def load_and_index_data():
    if not os.path.exists("sample_university_info.txt"):
        st.error("Please add 'sample_university_info.txt' to the project directory.")
        return None
    
    with open("sample_university_info.txt", "r") as f:
        text = f.read()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    faiss_index_path = "vectorstore/index.faiss"
    if os.path.exists("vectorstore") and os.path.exists(faiss_index_path):
        try:
            vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.warning(f"Failed to load FAISS index: {e}. Creating a new index.")
            vectorstore = FAISS.from_texts(chunks, embeddings)
            os.makedirs("vectorstore", exist_ok=True)
            vectorstore.save_local("vectorstore")
    else:
        vectorstore = FAISS.from_texts(chunks, embeddings)
        os.makedirs("vectorstore", exist_ok=True)
        vectorstore.save_local("vectorstore")
    
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# RAG pipeline
def rag_query(query, retriever):
    if retriever is None:
        return "Error: Knowledge base not loaded."
    
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs])
    
    prompt = (
        "You are a helpful assistant providing accurate information about Sample University. "
        "Use only the following context to answer the question concisely. If the context doesn't "
        "contain the answer, say 'I don’t have enough information to answer this.'\n\n"
        f"Context: {context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""

# Custom CSS for improved UI/UX
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
    .sidebar-section {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .sidebar-header {
        color: #333;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #f0f0f0;
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

# Sidebar with example questions, quick links, and contact info
with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-header">👋 Welcome!</div>', unsafe_allow_html=True)
    st.write("I'm here to assist you with  University information!")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-header">💡 Example Questions</div>', unsafe_allow_html=True)
    example_questions = [
        "What programs does Sample University offer?",
        "What are the tuition fees?",
        "When are application deadlines?",
        "What financial aid is available?"
    ]
    for question in example_questions:
        if st.button(question, key=f"example_{question}"):
            st.session_state.current_question = question
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-header">🔗 Quick Links</div>', unsafe_allow_html=True)
    st.markdown("[University Website](http://www.sampleuniversity.edu)")
    st.markdown("[Admissions](http://www.sampleuniversity.edu/admissions)")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-header">📞 Contact Support</div>', unsafe_allow_html=True)
    st.write("Email: info@sampleuniversity.edu")
    st.write("Phone: (555) 123-4567")
    st.markdown('</div>', unsafe_allow_html=True)

# Main chat interface
st.title("🎓University Chatbot")
chat_container = st.container()

# Display chat history
for user_msg, bot_msg, timestamp in st.session_state.chat_history:
    with chat_container:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {user_msg}</div>', unsafe_allow_html=True)
            st.caption(timestamp)
        with col2:
            st.markdown(f'<div class="chat-message bot-message"><strong>Assistant:</strong> {bot_msg}</div>', unsafe_allow_html=True)
            st.caption(timestamp)

# Input area
input_col1, input_col2 = st.columns([4, 1])
with input_col1:
    user_input = st.text_input("Ask your question:", value=st.session_state.current_question, key="input", placeholder="e.g., What programs are offered?")
with input_col2:
    send_button = st.button("Send 📤")

# Handle user input
if send_button and user_input:
    retriever = load_and_index_data()
    if retriever:
        response = rag_query(user_input, retriever)
    else:
        response = "Error: Unable to load knowledge base."
    
    current_time = datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M')
    st.session_state.chat_history.append((user_input, response, current_time))
    st.session_state.current_question = ""
    st.rerun()