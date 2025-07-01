Sample University Chatbot
A chatbot designed to provide accurate information about Sample University using Retrieval-Augmented Generation (RAG) with the Google Gemini API.
Features

Answers queries about academics, admissions, fees, and more based solely on sample_university_info.txt.
Enhanced UI/UX with a sidebar featuring example questions, quick links, and contact info.
Chat history with timestamps displayed in a conversation-like layout.
Built with Streamlit, LangChain, FAISS, and Google Gemini API.

Setup Instructions (Local)

Clone the Repository
git clone https://github.com/your-username/sample-university-chatbot.git
cd sample-university-chatbot


Create a Virtual Environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows


Install Dependencies
pip install -r requirements.txt


Configure SecretsCreate a .streamlit/secrets.toml file with:
[secrets]
GOOGLE_API_KEY = "your-actual-gemini-api-key"


Add DatasetPlace sample_university_info.txt in the project root directory.

Run the Application
streamlit run main.py



Deployment Instructions (Streamlit Cloud)

Push the repository to GitHub, ensuring .gitignore excludes venv/, vectorstore/, and .streamlit/secrets.toml.
Sign up at share.streamlit.io.
Create a new app, select your repo, and set main.py as the entry point.
In the app settings, add a secret:GOOGLE_API_KEY = "your-actual-gemini-api-key"


Deploy and test the live URL.

Test Queries

"What programs does Sample University offer?"
"What are the tuition fees?"
"When are application deadlines?"
"What financial aid is available?"

Notes

Ensure sample_university_info.txt is present before running.
The FAISS index is generated on the first run and saved in vectorstore/.
Chat history is stored in session state and resets when the app closes.
