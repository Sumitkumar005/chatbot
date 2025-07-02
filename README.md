University Chatbot
A chatbot for university information using RAG with Google Gemini API and web scraping with Tavily API.
Features

Answers queries from university_info.txt and scraped web data (e.g., MIT website).
Multilingual support (English, Hindi, Spanish, etc.).
Text-to-Speech (TTS) for responses.
Feedback system (thumbs-up/thumbs-down).
Follow-up question suggestions.
Admin feature to scrape websites and save data as text files.
UI with sidebar, chat history, and example questions.
Deployed on Streamlit Cloud.

Setup (Local)

Clone: git clone https://github.com/sumitkumar005/chatbot.git
Create virtual environment: python -m venv venv
Activate: venv\Scripts\activate (Windows) or source venv/bin/activate (Linux/Mac)
Install dependencies: pip install -r requirements.txt
Create .streamlit/secrets.toml:[secrets]
GOOGLE_API_KEY = "your-actual-gemini-api-key"
TAVILY_API_KEY = "your-actual-tavily-api-key"


Add university_info.txt to the root.
Run: streamlit run main.py

Deployment (Streamlit Cloud)

Push to GitHub (https://github.com/sumitkumar005/chatbot).
At share.streamlit.io, create an app, select the repo, set main.py as the entry point.
Add secrets in app settings:[secrets]
GOOGLE_API_KEY = "your-actual-gemini-api-key"
TAVILY_API_KEY = "your-actual-tavily-api-key"


Deploy and test: https://sumit-chatbot.streamlit.app/.

Admin Features

Scrape Website: Enter a URL (e.g., https://www.mit.edu) in the admin section (password: admin123) to scrape content. Data is saved in scraped_data/ as text files and indexed with FAISS.
Password: Replace admin123 with a secure password in production.

Test Queries

"What programs does the university offer?"
"What are the tuition fees?" (try in Hindi: "ट्यूशन फीस क्या है?")
"What programs does MIT offer?" (after scraping https://www.mit.edu)
Click follow-up questions to explore more.

Notes

Uses FAISS for RAG and session state for chat history (no MongoDB).
Scraped data is stored in scraped_data/ and indexed with university_info.txt.
Ensure university_info.txt is present.
Admin password (admin123) should be changed for security.
