# RAG-CHATBOT
# 🤖 AI-Powered RAG Chatbot

A document-based Q&A chatbot built using **LLMs + Embeddings + FAISS**.  
This project was developed as part of the technical assignment for **Amlgo Labs**.

## 📄 Features

- 🔍 Upload any `.txt`, `.md`, or `.pdf` document
- 🧠 Chunks and embeds the document using **MiniLM** model
- 🗃️ Stores embeddings in **FAISS vector DB**
- 💬 Allows natural language queries to search and retrieve answers
- 🧵 Simulated streaming responses for better UX
- 🎨 Clean, light UI with Streamlit

---
## 📂 File Structure

rag-chatbot-amlgo/
│
├── app.py # Streamlit app
├── requirements.txt # Python dependencies
├── report.pdf # Assignment 

## 🚀 How to Run Locally

1. **Clone the repo**
```bash
git clone https://github.com/yourusername/rag-chatbot-amlgo.git
cd rag-chatbot-amlgo

Create & activate virtual environment (optional)

bash
Copy
Edit
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the app

bash
Copy
Edit
streamlit run app.py


🧠 Models & Tools Used
Component	Description
LLM Embeddings	all-MiniLM-L6-v2 via sentence-transformers
Vector Store	FAISS (Facebook AI Similarity Search)
UI Framework	Streamlit
PDF Parsing	PyMuPDF (fitz)

👨‍💻 Developed By
Naman Jain
LinkedIn: linkedin.com/in/naman-jain-770440222
