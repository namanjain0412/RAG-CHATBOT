

# 🤖 RAG-Chatbot

A document-based Q&A chatbot built using **LLMs + Embeddings + FAISS**.  
This project was developed as part of the technical assignment for **Amlgo Labs**.

---

## 📄 Features

- 🔍 Upload any `.txt`, `.md`, or `.pdf` document
- 🧠 Splits and embeds the document using **MiniLM** model
- 🗃️ Stores embeddings in a **FAISS vector database**
- 💬 Ask natural language questions and get contextual answers
- 🧵 Simulated streaming responses for better chat feel
- 🎨 Clean and responsive UI built with **Streamlit**

---

## 📂 File Structure

```

rag-chatbot-amlgo/
│
├── app.py              # Streamlit chatbot app
├── requirements.txt    # Python dependencies
├── report.pdf          # 2–3 page technical write-up (for Amlgo submission)
├── README.md           # Project overview 
└── docs/               # Screenshots, sample PDFs, etc.

````

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/rag-chatbot-amlgo.git
cd rag-chatbot-amlgo
````

### 2. Create and Activate Virtual Environment (optional but recommended)

```bash
python -m venv venv
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Chatbot

```bash
streamlit run app.py
```

---

## 🧠 Models & Tools Used

| Component      | Description                                    |
| -------------- | ---------------------------------------------- |
| LLM Embeddings | `all-MiniLM-L6-v2` via `sentence-transformers` |
| Vector Store   | FAISS (Facebook AI Similarity Search)          |
| UI Framework   | Streamlit                                      |
| PDF Parsing    | PyMuPDF (`fitz`)                               |

---

## 👨‍💻 Developed By

**Naman Jain**
📧 Email: [namanofficial57@gmail.com](mailto:namanofficial57@gmail.com)
🔗 LinkedIn: [linkedin.com/in/naman-jain-770440222](https://linkedin.com/in/naman-jain-770440222)

---




