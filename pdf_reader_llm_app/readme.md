Here’s a `README.md` file for your **RAG-based PDF Chatbot** project using Flask, LangChain, and Groq:

---

# 🧠 RAG-Powered Multi-PDF Chatbot

This is a Flask-based AI assistant that uses **Retrieval-Augmented Generation (RAG)** to allow users to upload multiple PDF files and ask natural language questions about them. The chatbot answers based on the uploaded content using **LangChain**, **Groq's LLaMA3-70B**, **Nomic Embeddings**, and **Chroma vector store**.

---

## 🚀 Features

* 📄 Upload and query **multiple PDFs**
* 🧠 **Context-aware Q\&A** using document content
* 🤖 Powered by **LLaMA3-70B via Groq**
* 📚 **Nomic Embeddings** for vectorization
* 🔍 Similarity-based document retrieval (via **Chroma**)
* 💬 Maintains **conversation history**
* 🔥 Deployed using **Flask** for a simple web interface

---

## 🛠️ Tech Stack

* **Backend**: Flask (Python)
* **LLM**: Groq (LLaMA3-70B)
* **Embeddings**: Nomic Embed Text v1.5
* **RAG Framework**: LangChain
* **Vector DB**: Chroma
* **PDF Parsing**: LangChain’s `PyPDFLoader`
* **Text Splitting**: `RecursiveCharacterTextSplitter`

---

## 📁 Project Structure

```
📦project-root/
 ┣ 📂uploads/                  # Stores uploaded PDF files
 ┣ 📜app.py                    # Flask backend server
 ┣ 📜.env                      # API keys and secrets
 ┣ 📜requirements.txt          # Python dependencies
 ┗ 📜README.md                 # This file
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/rag-pdf-chatbot.git
cd rag-pdf-chatbot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables

Create a `.env` file in the root directory and add:

```
GROQ_API_KEY=your_groq_api_key
```

### 4. Run the App

```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

---

## 💡 How It Works

1. **User Uploads PDF(s)**

   * Extracts and chunks text
   * Generates embeddings and stores in Chroma

2. **User Asks a Question**

   * Retrieves relevant chunks using vector similarity
   * Constructs a prompt using retrieved context + chat history
   * Sends to Groq's LLaMA3-70B model for response

3. **Chatbot Responds**

   * Maintains conversational flow
   * References relevant document(s) if possible

---

## 🧹 Clear Session

Use the **`/clear`** endpoint to:

* Delete uploaded PDFs
* Reset the vector store
* Clear conversation history

---

## 📌 Limitations

* Only supports `.pdf` files
* Maximum upload size: 16MB
* Doesn't handle scanned/image PDFs (OCR not implemented)

---

## 📬 Contact

Built with ❤️ by **Aitzaz Ahmed**
[Portfolio](https://ahtezaz.streamlit.app)


