# 🧠 AI PDF to MCQ Generator (Flask + LangChain + Pinecone)

This is a Flask-based web application that allows users to upload a **PDF file**, and then intelligently generates **Multiple Choice Questions (MCQs)** using a Large Language Model (LLM). It uses LangChain to orchestrate LLMs, vector databases, and parsing logic — with Pinecone for vector storage and HuggingFace for embedding.

---

## 📦 Features

- ✅ Upload any **PDF file** (up to 16MB)
- 🧠 Extracts and embeds document content using `HuggingFaceEmbeddings`
- 🔍 Stores chunks in **Pinecone Vector DB**
- 💬 Uses `llama3-70b-8192` (or any OpenAI-compatible LLM) via `LangChain`
- 📝 Generates **exactly 10 MCQs** using only relevant document content
- 📄 Custom `PromptTemplate` ensures JSON-only output with strict format
- 🔍 Full debugging & trace logging for prompt, LLM output, and parser
- ♻️ Index cleanup endpoint to reset Pinecone state
- ⚠️ Built-in error handling and validation

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/pdf-mcq-generator.git
cd pdf-mcq-generator
2. Create a Virtual Environment
bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies
bash
Copy code
pip install -r requirements.txt
4. Set Up .env
Create a .env file in the root folder with the following:

env
Copy code
GROQ_API_KEY=your_groq_or_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=your_pinecone_environment  # e.g., "gcp-starter"
🛠️ Folder Structure
graphql
Copy code
📁 uploads/                # Uploaded PDF files (auto-created)
📁 templates/
    ├── index.html
    ├── ask_topic.html
    └── mcqs.html
📄 app.py                  # Main Flask application
📄 .env                    # API keys and config
📄 requirements.txt        # Python dependencies
▶️ Run the App
bash
Copy code
python app.py
Visit: http://127.0.0.1:5000

🧪 Usage
Go to the homepage

Upload a PDF

After successful upload, input a topic or just press submit to use full document

MCQs are displayed with question, choices, and correct answer

Use /cleanup route to clear Pinecone index manually

🧰 Tech Stack
Tool	Purpose
Flask	Web server
LangChain	LLM chaining, prompting, parsing
PyMuPDF	PDF loading
HuggingFace	Embeddings (MiniLM-L6-v2)
Pinecone	Vector database
Pydantic	Schema validation
OpenAI / Groq	LLM Backend (LLaMA3 / GPT models)

📄 Sample Prompt Behavior
The app sends this prompt structure to the LLM:

text
Copy code
You are a helpful AI tutor...
[context from uploaded PDF]
It then parses the strict JSON response into usable MCQs.

⚠️ Notes
You must configure Pinecone before running. Default index name: quiz-maker-store.

The app clears the Pinecone index before each upload to avoid cross-contamination.

The current LLM is set to: llama3-70b-8192 — update as needed.

🧹 Cleanup
To manually clear the current index:

bash
Copy code
http://localhost:5000/cleanup
🙏 Credits
LangChain

HuggingFace

Pinecone

OpenAI or Groq

📜 License
MIT License — Free to use and modify with attribution.

yaml
Copy code


