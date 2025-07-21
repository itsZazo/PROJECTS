from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import os
import time
import uuid
from werkzeug.utils import secure_filename

# Your original imports (only fixing the deprecated one)
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field
from typing import List
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variables to store your original components
llm = None
embeddings = None
pc = None
vectorstore = None
retreiver = None
concat_docs = None
parser = None
prompt = None
chain = None
current_index_name = None  # Track current index

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Your original Pydantic models - EXACTLY the same
class MCQ(BaseModel):
    question: str = Field(description="The question of the MCQ")
    choices: List[str] = Field(description="The choices of the MCQ, (MUST BE EXACTLY 4)")
    correct_answer: str = Field(description="The correct answer of the MCQ, MUST be one of the choices") # must match one of the choices

class MCQList(BaseModel):
    mcqs: List[MCQ] = Field(description="list of 10 MCQS")

def initialize_components():
    """Initialize your original components"""
    global llm, embeddings, pc, concat_docs, parser, prompt
    
    # Your original initialization - EXACTLY the same
    llm = ChatOpenAI(
        model="llama3-70b-8192", 
        temperature=0.5
    )
    
    embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
    
    pc = Pinecone()
    
    # Your original concat_docs function
    def concat_docs_func(documents):
        return "\n\n".join(doc.page_content for doc in documents)
    
    concat_docs = RunnableLambda(concat_docs_func)
    print(concat_docs)
    
    # Your original parser
    parser = PydanticOutputParser(pydantic_object=MCQList)
    
    # Your original prompt template - EXACTLY the same
    prompt = PromptTemplate(
    template="""
You are a helpful AI tutor.

Your task is to generate EXACTLY 10 multiple-choice questions (MCQs) from the provided context.

### INSTRUCTIONS (READ CAREFULLY):
- Use ONLY the information from the context — do NOT invent anything.
- Each MCQ must have:
    - "question": string (the question)
    - "choices": list of 4 string answer options
    - "correct_answer": one of the 4 choices

### VALIDATION CHECKLIST:
Before responding, verify each MCQ has:
✓ "question" field with a string
✓ "choices" field with exactly 4 options
✓ "correct_answer" field that matches one of the choices

### RULES:
- Provide **exactly 4** answer choices for each question — no more, no less.
- The correct answer MUST be present in the "choices".
- Do NOT include introductions like: "Here are the 10 MCQs", "Sure", or "Okay". (IMPORTANT)
- Do NOT wrap the output in markdown (no triple backticks like ```json).
- If no answer is found in the context, set `"correct_answer": "No Answer Found"` and provide 3 distractors plus "No Answer Found" in choices.
- DO NOT include any explanation, comments, or extra text — just the JSON.
- Use proper syntax: commas, brackets, and quotes MUST be correctly placed.
- DO NOT wrap the response in Markdown, triple backticks, or any text outside the JSON object.


### OUTPUT FORMAT (STRICTLY FOLLOW THIS):
{format_instruction} 
IMPORTANT: DO NOT FORGET ANYTHING FROM THE JSON SYNTAX ABOVE
CRITICAL: End your JSON with: closing bracket followed by closing brace, Nothing More, Nothing Less

Now, generate the 10 MCQs based on the following context:

Context:
{context}
""",
        input_variables=["context"],
        partial_variables={"format_instruction": parser.get_format_instructions()}
    )

def cleanup_previous_index():
    """Clear existing index data to prevent data leakage"""
    global current_index_name, pc
    
    if current_index_name and pc:
        try:
            # Get the index and delete all vectors
            index = pc.Index(current_index_name)
            # Delete all vectors from the index
            index.delete(delete_all=True)
            print(f"Cleared all data from index: {current_index_name}")
        except Exception as e:
            print(f"Note: Could not clear index {current_index_name}: {e}")

def process_document(file_path):
    """Process uploaded document using your EXACT original approach with data isolation"""
    global vectorstore, retreiver, chain, current_index_name
    
    try:
        # Your original document processing - EXACTLY the same
        loader = PyMuPDFLoader(file_path)
        doc = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(doc)

        # Use your existing index name but clear it first to prevent data leakage
        current_index_name = "quiz-maker-store"
        
        # Clear previous data from the index
        try:
            index = pc.Index(current_index_name)
            index.delete(delete_all=True)
            print(f"Cleared previous data from index: {current_index_name}")
            # Small delay to ensure deletion completes
            import time
            time.sleep(2)
        except Exception as e:
            print(f"Note: Could not clear index: {e}")

        vectorstore = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=current_index_name  
        )

        retreiver = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        # Your original chain - EXACTLY the same structure
        chain = retreiver | concat_docs | prompt | llm | parser
        
        return True, f"Document processed successfully (Using index: {current_index_name})"
        
    except Exception as e:
        return False, f"Error processing document: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Add timestamp to filename to avoid conflicts
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process the document using your original approach
        success, message = process_document(file_path)
        
        if success:
            flash('Document uploaded and processed successfully!')
            return redirect(url_for('ask_topic'))
        else:
            flash(f'Error: {message}')
            return redirect(url_for('index'))
    else:
        flash('Invalid file type. Please upload a PDF file.')
        return redirect(url_for('index'))

@app.route('/ask_topic')
def ask_topic():
    if chain is None:
        flash('Please upload a document first.')
        return redirect(url_for('index'))
    return render_template('ask_topic.html')

@app.route('/generate_mcqs', methods=['POST'])
def generate_mcqs():
    if chain is None:
        flash('Please upload a document first.')
        return redirect(url_for('index'))
    
    topic = request.form.get('topic', 'From Entire Document')
    
    try:
        # Your original chain invocation - EXACTLY the same
        result = chain.invoke(topic)
        mcqs = result.mcqs
        
        return render_template('mcqs.html', mcqs=mcqs, topic=topic, index_name=current_index_name)
        
    except Exception as e:
        flash(f'Error generating MCQs: {str(e)}')
        return redirect(url_for('ask_topic'))

@app.route('/cleanup')
def cleanup():
    """Optional route to manually clean up current index"""
    cleanup_previous_index()
    flash('Index cleaned up successfully!')
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Initialize components on startup
    initialize_components()
    app.run(debug=True)
