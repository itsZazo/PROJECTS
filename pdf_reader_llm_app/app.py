from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_nomic import NomicEmbeddings

load_dotenv()

history = []

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to store the chain and current files
current_chain = None
current_files = []  # List to store multiple files
vector_store = None  # Global vector store for all documents

# Initialize LLM and embeddings
llm = ChatGroq(
    model="llama3-70b-8192", 
    temperature=0.3
)

embeddings = NomicEmbeddings(
    model="nomic-embed-text-v1.5"
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'

def process_pdf(file_path, filename):
    """Process PDF and add to existing vector store or create new one"""
    global current_chain, vector_store
    
    try:
        # Load PDF
        pdf_loader = PyPDFLoader(file_path)
        documents = pdf_loader.load()
        
        # Add filename metadata to documents
        for doc in documents:
            doc.metadata['source_file'] = filename
        
        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200    
        )
        chunks = splitter.split_documents(documents)
        
        # Create or update vector store
        if vector_store is None:
            vector_store = Chroma.from_documents(chunks, embeddings)
        else:
            vector_store.add_documents(chunks)
        
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        
        def retriever_func(question):
            global history 
            history.append(HumanMessage(content=question))
            retrieved_docs = retriever.invoke(question)
            return {"context": retrieved_docs, "question": question, "history": history}
        
        retrieve_docs = RunnableLambda(retriever_func)
        parser = StrOutputParser()
        
        prompt = PromptTemplate(
            template="""
            You are an Intelligent AI that answers questions about Uploaded Documents.
            - Answer the question as truthfully as possible using the provided context.
            - If you can't find the answers in the context, just say that you don't know.
            - when the user says page no 1, he means page 0
            - when the user says page no 2, he means page 1, and so on
            - don't say which context it is or which page number
            - When referencing information, mention which document it comes from if relevant
            
            Context: {context}

            Question: {question}

            Conversation History so far : {history}
            
            """,
            input_variables=["context", "question", "history"]
        )
        
        # Update chain
        current_chain = retrieve_docs | prompt | llm | parser
        return True
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_files, history
    
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file selected'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if not file or not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type. Please upload a PDF file.'})
        
        filename = secure_filename(file.filename)
        
        # Check if file already exists
        if filename in current_files:
            return jsonify({'success': False, 'error': 'File already uploaded'})
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the file
        file.save(file_path)
        
        # Process the PDF
        if process_pdf(file_path, filename):
            current_files.append(filename)
            return jsonify({
                'success': True, 
                'filename': filename,
                'total_files': len(current_files),
                'all_files': current_files
            })
        else:
            # Clean up failed file
            try:
                os.remove(file_path)
            except:
                pass
            return jsonify({'success': False, 'error': 'Error processing PDF file. Please try again.'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': f'Upload failed: {str(e)}'})

@app.route('/ask', methods=['POST'])
def ask_question():
    global current_chain, history
    
    if current_chain is None:
        return jsonify({'success': False, 'error': 'No PDF file processed yet'})
    
    data = request.get_json()
    question = data.get('question', '')
    
    if not question:
        return jsonify({'success': False, 'error': 'No question provided'})
    
    try:
        result = current_chain.invoke(question)
        history.append(AIMessage(content=result))
        return jsonify({'success': True, 'answer': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/status')
def status():
    global current_files
    return jsonify({
        'has_file': len(current_files) > 0, 
        'files': current_files,
        'total_files': len(current_files)
    })

@app.route('/clear', methods=['POST'])
def clear_files():
    global current_files, current_chain, vector_store, history
    
    try:
        # Clear uploaded files
        for filename in current_files:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                os.remove(file_path)
            except:
                pass
        
        # Clear vector store properly
        if vector_store is not None:
            try:
                # Delete the vector store collection
                vector_store.delete_collection()
            except:
                pass
        
        # Reset global variables
        current_files = []
        current_chain = None
        vector_store = None
        history = []
        
        return jsonify({'success': True, 'message': 'All files cleared'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)