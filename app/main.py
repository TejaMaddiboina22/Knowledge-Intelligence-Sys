from ..models.vector_store import VectorStore
from ..services.storage_service import StorageService
from ..services.llm_service import LLMService
from config import Config
import os
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import logging 
from flask import Flask, request,render_template, jsonify
app = Flask(__name__)

VectorStore = VectorStore(Config)
StorageService = StorageService()
LLMService = LLMService(VectorStore)

@app.route('/')
def index():
    return render_template('index.html')

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def process_document(file):
    '''process document based on file type and return text chunks'''
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    try:
        #sace file temporarily
        file.save(temp_path)

        #Process based on file type
        if file.filename.endswith('.pdf'):
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
        elif file.filename.endswith('.txt'):
            loader = TextLoader(temp_path)
            documents = loader.load()
        else:
            raise ValueError("Unsupported file type. Only PDF and TXT are allowed.")
        #Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        text_chunks = text_splitter.split_documents(documents)
        return text_chunks
    
    finally:
        #Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        os.rmdir(temp_dir)

""" @app.route('/upload', methods=['POST'])
def upload_document():
    try:
        logger.debug("Received upload request.")
        if 'file' not in request.files:
            logger.error("No file part in the request.")
            return jsonify({"error": "No file part in the request."}), 400  
    #Add to vector store 
        file = request.files['file']
        if file.filename == '':
            logger.warning("Empty filename.")
            return jsonify({"error": "No selected file."}), 400
        
        #check file extension
        if not file.filename.endswith(('.pdf', '.txt')):
            logger.warning("Unsupported file type:{file.filename}")
            return jsonify({"error": "Unsupported file type. Only PDF and TXT are allowed."}), 400
        
        logger.debug(f"Processing file: {file.filename}")

        #Process the document 

        try:
            text_chunks = process_document(file)
            logger.debug(f"Processed document into {len(text_chunks)} chunks.")
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return jsonify({"error": f"Error processing document: {str(e)}"}), 500
        
        #Upload to S3
        try:
            file.seek(0)  #Reset file pointer
            storage_service.upload_file(file, file.filename)
            logger.debug(f"Uploaded file to S3")
        except Exception as e:
            logger.error(f"Error uploading file to S3: {str(e)}")
            return jsonify({"error": f"Error uploading file to S3: {str(e)}"}), 500
        
        return jsonify({
            'message': 'File uploaded and processed successfully.',
            'chunks_added': len(text_chunks)
        })
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
     """
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)  