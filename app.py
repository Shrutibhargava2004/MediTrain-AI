import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv
from flask import Flask, render_template, request, jsonify, session

# Loading the .env file
load_dotenv(find_dotenv())

# Setup Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load environment variables
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Load LLM
def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )
    return llm

# Set custom prompt template
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Don't provide anything out of the given context.Respond with empathy, gratitude, and encouragement to make the conversation warm and helpful.

Context: {context}
Question: {question}

Start the answer with a supportive tone. Offer actionable advice and encouragement. 
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load the dataset
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    user_query = request.form['query']

    # Retrieve the previous chat context from session
    context = session.get('chat_history', "")

    # Add the new query to the chat history
    context += f"User: {user_query}\n"

    # Get the response from the QA chain
    response = qa_chain.invoke({'query': user_query, 'context': context})
    
    # Update chat history in session
    session['chat_history'] = context + f"Bot: {response['result']}\n"
    
    # Convert source documents to strings
    source_documents = [str(doc) for doc in response.get("source_documents", [])]

    return jsonify({'result': response["result"], 'sources': source_documents})

if __name__ == '__main__':
    app.run(debug=True)
