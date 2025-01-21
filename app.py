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
You are Meditrain AI, a sophisticated medical assistant. Your role is to respond to the user's query with empathy and clarity, breaking down your answers into four parts:

1. **<b>Disease Overview and Symptoms</b>**: Start by describing the disease and its common symptoms. Make sure to address the user's query thoroughly.
2. **<b>Treatment and Recommendations</b>**: Then, explain the treatment options for the disease. Include lifestyle changes, possible procedures, and things the patient should avoid. Offer supportive advice.
3. **<b>Severity</b>**: Include the severity of the condition (low, medium, or high).
4. **<b>Medications</b>**: Finally, provide a list of common medications that could be used to treat the disease. Include names and, if applicable, recommended dosages or forms.

Context: {context}
Question: {question}

Start your response with a supportive tone, offer actionable advice, and be empathetic. Ensure you break your answer into the sections described above.
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
