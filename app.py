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

# Custom prompt template for reduced word responses
CUSTOM_PROMPT_TEMPLATE = """
If the user greets you (e.g., "hello", "hi doctor"), respond with a friendly greeting and let them know you're ready to assist with their medical queries. Do start talking about any disease on your own.

Otherwise, if the user asks about a disease, provide a concise and summarized answer. Break your answer into:

1. <b>Overview</b>: A brief description of the disease.
2. <b>Key Symptoms</b>: Mention only the most critical symptoms.
3. <b>Short Treatment</b>: Offer concise recommendations or treatments.
4. <b>Medicines</b>: Mention 1-2 commonly used medicines or treatments.

Keep the response concise, clear, and supportive.

Context: {context}
Question: {question}

Ensure brevity in your responses without losing clarity. 
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
    return_source_documents=True,  # This will be handled below
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/login')
def login():
    return render_template('login.html')

# Define the route for the register page
@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')


import re
from bs4 import BeautifulSoup
from flask import request, jsonify, session

# Function to remove HTML tags from a response
def remove_html_tags(text):
    return BeautifulSoup(text, "html.parser").get_text()

# Function to split the response by new lines
def split_response_by_newline(response_text):
    # Clean the response to remove HTML tags
    cleaned_response = remove_html_tags(response_text)
    
    # Split the response by new lines
    lines = cleaned_response.split("\n")
    
    # Filter out empty lines and strip unnecessary spaces
    messages = [line.strip() for line in lines if line.strip()]
    
    return messages


@app.route('/query', methods=['POST'])
def query():
    user_query = request.form['query'].strip()

    # Retrieve the previous chat context from session
    context = session.get('chat_history', "")

    # Add the new query to the chat history
    context += f"User: {user_query}\n"

    # Get the response from the QA chain
    response = qa_chain.invoke({'query': user_query, 'context': context})

    # Update chat history in session
    session['chat_history'] = context + f"Bot: {response['result']}\n"

    # Process the response text
    response_text = response["result"]

    # Split the response based on new lines into a list of messages
    messages = split_response_by_newline(response_text)

    # Print the split messages for debugging
    print("Messages after splitting:", messages)

    # Check if messages are being created
    if not messages:
        print("No messages to return.")

    # Assign pastel colors to each part of the message
    pastel_colors = [
        "#FFCCCB", "#D9F9D9", "#FFFACD", "#FFDAB9", "#D1C4E9",
        "#FFECB3", "#F8BBD0", "#E6EE9C", "#C8E6C9", "#BBDEFB", "#F5E0B7"
    ]
    colored_messages = [{"text": msg, "bg_color": pastel_colors[i % len(pastel_colors)]} for i, msg in enumerate(messages)]

    # Return the response in JSON format
    return jsonify({'messages': colored_messages})


if __name__ == '__main__':
    app.run(debug=True)
