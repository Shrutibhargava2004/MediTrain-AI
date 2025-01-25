import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv
from flask import Flask, render_template, request, jsonify, session, redirect, flash, url_for
import numpy as np
import pickle, joblib, re
import pandas as pd
from bs4 import BeautifulSoup
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash


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
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# DATABASE 
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:Shruti%402004@localhost/MeditrainAI'  # Update password and DB name
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

# Register Route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email is already registered. Please login.', 'error')
            return redirect('/register')
        # Hash the password
        hashed_password = generate_password_hash(password)

        # Add user to the database
        new_user = User(name=name, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! Please login.', 'success')
        return redirect('/login')

    return render_template('register.html')
# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):  # Assuming password is hashed
            session['user_id'] = user.id  # Store user ID in the session
            flash('Logged in successfully!', 'success')  # Send a success message to the template
            return render_template('login.html', login_success=True)  # Pass a flag to trigger the alert
        else:
            flash('Invalid credentials. Please try again.', 'danger')  # Error message
            return render_template('login.html', login_success=False)
    return render_template('login.html', login_success=False)  # Default behavior

# Logout Route
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))








# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/chat')
def chat():
    if 'user_id' not in session:
        flash('Please log in to access the chat.', 'warning')
        return redirect(url_for('login'))
    return render_template('chat.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/diagnose')
def diagnose():
    return render_template('diagnose.html')

@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')  # Or any content you want to display

@app.route('/heartdisease')
def heartdisease():
    return render_template('heart_disease.html')

@app.route('/alzheimer')
def alzheimer():
    return render_template('alzheimer.html')

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def remove_html_tags(text):
    return BeautifulSoup(text, "html.parser").get_text()

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
    # print("Messages after splitting:", messages)

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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load the diabetes classifier model
        diabetes_filename = 'model/voting_diabetes.pkl'
        with open(diabetes_filename, 'rb') as model_file:
            diabetes_classifier = pickle.load(model_file)

        if request.method == 'POST':
            # Get input values from the form
            glucose = int(request.form.get('glucose', 0))
            bp = int(request.form.get('bloodpressure', 0))
            st = int(request.form.get('skinthickness', 0))
            insulin = int(request.form.get('insulin', 0))
            bmi = float(request.form.get('bmi', 0))
            dpf = float(request.form.get('dpf', 0))
            age = int(request.form.get('age', 0))
            
            # Prepare data for prediction
            data = np.array([[glucose, bp, st, insulin, bmi, dpf, age]])
            
            # Make the prediction
            my_prediction = diabetes_classifier.predict(data)

            # Interpret the prediction
            if my_prediction[0] == 0:
                output = "No Diabetes"
            else:
                output = "Diabetes"

            # Render the template with the result
            return render_template('diabetes.html', prediction_text="Result: {}".format(output))

    except FileNotFoundError:
        return "Model file not found. Please ensure 'voting_diabetes.pkl' is in the 'model/' directory.", 500
    except Exception as e:
        return f"An error occurred: {str(e)}", 500
# heart disease
@app.route('/heart_predict', methods=['POST'])
def heart_predict():
    heartmodel = pickle.load(open('model/hdp_model.pkl', 'rb'))
    try:
        # Debug: Print received form data keys
        # print("Form keys received:", request.form.keys())

        # Collect form data
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        # Prepare the input data for prediction
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        # Make prediction
        prediction = heartmodel.predict(input_data)[0]

        # Map prediction to a human-readable result
        result = "High risk of heart disease" if prediction == 1 else "Low risk of heart disease"

        return render_template('heart_disease.html', prediction=result)

    except Exception as e:
        return render_template('heart_disease.html', prediction=f"An error occurred: {e}")

# Alzheimer disease
APOE_CATEGORIES = ['APOE Genotype_2,2', 'APOE Genotype_2,3', 'APOE Genotype_2,4', 
                   'APOE Genotype_3,3', 'APOE Genotype_3,4', 'APOE Genotype_4,4']
PTHETHCAT_CATEGORIES = ['PTETHCAT_Hisp/Latino', 'PTETHCAT_Not Hisp/Latino', 'PTETHCAT_Unknown']
IMPUTED_CATEGORIES = ['imputed_genotype_True', 'imputed_genotype_False']
PTRACCAT_CATEGORIES = ['PTRACCAT_Asian', 'PTRACCAT_Black', 'PTRACCAT_White']
PTGENDER_CATEGORIES = ['PTGENDER_Female', 'PTGENDER_Male']
APOE4_CATEGORIES = ['APOE4_0', 'APOE4_1', 'APOE4_2']
ABBREVIATION = {
    "AD": "Alzheimer's Disease ",
    "LMCI": "Late Mild Cognitive Impairment ",
    "CN": "Cognitively Normal"
}
CONDITION_DESCRIPTION = {
    "AD": "This indicates that the individual's data aligns with characteristics commonly associated with "
        "Alzheimer's disease. Alzheimer's disease is a progressive neurodegenerative disorder that affects "
        "memory and cognitive functions.",
    "LMCI": "This suggests that the individual is in a stage of mild cognitive impairment that is progressing "
            "towards Alzheimer's disease. Mild Cognitive Impairment is a transitional state between normal "
            "cognitive changes of aging and more significant cognitive decline.",
    "CN": "This suggests that the individual has normal cognitive functioning without significant impairments. "
        "This group serves as a control for comparison in Alzheimer's research."
}
# convert selected category into one-hot encoding
def convert_to_one_hot(selected_category, all_categories, user_input):
    one_hot = [1 if category == selected_category else 0 for category in all_categories]
    user_input.extend(one_hot)
alzheimer_model = joblib.load('model/alzheimer_model.pkl')

def alzheimer_predict(input_data):
    predictions = alzheimer_model.predict(input_data)
    return predictions

@app.route('/alzheimer_predict', methods=['POST'])
def alzheimer_disease_predict():
    # Extract data from the form
    age = int(request.form['age'])
    education = int(request.form['education'])
    mmse = int(request.form['mmse'])
    gender = request.form['gender']
    ethnicity = request.form['ethnicity']
    race_cat = request.form['race']
    apoe_allele_type = request.form['apoe_allele']
    apoe_genotype = request.form['apoe_genotype']
    imputed_genotype = request.form['imputed_genotype']

    # Initialize user input list
    user_input = [age, education, mmse]

    # Convert categorical fields to one-hot encoding
    convert_to_one_hot("PTRACCAT_" + race_cat, PTRACCAT_CATEGORIES, user_input)
    convert_to_one_hot("APOE Genotype_" + apoe_genotype, APOE_CATEGORIES, user_input)
    convert_to_one_hot("PTETHCAT_" + ethnicity, PTHETHCAT_CATEGORIES, user_input)
    convert_to_one_hot(apoe_allele_type, APOE4_CATEGORIES, user_input)
    convert_to_one_hot("PTGENDER_" + gender, PTGENDER_CATEGORIES, user_input)
    convert_to_one_hot("imputed_genotype_" + imputed_genotype, IMPUTED_CATEGORIES, user_input)

    # Convert user input into a DataFrame
    input_df = pd.DataFrame([user_input])

    # Make prediction
    predicted_condition = alzheimer_predict(input_df)

    # Prepare the result
    result = {
        "predicted_condition": f"{ABBREVIATION[predicted_condition[0]]} ({predicted_condition[0]}) - {CONDITION_DESCRIPTION[predicted_condition[0]]}"
    }

    # Return the template with the result
    return render_template('alzheimer.html', result=result)









def init_db():
    with app.app_context():
        db.create_all()

if __name__ == '__main__':
    init_db()  # Initialize the database
    app.run(debug=True)
