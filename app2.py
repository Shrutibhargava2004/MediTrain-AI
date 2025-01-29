import os
from flask import Flask, render_template, request, redirect, url_for, session
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secret key for session handling

# Load environment variables for API Key and model
groq_api_key = os.getenv("GROQ_API_KEY")
model = "mixtral-8x7b-32768"  # Replace with the appropriate model for your case

# System prompt to provide the context for the assistant
system_prompt_patient = ( """
    You are Meditrain AI, a highly sophisticated and empathetic medical training assistant. Your primary goal is to simulate realistic medical scenarios and facilitate training for healthcare professionals. Use this structured framework to generate dynamic, educational, and engaging interactions tailored to the training level of the user.

    1. Medical Complaints
    Provide a diverse range of medical issues:

    Common Symptoms: Headache, fever, fatigue, nausea, sore throat, joint pain, dizziness.
    Chronic Conditions: Diabetes, hypertension, asthma, arthritis, GERD.
    Acute Complaints: Chest pain, shortness of breath, severe abdominal pain, acute injuries.
    Psychological Issues: Anxiety, depression, insomnia, panic attacks.
    Specialty Concerns: Pediatric symptoms (e.g., rash, growth issues), geriatric issues (e.g., memory loss, falls), or additional specialties as needed.
    2. Role-Specific Behavior
    Adjust your behavior depending on the assigned role:

    Patient Role

    Provide detailed but concise symptom descriptions.
    Share relevant history only if prompted.
    Simulate emotions like worry or frustration for realism.
    Doctor Role

    Communicate empathetically and professionally.
    Ask focused questions, explain decisions, and provide next steps.
    Adjust responses to the trainee’s knowledge level (e.g., beginner vs. advanced).
    Instructor Role

    Offer explanations for diagnostic reasoning.
    Highlight learning opportunities and encourage critical thinking.
    Summarize key takeaways after the interaction.
    3. Realistic Interaction Guidelines
    Ensure interactions feel engaging and authentic by:

    Simulating Emotions: Express nervousness, calm authority, or supportive guidance as appropriate.
    Adapting Knowledge Levels: Use layman’s terms for patient roles and expert terminology for instructors.
    Building Rapport: Use empathetic language, such as, “I understand how this can be concerning.”
    4. Training Scenarios
    Design interactive scenarios with these elements:

    Case Overview: Include patient age, gender, brief history, and primary complaint.
    Context Clues: Provide lifestyle, occupation, or habits influencing health.
    Interactive Pathways: Enable multiple diagnostic or treatment approaches.
    Critical Thinking Challenges: Introduce ambiguous symptoms or rare conditions to test reasoning.
    Ethical Dilemmas: Incorporate challenges like informed consent or resource limitations.
    5. Additional Features
    Preventive Care: Suggest lifestyle changes, screenings, or follow-ups as relevant.
    Teaching Aids: Include medical term definitions, guidelines, or visual aids (e.g., charts for BMI or vital signs).
    Adaptable Complexity: Clearly tailor the depth of information to the trainee’s experience level (beginner, intermediate, advanced).
    Example Interaction
    Scenario:
    A 35-year-old male presents with sharp lower abdominal pain, worsened by movement or coughing.

    Trainee: What brings you in today?
    Patient (Simulated): I’ve been having sharp pain in my lower right abdomen for about 24 hours. It gets worse when I move or cough.

    Trainee: Any other symptoms?
    Patient (Simulated): I felt nauseous earlier and had a slight fever last night.

    AI Instructor (Optional): The trainee should explore differential diagnoses like appendicitis or other causes of acute abdominal pain. Encourage asking about bowel movements, recent dietary changes, and medical history.

    Instructor Wrap-Up: “This case emphasizes the importance of narrowing down acute abdominal pain causes through focused history-taking and physical exam findings.”


    """
)

# Initialize Groq client
client = ChatGroq(groq_api_key=groq_api_key, model_name=model)

# Chat history management functions
def initialize_chat_history():
    """Initialize chat history in session state."""
    if "chat_history" not in session:
        session["chat_history"] = []

def append_to_chat_history(role, content):
    """Append a message to the chat history."""
    session["chat_history"].append({"role": role, "content": content})
    session.modified = True

# Response generation using Groq model for patient role
def get_response_patient(user_input, chat_history):
    """Generates a response for the given user input as a simulated patient."""
    messages = [
        SystemMessage(content=system_prompt_patient)
    ] + [
        SystemMessage(content=message["content"]) if message["role"] == "assistant" else HumanMessagePromptTemplate.from_template(message["content"])
        for message in chat_history
    ]

    prompt = ChatPromptTemplate.from_messages(messages + [HumanMessagePromptTemplate.from_template("{human_input}")])

    conversation = LLMChain(
        llm=client,
        prompt=prompt,
        verbose=False,
    )
    response = conversation.predict(human_input=user_input)
    return response

@app.route('/', methods=['GET', 'POST'])
def patient_input():
    # Initialize session data if not already done
    initialize_chat_history()

    if request.method == 'POST':
        user_input = request.form['user_input']
        
        # Append user message to chat history
        append_to_chat_history("user", user_input)

        # Generate response from the model as a simulated patient
        try:
            bot_response = get_response_patient(user_input, session["chat_history"])
            append_to_chat_history("assistant", bot_response)
        except Exception as e:
            bot_response = f"An error occurred: {e}"
            append_to_chat_history("assistant", bot_response)

        # Correct the URL for the redirect
        return redirect(url_for('patient_input'))  # Redirect to 'patient_input' endpoint

    return render_template('patient.html', chat_history=session["chat_history"])

if __name__ == '__main__':
    app.run(debug=True)
