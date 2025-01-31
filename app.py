import streamlit as st
import speech_recognition as sr
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

# Load environment variables
load_dotenv()

# Initialize the speech recognizer
recognizer = sr.Recognizer()

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

def get_speech_input():
    """Record speech and convert to text"""
    try:
        with sr.Microphone() as source:
            st.write("Let's start speaking...")
            audio = recognizer.listen(source)
            st.write("Your speech is being processed..")
            
            text = recognizer.recognize_google(audio)
            return text
    except Exception as e:
        st.error(f"Error recording speech: {str(e)}")
        return None

def analyze_speech(text):
    """Analyze speech using Groq LLM"""
    prompt = f"""
    Analyze the following English speech:
    "{text}"
    
    Provide your response in the following format:

    1. IMPROVED VERSION (in English):
    [Provide the corrected and improved version of the speech in proper English]

    2. EXPLANATION (in Hinglish):
    - Grammar mistakes ki explanation
    - Word choice ke suggestions
    - Sentence structure mein improvement ke points
    
    3. PRONUNCIATION TIPS (in Hinglish):
    - Specific words ki pronunciation ke tips
    - Overall speaking improvement ke suggestions
    """
    
    response = llm.invoke(prompt)
    return response.content

def chat_with_assistant(question):
    """Chat with the AI assistant about English learning"""
    prompt = f"""
    Answer the following question about English learning in Hinglish:
    "{question}"
    
    Provide a helpful, detailed response that:
    - Explains concepts in simple Hinglish
    - Gives practical examples
    - Provides tips for improvement
    - Uses relatable examples from daily life
    """
    
    response = llm.invoke(prompt)
    return response.content

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Streamlit UI with tabs
tab1, tab2 = st.tabs(["Speech Analysis", "English Learning Chat"])

# Tab 1: Speech Analysis
with tab1:
    st.title("English Speech Improvement Assistant")
    st.write("Click the button below and start speaking in English")

    if st.button("Start Recording"):
        speech_text = get_speech_input()
        
        if speech_text:
            st.write(" You said:")
            st.write(speech_text)
            
            st.write("Your speech is being analyzed...")
            analysis = analyze_speech(speech_text)
            
            st.write(" AI Analysis and Improvements:")
            st.markdown(analysis)

# Tab 2: Chat Interface
with tab2:
    st.title("English Learning Assistant")
    st.write(" Ask your questions - like the meaning of a word, grammar rules, or speaking tips")

    # Chat input
    user_question = st.text_input("Type your question here:", key="user_input")
    if st.button("Ask a question"):
        if user_question:
            # Get AI response
            response = chat_with_assistant(user_question)
            
            # Add to chat history
            st.session_state.chat_history.append(("You", user_question))
            st.session_state.chat_history.append(("Assistant", response))

    # Display chat history
    st.write("Chat History:")
    for role, message in st.session_state.chat_history:
        if role == "You":
            st.markdown(f"**You:** {message}")
        else:
            st.markdown(f"**Assistant:** {message}")

# Sidebar Instructions
st.sidebar.header("How to use the app")  
st.sidebar.write("""  
1. In the Speech Analysis tab:  
   - Click on 'Start Recording'  
   - Speak in English  
   - View the analysis of your speech  

2. In the Chat tab:  
   - Ask any questions about learning English  
   - Inquire about new words  
   - Understand grammar rules  
   - Ask for speaking tips  
""")  
