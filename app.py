import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

st.set_page_config(page_title="AI Math Mentor", page_icon="🧮", layout="wide")
st.title("🧮 AI Math Mentor")
st.markdown("*Solve JEE-style math problems with AI assistance*")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("ERROR: OPENAI_API_KEY not set!")
    st.stop()

client = OpenAI(api_key=api_key)

with st.sidebar:
    st.header("Input Mode")
    input_mode = st.radio("Choose:", ["Text Input"])

st.header("Problem Input")
raw_input = st.text_area("Type your math problem:", height=120)

if st.button("Solve Problem", type="primary"):
    if not raw_input:
        st.error("Please enter a problem!")
    else:
        with st.spinner("Solving..."):
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",

                    messages=[
                        {"role": "system", "content": "You are a math tutor. Solve step-by-step."},
                        {"role": "user", "content": raw_input}
                    ],
                    temperature=0
                )
                
                solution = response.choices[0].message.content
                st.success("Solution found!")
                st.markdown(solution)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

st.divider()
st.markdown("**AI Math Mentor** | Powered by GPT-4")
