import streamlit as st
import os
from groq import Groq

st.set_page_config(page_title="AI Math Mentor", page_icon="🧮", layout="wide")
st.title("🧮 AI Math Mentor")
st.markdown("*Solve JEE-style math problems with AI assistance*")

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("ERROR: GROQ_API_KEY not set!")
    st.stop()

client = Groq(api_key=groq_api_key)

with st.sidebar:
    st.header("Input Mode")
    st.radio("Choose:", ["Text Input"])

st.header("Problem Input")
raw_input = st.text_area("Type your math problem:", height=120)

if st.button("Solve Problem", type="primary"):
    if not raw_input.strip():
        st.error("Please enter a problem!")
    else:
        with st.spinner("Solving..."):
            try:
                response = client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=[
                        {"role": "system", "content": "You are an expert math tutor. Solve step-by-step."},
                        {"role": "user", "content": raw_input}
                    ],
                    temperature=0.3
                )
                solution = response.choices[0].message.content
                st.success("✅ Solution found!")
                st.markdown(solution)
            except Exception as e:
                st.error(f"Error: {str(e)}")

st.divider()
st.markdown("**AI Math Mentor** | Powered by Groq")
