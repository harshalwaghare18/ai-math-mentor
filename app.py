import streamlit as st
import os
from groq import Groq

st.set_page_config(page_title="AI Math Mentor", page_icon="🧮", layout="wide")
st.title("🧮 AI Math Mentor")
st.markdown("*Solve JEE-style math problems with AI assistance*")

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("ERROR: GROQ_API_KEY not set!")
    st.info("**Local:** Add to `.env` file\n**Cloud:** Add to Streamlit Secrets")
    st.stop()

client = Groq(api_key=groq_api_key)

st.sidebar.header("Input Mode")
st.sidebar.radio("Choose:", ["Text Input"])

st.header("Problem Input")
raw_input = st.text_area("Type your math problem:", height=120)

if st.button("Solve Problem", type="primary"):
    if not raw_input.strip():
        st.error("Please enter a problem!")
    else:
        with st.spinner("Solving with Groq..."):
            try:
                response = client.chat.completions.create(
                    model="llama3-70b-8192",

                    messages=[
                        {"role": "system", "content": "You are an expert math tutor. Solve step-by-step clearly."},
                        {"role": "user", "content": raw_input}
                    ],
                    temperature=0.1
                )
                st.success("✅ Solution!")
                st.markdown(response.choices[0].message.content)
            except Exception as e:
                st.error(f"Error: {str(e)}")

st.markdown("---")
st.markdown("**Powered by Groq** | Free & Fast")
