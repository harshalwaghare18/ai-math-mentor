import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from pathlib import Path

# Import agents
from agents.parser_agent import ParserAgent
from agents.router_agent import RouterAgent
from agents.solver_agent import SolverAgent
from agents.verifier_agent import VerifierAgent
from agents.explainer_agent import ExplainerAgent
from utils.ocr_handler import OCRHandler
from utils.audio_handler import AudioHandler
from memory.memory_manager import MemoryManager

# Load environment
load_dotenv()

# Initialize session state
if "conversation_log" not in st.session_state:
    st.session_state.conversation_log = []

# Page config
st.set_page_config(
    page_title="AI Math Mentor",
    page_icon="🧮",
    layout="wide"
)

# Initialize components
@st.cache_resource
def init_components():
    return {
        "parser": ParserAgent(),
        "router": RouterAgent(),
        "solver": SolverAgent(),
        "verifier": VerifierAgent(),
        "explainer": ExplainerAgent(),
        "ocr": OCRHandler(),
        "audio": AudioHandler(),
        "memory": MemoryManager()
    }

components = init_components()

# UI Layout
st.title("🧮 AI Math Mentor")
st.markdown("*Solve JEE-style math problems with AI assistance*")

# Sidebar: Input Mode Selection
with st.sidebar:
    st.header("📥 Input Mode")
    input_mode = st.radio(
        "Choose input method:",
        ["📝 Text", "🖼️ Image", "🎤 Audio"]
    )
    
    st.divider()
    st.header("📋 Recent Attempts")
    if st.session_state.conversation_log:
        for i, attempt in enumerate(st.session_state.conversation_log[-5:]):
            st.text(f"{i+1}. {attempt['problem'][:50]}...")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Problem Input")
    
    # Input handling
    raw_input = None
    input_metadata = {}
    
    if input_mode == "📝 Text":
        raw_input = st.text_area(
            "Type your math problem:",
            height=150,
            placeholder="e.g., Find the derivative of x² + 3x + 2"
        )
    
    elif input_mode == "🖼️ Image":
        uploaded_image = st.file_uploader("Upload image (JPG/PNG):", type=["jpg", "png"])
        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded image", use_column_width=True)
            
            # OCR extraction
            with st.spinner("🔍 Extracting text from image..."):
                image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
                with open(image_path, "wb") as f:
                    f.write(uploaded_image.getbuffer())
                
                ocr_result = components["ocr"].extract_text_from_image(image_path)
                raw_input = ocr_result["text"]
                input_metadata = ocr_result
        
        if raw_input:
            st.subheader("Extracted Text (Review & Edit)")
            raw_input = st.text_area(
                "Edit if needed:",
                value=raw_input,
                height=100,
                key="ocr_text"
            )
            
            if ocr_result.get("warnings"):
                for warning in ocr_result["warnings"]:
                    st.warning(warning)
    
    elif input_mode == "🎤 Audio":
        uploaded_audio = st.file_uploader("Upload audio file:", type=["wav", "mp3", "m4a"])
        if uploaded_audio:
            st.audio(uploaded_audio)
            
            with st.spinner("🎵 Transcribing audio..."):
                audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
                with open(audio_path, "wb") as f:
                    f.write(uploaded_audio.getbuffer())
                
                audio_result = components["audio"].transcribe_audio(audio_path)
                raw_input = audio_result["text"]
                input_metadata = audio_result
        
        if raw_input:
            st.subheader("Transcribed Text (Review & Edit)")
            raw_input = st.text_area(
                "Edit if needed:",
                value=raw_input,
                height=100,
                key="audio_text"
            )
            
            if input_metadata.get("math_phrases_detected"):
                st.info(f"📐 Detected math terms: {', '.join(input_metadata['math_phrases_detected'])}")

# Solve button
if st.button("🚀 Solve Problem", key="solve_btn", type="primary"):
    if not raw_input or raw_input.strip() == "":
        st.error("❌ Please provide a problem first!")
    else:
        # Create tabs for step-by-step process
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "1️⃣ Parse", "2️⃣ Route", "3️⃣ Solve", "4️⃣ Verify", "5️⃣ Explain"
        ])
        
        # STEP 1: Parse
        with tab1:
            with st.spinner("🔄 Parsing problem..."):
                parsed = components["parser"].parse(raw_input)
                st.success("✅ Problem parsed")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write("**Topic:**", parsed.topic)
                    st.write("**Variables:**", ", ".join(parsed.variables))
                
                with col_b:
                    st.write("**Constraints:**", ", ".join(parsed.constraints))
                
                if parsed.needs_clarification:
                    st.warning("⚠️ Clarification needed:")
                    for q in parsed.clarification_questions:
                        st.write(f"- {q}")
                        user_answer = st.text_input(f"Answer: {q}", key=f"clarify_{q}")
        
        # STEP 2: Route
        with tab2:
            with st.spinner("🧭 Routing request..."):
                parsed_dict = {
                    "problem_text": parsed.problem_text,
                    "topic": parsed.topic,
                    "variables": parsed.variables,
                    "constraints": parsed.constraints
                }
                route_info = components["router"].route(parsed_dict)
                st.success("✅ Route determined")
                
                st.write("**Strategy:**", route_info.get("strategy"))
                st.write("**Use RAG?**", route_info.get("use_rag"))
                st.write("**Tools needed:**", ", ".join(route_info.get("computational_tools", [])))
                st.write("**Confidence:**", f"{route_info.get('confidence', 0):.0%}")
        
        # STEP 3: Solve
        with tab3:
            with st.spinner("🧠 Solving..."):
                solution = components["solver"].solve(parsed_dict, route_info)
                st.success("✅ Solution found")
                
                # Display solution steps
                st.subheader("Solution Steps:")
                for step in solution.get("steps", []):
                    st.write(f"**Step {step['step']}:** {step['description']}")
                    st.code(step['calculation'])
                
                st.subheader("Final Answer:")
                st.markdown(f"### {solution.get('final_answer')}")
                
                st.write(f"**Confidence:** {solution.get('confidence', 0):.0%}")
                
                if solution.get("retrieved_sources"):
                    st.subheader("📚 Retrieved Knowledge:")
                    for doc, score in solution["retrieved_sources"]:
                        st.info(f"*Relevance: {score:.0%}*\n\n{doc.page_content[:300]}...")
        
        # STEP 4: Verify
        with tab4:
            with st.spinner("✔️ Verifying solution..."):
                verification = components["verifier"].verify(
                    raw_input,
                    json.dumps(solution, indent=2)
                )
                st.success("✅ Verification complete")
                
                if verification.get("is_correct"):
                    st.success(f"✅ Solution is correct! (Confidence: {verification.get('confidence', 0):.0%})")
                else:
                    st.error(f"❌ Solution may have issues (Confidence: {verification.get('confidence', 0):.0%})")
                
                if verification.get("issues"):
                    st.subheader("Issues Found:")
                    for issue in verification["issues"]:
                        st.warning(f"- {issue}")
                
                if verification.get("suggestions"):
                    st.subheader("Suggestions:")
                    for suggestion in verification["suggestions"]:
                        st.info(f"- {suggestion}")
                
                # HITL trigger
                if verification.get("needs_human_review"):
                    st.warning("⚠️ This solution needs human review. Please verify the answer.")
        
        # STEP 5: Explain
        with tab5:
            with st.spinner("📖 Generating explanation..."):
                explanation = components["explainer"].explain(raw_input, solution)
                st.success("✅ Explanation ready")
                st.markdown(explanation)
        
        # Feedback section
        st.divider()
        st.subheader("📝 Feedback (HITL)")
        
        feedback_col1, feedback_col2 = st.columns(2)
        with feedback_col1:
            if st.button("✅ This solution is correct"):
                components["memory"].save_attempt(
                    raw_input,
                    json.dumps(solution, indent=2),
                    feedback="correct",
                    verified=True
                )
                st.success("✅ Feedback saved. Solution added to knowledge base.")
        
        with feedback_col2:
            if st.button("❌ This solution is incorrect"):
                correction = st.text_area("Provide the correct answer or explanation:")
                if correction:
                    components["memory"].save_attempt(
                        raw_input,
                        json.dumps(solution, indent=2),
                        feedback=f"incorrect: {correction}",
                        verified=False
                    )
                    st.info("📚 Correction saved for future learning.")
        
        # Check for similar problems
        st.divider()
        st.subheader("🔍 Similar Solved Problems")
        similar_problems = components["memory"].find_similar_problems(raw_input)
        
        if similar_problems:
            for i, similar in enumerate(similar_problems[:3]):
                with st.expander(f"Similar problem (Match: {similar['similarity']:.0%})"):
                    st.write("**Problem:**", similar['problem'][:200])
                    st.write("**Solution approach:**", similar['solution'][:300])
        else:
            st.info("No similar problems found yet.")
        
        # Log conversation
        st.session_state.conversation_log.append({
            "problem": raw_input,
            "timestamp": datetime.now().isoformat()
        })

# Footer
st.divider()
st.markdown("""
---
**AI Math Mentor v1.0** | Built with Streamlit + LangChain + OpenAI
📦 [GitHub](https://github.com) | 🚀 [Deploy with HuggingFace Spaces](https://huggingface.co/spaces)
""")
