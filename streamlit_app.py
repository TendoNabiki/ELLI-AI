import streamlit as st
import tempfile
from pathlib import Path

from elli_project_mainbody import load_model, generate_text

st.set_page_config(page_title="ELLI AI", layout="wide")
st.title("ELLI AI — Web Interface")

st.markdown("Simple Streamlit UI to load a checkpoint and generate text with ELLI.")

checkpoint_default = "checkpoints/best_model.pt"
checkpoint_path = st.text_input("Checkpoint path", checkpoint_default)

uploaded = st.file_uploader("Or upload a checkpoint (.pt)", type=["pt"])

if 'model' not in st.session_state:
    st.session_state['model'] = None

if uploaded is not None:
    tmp = Path(tempfile.gettempdir()) / uploaded.name
    with open(tmp, "wb") as f:
        f.write(uploaded.getbuffer())
    checkpoint_path = str(tmp)
    st.success(f"Uploaded checkpoint to {checkpoint_path}")

if st.button("Load model"):
    if not Path(checkpoint_path).exists():
        st.error(f"Checkpoint not found: {checkpoint_path}")
    else:
        with st.spinner("Loading model — this may take a minute..."):
            model = load_model(checkpoint_path)
            st.session_state['model'] = model
            st.success("Model loaded")

if st.session_state.get('model') is not None:
    st.subheader("Generate")
    prompt = st.text_area("Prompt", value="Once upon a time,", height=200)
    max_new_tokens = st.slider("Max new tokens", 1, 1024, 200)
    col1, col2 = st.columns([1,3])
    with col1:
        if st.button("Generate"):
            with st.spinner("Generating..."):
                out = generate_text(st.session_state['model'], prompt, max_new_tokens=max_new_tokens)
                st.session_state['last_output'] = out
    with col2:
        st.subheader("Output")
        out = st.session_state.get('last_output', "")
        st.text_area("", value=out, height=300)

st.markdown("---")
st.info("Run: `streamlit run streamlit_app.py`")
