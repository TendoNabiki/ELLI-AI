import importlib
import tempfile
from pathlib import Path

import streamlit as st


def build_demo_response(prompt: str) -> str:
    return (
        "Demo mode: the AI backend is unavailable right now. "
        f"Your prompt was: {prompt}\n"
        "The interface is still visible so you can continue building and testing the UI."
    )


try:
    from elli_project_mainbody import load_model, generate_text
except Exception as exc:
    load_model = None
    generate_text = None
    st.set_option("client.showErrorDetails", False)


st.set_page_config(page_title="ELLI AI", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --bg: #141414;
        --panel: #1b1b1b;
        --panel-2: #222222;
        --text: #f3f7f1;
        --muted: #a4b29d;
        --accent: #10eb9e;
        --accent-2: #3cff00;
    }

    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background: var(--bg);
        color: var(--text);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #171717 0%, #0f0f0f 100%);
        border-right: 1px solid rgba(141, 255, 0, 0.22);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3, h4 {
        color: var(--accent);
    }

    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: var(--panel);
        color: var(--text);
        border: 1px solid rgba(141, 255, 0, 0.28);
        border-radius: 8px;
    }

    .stButton > button {
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
        color: #061105;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        box-shadow: 0 0 12px rgba(141, 255, 0, 0.25);
    }

    .stButton > button:hover {
        box-shadow: 0 0 16px rgba(141, 255, 0, 0.45);
    }

    .stAlert, .stSuccess, .stError, .stInfo {
        border-radius: 10px;
    }

    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid rgba(141, 255, 0, 0.18);
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
    if load_model is None or generate_text is None:
        st.warning("AI backend unavailable. Running in demo mode so the interface remains visible.")
        st.session_state['model'] = "demo"
    elif not Path(checkpoint_path).exists():
        st.error(f"Checkpoint not found: {checkpoint_path}")
    else:
        with st.spinner("Loading model — this may take a minute..."):
            try:
                model = load_model(checkpoint_path)
                st.session_state['model'] = model
                st.success("Model loaded")
            except Exception as exc:
                st.warning(f"Model could not be loaded: {exc}")
                st.session_state['model'] = "demo"

if st.session_state.get('model') is not None:
    st.subheader("Generate")
    prompt = st.text_area("Prompt", value="Once upon a time,", height=200)
    max_new_tokens = st.slider("Max new tokens", 1, 1024, 200)
    col1, col2 = st.columns([1,3])
    with col1:
        if st.button("Generate"):
            with st.spinner("Generating..."):
                if st.session_state['model'] == "demo":
                    out = build_demo_response(prompt)
                else:
                    out = generate_text(st.session_state['model'], prompt, max_new_tokens=max_new_tokens)
                st.session_state['last_output'] = out
    with col2:
        st.subheader("Output")
        out = st.session_state.get('last_output', "")
        st.text_area("", value=out, height=300)

st.markdown("---")
st.info("Run: `streamlit run streamlit_app.py`")
