"""The Streamlit interface for ELLI (Evolving Large Language Intelligence)."""

import base64
import random
import time
from html import escape
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


ROOT = Path(__file__).parent
st.set_page_config(
    page_title="ELLI | Evolving Large Language Intelligence", page_icon="✦", layout="wide"
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
        :root { --ink:#181b1a; --panel:#202523; --mint:#1ee5aa; --gold:#ffcb05; --soft:#b9c0bc; }
        .stApp { background:radial-gradient(circle at 25% 12%, #2a3530 0, #181b1a 32rem); color:#f5f7f5; }
        [data-testid="stHeader"] { background:transparent; } #MainMenu, footer { visibility:hidden; }
        .block-container { max-width:1400px; padding:2.8rem 3.5rem 2rem; }
        .elli-brand { display:flex; align-items:flex-end; gap:1.15rem; margin:.2rem 0 2.5rem .25rem; }
        .elli-brand h1 { font:700 clamp(4.8rem,10vw,8.2rem)/.72 "Space Grotesk",sans-serif; letter-spacing:0; margin:0; color:#f2f4f2; }
        .elli-brand p { font:600 1.05rem/1.22 "Space Grotesk",sans-serif; color:#c5cbc7; margin:0 0 1.1rem 1rem; max-width:11rem; }
        .chat-shell { background:rgba(32,37,35,.88); border:2px solid var(--mint); border-radius:3.2rem; padding:1.5rem 1.6rem 1.2rem; min-height:32rem; box-shadow:0 0 32px rgba(30,229,170,.06); }
        .chat-title { display:flex; justify-content:space-between; align-items:center; color:#e9efea; font:500 .77rem "DM Mono",monospace; letter-spacing:.1em; text-transform:uppercase; margin:0 .5rem 1.2rem; }
        .online-dot { display:inline-block; width:.55rem; height:.55rem; background:var(--mint); border-radius:50%; margin-right:.45rem; box-shadow:0 0 12px var(--mint); }
        .message { width:fit-content; max-width:76%; padding:1rem 1.2rem; margin:.85rem .45rem; border-radius:1.35rem; font:500 1rem/1.45 "Space Grotesk",sans-serif; }
        .assistant-message { background:#29302d; border:1px solid var(--mint); border-bottom-left-radius:.35rem; color:#f4f7f4; }
        .user-message { background:transparent; border:1px solid #86aaa0; border-bottom-right-radius:.35rem; color:var(--gold); margin-left:auto; }
        .message-label { display:block; font:500 .65rem "DM Mono",monospace; letter-spacing:.1em; opacity:.72; text-transform:uppercase; margin-bottom:.38rem; }
# .info-card { border:2px solid var(--mint); border-radius:2.4rem; padding:1.6rem 1.25rem; background:rgba(32,37,35,.72); margin-top:1.4rem; text-align:center; }

        .info-card {padding:1.6rem 1.25rem; margin-top:1.4rem; text-align:center; }
        .info-card h3 { font:600 1.15rem "Space Grotesk",sans-serif; margin:.25rem 0 1rem; color:#f5f7f5; }
        .info-toggle button { width:3.35rem; height:3.35rem; padding:0!important; border:2px solid #edf1ee!important; border-radius:50%!important; background:var(--ink)!important; color:#f5f7f5!important; font:700 2rem/1 Georgia,serif!important; }
        .info-toggle button:hover { border-color:var(--mint)!important; color:var(--mint)!important; }
        .info-link button { width:100%; background:transparent!important; border:0!important; color:#dfe5e1!important; font:500 1rem "Space Grotesk",sans-serif!important; padding:.55rem 0!important; }
        .info-link button:hover { color:var(--mint)!important; transform:translateX(3px); }
        [data-testid="stChatInput"] { border:2px solid var(--mint)!important; background:#202523!important; border-radius:1.5rem!important; padding:.38rem .55rem!important; margin-top:1.3rem; }
        [data-testid="stChatInput"] textarea { color:var(--gold)!important; font:500 1.1rem "Space Grotesk",sans-serif!important; }
        [data-testid="stChatInput"] textarea::placeholder { color:#a8b0ab!important; }
        [data-testid="stChatInput"] button { background:var(--mint); border-radius:50%; }
        [data-testid="stChatInput"] button svg { fill:#13221b; }
        .clear-button button, .back-button button { border-color:#61716a!important; color:#b9c0bc!important; border-radius:1rem!important; font:.75rem "DM Mono",monospace!important; }
        .detail-panel { background:rgba(32,37,35,.88); border:2px solid var(--mint); border-radius:2.5rem; padding:2rem 2.3rem; margin-top:1rem; }
        .detail-panel h2 { color:var(--gold); font:600 2rem "Space Grotesk",sans-serif; margin-top:0; }
        .detail-panel p, .detail-panel li { font:400 1rem/1.6 "Space Grotesk",sans-serif; color:#e2e8e3; }
        @media (max-width:800px) { .block-container{padding:2rem 1rem;} .elli-brand h1{font-size:5.4rem;} .elli-brand p{font-size:.85rem;margin-left:.3rem;} .chat-shell{min-height:24rem;border-radius:2rem;} .message{max-width:92%;} }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am ELLI. What would you like to explore today?"}]
if "info_open" not in st.session_state:
    st.session_state.info_open = False
if "page" not in st.session_state:
    st.session_state.page = "home"


def go_to(page: str) -> None:
    st.session_state.page = page
    st.rerun()


def show_detail_page() -> None:
    with st.container():
        st.markdown('<div class="back-button">', unsafe_allow_html=True)
        if st.button("← Back to ELLI", key="back_home"):
            go_to("home")
        st.markdown("</div>", unsafe_allow_html=True)

    page = st.session_state.page
    titles = {"stats": "Stats for nerds", "creators": "Learn about the creators", "proposal": "The original idea", "sources": "Works cited"}
    st.markdown(f'<div class="detail-panel"><h2>{titles[page]}</h2>', unsafe_allow_html=True)

    if page == "stats":
        st.markdown("### ELLI’s thinking layer")
        st.info("The updated ELLI thinking-layer code has not been added yet. This page is ready for model statistics, tokenization details, training metrics, and an architecture diagram when it is available.")
    elif page == "creators":
        st.markdown("### Team Eightfold")
        st.markdown((ROOT / "README2.md").read_text(encoding="utf-8"))
    elif page == "proposal":
        proposal = ROOT / "_Proposal of ELLI.pdf"
        st.markdown("This proposal describes the original concept behind ELLI.")
        pdf_data = base64.b64encode(proposal.read_bytes()).decode("utf-8")
        components.html(f'<iframe src="data:application/pdf;base64,{pdf_data}" width="100%" height="650" style="border:0;border-radius:12px;"></iframe>', height=665)
        st.download_button("Download the ELLI proposal (PDF)", proposal.read_bytes(), file_name=proposal.name, mime="application/pdf")
    elif page == "sources":
        st.markdown("The datasets, research, tools, and acknowledgements used for ELLI are listed below.")
        st.code((ROOT / "SimplifiedSources.txt").read_text(encoding="utf-8"), language="bibtex")
    st.markdown("</div>", unsafe_allow_html=True)


def show_info_card() -> None:
    st.markdown('<div class="info-card"><h3>About ELLI</h3>', unsafe_allow_html=True)
    for label, page in [("Stats for nerds", "stats"), ("Learn about the creators", "creators"), ("The original idea", "proposal"), ("Works cited", "sources")]:
        st.markdown('<div class="info-link">', unsafe_allow_html=True)
        if st.button(label, key=f"info_{page}"):
            go_to(page)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def show_chat() -> None:
    # Initialize the confirmation state variable if it doesn't exist
    if "confirm_clear" not in st.session_state:
        st.session_state.confirm_clear = False

    # Render conversation log
    conversation = '<div class="chat-shell"><div class="chat-title"><span><span class="online-dot"></span>ELLI conversation</span><span>v0.1</span></div>'
    for message in st.session_state.messages:
        style = "assistant-message" if message["role"] == "assistant" else "user-message"
        label = "ELLI reply" if message["role"] == "assistant" else "Your message"
        conversation += f'<div class="message {style}"><span class="message-label">{label}</span>{escape(message["content"])}</div>'
    st.markdown(conversation + "</div>", unsafe_allow_html=True)

    # Place the clear button directly above the chat input box
    st.markdown('<div class="clear-button">', unsafe_allow_html=True)
    
    if not st.session_state.confirm_clear:
        # Initial button state
        if st.button("Clear conversation"):
            st.session_state.confirm_clear = True
            st.rerun()
    else:
        # Confirmation state with side-by-side choices
        st.write("⚠️ Are you sure you want to clear this conversation?")
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Yes, Clear", key="confirm_yes"):
                st.session_state.messages = [{"role": "assistant", "content": "Conversation reset. How can I help?"}]
                st.session_state.confirm_clear = False
                st.rerun()
        with col2:
            if st.button("Cancel", key="confirm_no"):
                st.session_state.confirm_clear = False
                st.rerun()
                
    st.markdown("</div>", unsafe_allow_html=True)

    # Chat input field
# 1. Chat input field
    if prompt := st.chat_input("Ask ELLI anything…"):
        st.session_state.confirm_clear = False
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("ELLI is thinking…"):
            try:
                # 2. Setup tokenization and device
                from tokenizers import Tokenizer
                import torch
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                tok = Tokenizer.from_file("my_custom_tokenizer.json")
                
                # 3. Initialize model architecture and load trained weights
                # (Make sure the ELLI class definition is accessible in this file)
                model = ELLI(
                    vocab_size=tok.get_vocab_size(), 
                    d_model=1024, 
                    n_heads=16, 
                    n_layers=20, 
                    block_size=128
                ).to(device)
                
                checkpoint = torch.load("checkpoints/best_model.pt", map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                
                # 4. Convert user prompt to tokens and generate a response
                encoded_input = torch.tensor([tok.encode(prompt).ids], dtype=torch.long, device=device)
                generated_tokens = model.generate(encoded_input, max_new_tokens=150)
                
                # Decode tokens back into readable text
                ai_reply = tok.decode(generated_tokens[0].tolist())
                
                # Strip out the initial prompt if your model repeats it
                if ai_reply.startswith(prompt):
                    ai_reply = ai_reply[len(prompt):].strip()
                    
            except Exception as e:
                # Fallback message if the model hasn't been trained/saved yet
                ai_reply = f"System Error: Could not load the model checkpoint. ({str(e)})"

        st.session_state.messages.append({"role": "assistant", "content": ai_reply})
        st.rerun()


if st.session_state.page != "home":
    show_detail_page()
else:
    header, icon = st.columns([15, 1])
    with header:
        st.markdown('<div class="elli-brand"><h1>ELLI</h1><p>Evolving<br>Large<br>Language<br>Intelligence</p></div>', unsafe_allow_html=True)
    with icon:
        st.markdown('<div class="info-toggle">', unsafe_allow_html=True)
        if st.button("ℹ", key="toggle_info", help="Show or hide ELLI information"):
            st.session_state.info_open = not st.session_state.info_open
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.info_open:
        chat_column, info_column = st.columns([3.6, 1], gap="large")
        with chat_column:
            show_chat()
        with info_column:
            show_info_card()
    else:
        show_chat()