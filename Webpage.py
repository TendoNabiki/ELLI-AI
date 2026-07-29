import re
import uuid
from html import escape
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from groq import Groq

# --- PAGE CONFIGURATION ---
ROOT = Path(__file__).parent
st.set_page_config(
    page_title="ELLI | Evolving Large Language Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- 1. INITIALIZE API CLIENTS ---
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# --- 2. GLOBAL UI SETTINGS ---
components.html(
    """
    <script>
    const parentDoc = window.parent.document;
    if (!parentDoc.getElementById('elli-lottie-bg')) {
        const script = parentDoc.createElement('script');
        script.src = "https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js";
        parentDoc.head.appendChild(script);

        const container = parentDoc.createElement('div');
        container.id = 'elli-lottie-bg';
        container.style.position = 'fixed';
        container.style.top = '0';
        container.style.left = '0';
        container.style.width = '100vw';
        container.style.height = '100vh';
        container.style.zIndex = '0';
        container.style.pointerEvents = 'none';
        container.style.opacity = '0.12';

        script.onload = () => {
            container.innerHTML = `
                <lottie-player src="https://lottie.host/80f7602e-13cb-4a11-8ec8-8cf81e3c8ca4/4xJ1t2T0B8.json" background="transparent" speed="0.6" style="width: 100%; height: 100%;" loop autoplay></lottie-player>
            `;
        };
        const stApp = parentDoc.querySelector('[data-testid="stAppViewContainer"]') || parentDoc.body;
        stApp.appendChild(container);
    }
    </script>
    """,
    height=0,
    width=0,
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
        :root { --ink:#181b1a; --panel:#202523; --mint:#1ee5aa; --gold:#ffcb05; --soft:#b9c0bc; }
        [data-testid="stHeader"] { background:transparent; } #MainMenu, footer { visibility:hidden; }
        .block-container { max-width:1400px; padding:2rem 3.5rem 2rem; position: relative; z-index: 10; }

        .elli-brand { display:flex; align-items:flex-end; gap:0.8rem; margin:.2rem 0 1rem 0; }
        .elli-brand h1 { font:700 clamp(2.5rem,6vw,4rem)/.72 "Space Grotesk",sans-serif; letter-spacing:0; margin:0; color:#f2f4f2; }
        .elli-brand p { font:600 0.8rem/1.22 "Space Grotesk",sans-serif; color:#c5cbc7; margin:0 0 0.3rem 0; max-width:11rem; }

        .chat-shell { background:rgb(28, 36, 34); border:1px solid var(--mint); padding:1.5rem 1.6rem 1.2rem; min-height:32rem; box-shadow:0 0 32px rgba(30,229,170,.06); margin-top: 1rem; }
        .chat-title { display:flex; justify-content:space-between; align-items:center; color:#e9efea; font:500 .77rem "DM Mono",monospace; letter-spacing:.1em; text-transform:uppercase; margin:0 .5rem 1.2rem; }
        .message { width:fit-content; max-width:76%; padding:1rem 1.2rem; margin:.85rem .45rem; border-radius:1.35rem; font:500 1rem/1.45 "Space Grotesk",sans-serif; }
        .assistant-message { background:#29302d; border:1px solid var(--mint); border-bottom-left-radius:.35rem; color:#f4f7f4; }
        .user-message { background:transparent; border:1px solid #86aaa0; border-bottom-right-radius:.35rem; color:var(--mint); margin-left:auto; }
        .message-label { display:block; font:500 .65rem "DM Mono",monospace; letter-spacing:.1em; opacity:.72; text-transform:uppercase; margin-bottom:.38rem; }

        [data-testid="stChatInput"] {
            border: none !important;
            border-radius: 1.5rem !important;
            margin-top: 1.3rem;
            background: #202523 !important;
            box-shadow: none !important;
        }
        [data-testid="stChatInput"]:focus,
        [data-testid="stChatInput"]:focus-within,
        [data-testid="stChatInput"]:active {
            border: none !important;
            box-shadow: none !important;
        }
        [data-testid="stChatInput"] > div {
            border: none !important;
            box-shadow: none !important;
        }
        [data-testid="stChatInput"] textarea { color:var(--mint)!important; font:500 1.1rem "Space Grotesk",sans-serif!important; }
        [data-testid="stChatInput"] button { background:var(--mint); border-radius:50%; transition:background-color .2s ease, opacity .2s ease; }
        [data-testid="stChatInput"] button:disabled { background:#6b756f!important; opacity:.8; cursor:not-allowed!important; box-shadow:none!important; }
        [data-testid="stChatInput"] button:disabled svg { fill:#f4f7f4; }
        [data-testid="stChatInput"] button svg { fill:#13221b; }
        .clear-button button { border-color:#61716a!important; color:#b9c0bc!important; border-radius:1rem!important; font:.75rem "DM Mono",monospace!important; }

        @media (max-width:800px) { .block-container{padding:2rem 1rem;} .elli-brand h1{font-size:3rem;} .elli-brand p{font-size:.7rem;} .chat-shell{min-height:24rem;border-radius:2rem;} .message{max-width:92%;} }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- 3. SESSION STATE INIT (no login, no database) ---
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am ELLI."}]
if "confirm_clear" not in st.session_state:
    st.session_state.confirm_clear = False

# --- 4. SIDEBAR ---
st.sidebar.markdown("### ELLI")
if st.sidebar.button("+ New Chat", use_container_width=True):
    st.session_state.current_chat_id = str(uuid.uuid4())
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am ELLI. Nice to meet you."}]
    st.rerun()


def show_chat() -> None:
    conversation = '<div class="chat-shell"><div class="chat-title"><span><span class="online-dot"></span>ELLI conversation</span><span>v4.6.8</span></div>'

    # 1. RENDER CHAT INTERFACE & THINKING LAYER
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            content = message["content"]
            think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)

            if think_match:
                thinking_text = think_match.group(1).strip()
                final_answer = content.replace(think_match.group(0), "").strip()

                formatted_content = f'''
                <details style="margin-bottom: 12px; cursor: pointer;">
                    <summary style="font-size: 0.75rem; color: #1ee5aa; font-family: 'DM Mono', monospace; text-transform: uppercase;"> View ELLI Cognition</summary>
                    <div style="font-size: 0.9rem; color: #a8b0ab; margin-top: 8px; padding-left: 12px; border-left: 2px solid rgba(30,229,170,.4); white-space: pre-wrap; font-family: 'DM Mono', monospace;">{escape(thinking_text)}</div>
                </details>
                <div style="white-space: pre-wrap;">{escape(final_answer)}</div>
                '''
            else:
                formatted_content = f'<div style="white-space: pre-wrap;">{escape(content)}</div>'

            conversation += f'<div class="message assistant-message"><span class="message-label">ELLI reply</span>{formatted_content}</div>'
        else:
            conversation += f'<div class="message user-message"><span class="message-label">Your message</span><div style="white-space: pre-wrap;">{escape(message["content"])}</div></div>'

    st.markdown(conversation + "</div>", unsafe_allow_html=True)

    # 2. CLEAR CONVERSATION LOGIC
    st.markdown('<div class="clear-button">', unsafe_allow_html=True)
    if not st.session_state.confirm_clear:
        if st.button("Clear conversation", key="clear_chat_init_btn"):
            st.session_state.confirm_clear = True
            st.rerun()
    else:
        st.write("WARNING: Your chat will be lost forever! (A long time!)")
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Yes, Clear", key="confirm_yes"):
                st.session_state.current_chat_id = str(uuid.uuid4())
                st.session_state.messages = [{"role": "assistant", "content": "Conversation reset. How can I help?"}]
                st.session_state.confirm_clear = False
                st.rerun()
        with col2:
            if st.button("Cancel", key="confirm_no"):
                st.session_state.confirm_clear = False
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # 3. AI GENERATION (no database save)
    if prompt := st.chat_input("Ask ELLI anything…"):
        st.session_state.confirm_clear = False
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

    if st.session_state.messages[-1]["role"] == "user":
        with st.spinner("ELLI is thinking…"):
            try:
                system_instruction = {
                    "role": "system",
                    "content": "You are ELLI(Evolving Language Learning Intelligence), a hyper-adaptable AI agent. For every user message, you MUST and only output your internal thoughts and logic process wrapped exactly inside <think>...</think> tags BEFORE providing your final response to the user."
                }

                api_messages = [system_instruction] + st.session_state.messages

                chat_completion = groq_client.chat.completions.create(
                    messages=api_messages,
                    model="llama-3.1-8b-instant",
                    temperature=0.7,
                    max_tokens=1500,
                )
                ai_reply = chat_completion.choices[0].message.content
            except Exception as e:
                ai_reply = f"Error connecting to the model: {str(e)}"

            st.session_state.messages.append({"role": "assistant", "content": ai_reply})
            st.rerun()


# --- 5. Render Header ---
st.markdown(
    '''
    <div class="elli-brand">
        <h1>ELLI</h1>
        <p>Evolving<br>Large<br>Language<br>Intelligence</p>
    </div>
    ''',
    unsafe_allow_html=True
)

show_chat()
