import base64
import random
import time
import uuid
import re
from html import escape
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from groq import Groq
import auth

# --- PAGE CONFIGURATION ---
ROOT = Path(__file__).parent
st.set_page_config(
    page_title="ELLI | Evolving Large Language Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- 1. INITIALIZE API CLIENTS ---
groq_client = Groq(api_key=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)

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
        
        /* FIX FOR BORDER GLITCH (REMOVED ENTIRELY) */
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
        /* Override Streamlit's inner focused div */
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


# --- 3. AUTHENTICATION (ISOLATED PER USER) ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = str(uuid.uuid4())


# --- 4. LOGIN PAGE FUNCTION ---
def show_login_page():
    st.markdown("<h2 style='text-align: center; color: #a8c7fa; padding-top: 5rem;'>Welcome to ELLI</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab1, tab2, tab3 = st.tabs(["Login", "Sign Up", "Forgot Password"])
        
        with tab1:
            with st.form("login_form"):
                login_email = st.text_input("Email", key="login_email_input")
                login_pass = st.text_input("Password", type="password", key="login_pass_input")
                submitted = st.form_submit_button("Login", key="login_submit_btn")
                
                if submitted:
                    with st.spinner("Authenticating..."):
                        success, result = auth.verify_user(login_email, login_pass)
                        if success:
                            st.session_state.logged_in = True
                            st.session_state.user_email = login_email
                            st.session_state.current_chat_id = str(uuid.uuid4())
                            st.rerun()
                        else:
                            st.error(result)
                        
        with tab2:
            with st.form("signup_form"):
                new_email = st.text_input("Email", key="signup_email_input")
                new_pass = st.text_input("Choose a Password", type="password", key="signup_pass_input")
                confirm_pass = st.text_input("Confirm Password", type="password", key="signup_confirm_pass_input")
                signup_submitted = st.form_submit_button("Create Account", key="signup_submit_btn")
                
                if signup_submitted:
                    if new_pass != confirm_pass:
                        st.error("Passwords do not match.")
                    else:
                        with st.spinner("Creating account..."):
                            success, message = auth.create_user(new_email, new_pass)
                            if success:
                                st.success(message)
                            else:
                                if "rate limit" in message.lower() or "429" in message:
                                    st.error("Rate limit exceeded. Please wait or disable email confirmation in Supabase.")
                                else:
                                    st.error(message)

        with tab3:
            with st.form("forgot_password_form"):
                st.markdown("Enter your email address to receive a password reset link.")
                reset_email = st.text_input("Email", key="reset_email_input")
                reset_submitted = st.form_submit_button("Send Reset Link", key="reset_submit_btn")
                
                if reset_submitted:
                    with st.spinner("Sending link..."):
                        success, message = auth.send_password_reset(reset_email)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)


# --- 5. GATEKEEPER ---
if not st.session_state.logged_in:
    show_login_page()
    st.stop()


# ==========================================
# --- 6. MAIN ELLI INTERFACE (LOGGED IN) ---
# ==========================================

# Sidebar Controls & History
st.sidebar.markdown(f"**Logged in as:**<br>{st.session_state.user_email}", unsafe_allow_html=True)


with st.sidebar.expander("Update Your Password", expanded=False):
    with st.form("change_password_form"):
        update_pass = st.text_input("New Password", type="password", key="update_pass_input")
        update_confirm = st.text_input("Confirm Password", type="password", key="update_confirm_input")
        update_submitted = st.form_submit_button("Update Password", key="update_submit_btn")
        
        if update_submitted:
            if update_pass != update_confirm:
                st.error("Passwords do not match.")
            elif len(update_pass) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                success, message = auth.update_password(update_pass)
                if success:
                    st.success(message)
                else:
                    st.error(message)

    if st.sidebar.button("Logout", key="logout_sidebar_btn"):
        st.session_state.logged_in = False
        st.session_state.user_email = ""
        auth.get_client().auth.sign_out()
        st.rerun()



st.sidebar.divider()
# New Chat Button
if st.sidebar.button("+ New Chat", use_container_width=True):
    st.session_state.current_chat_id = str(uuid.uuid4())
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am ELLI. Nice to meet you."}]
    st.rerun()


def delete_all_chats_for_user():
    if not st.session_state.get("user_email"):
        return False, "Please log in before deleting chats."

    try:
        client = auth.get_client()
        res = client.table("chats").delete().eq("user_email", st.session_state.user_email).execute()
        if getattr(res, "error", None):
            return False, f"Could not delete chat history: {res.error}"
        return True, "All chat history deleted."
    except Exception as e:
        return False, f"Error deleting chats: {e}"


if "confirm_delete_all" not in st.session_state:
    st.session_state.confirm_delete_all = False

if st.sidebar.button("Delete ALL Chats", use_container_width=True, key="delete_all_chats_btn"):
    st.session_state.confirm_delete_all = True

if st.session_state.confirm_delete_all:
    st.sidebar.write("WARNING: This will permanently delete all your chats!")
    col1, col2 = st.sidebar.columns([1, 1])
    with col1:
        if st.button("Yes, Clear", key="confirm_yes_sidebar"):
            success, message = delete_all_chats_for_user()
            if success:
                st.success(message)
            else:
                st.error(message)

            st.session_state.confirm_delete_all = False
            st.session_state.current_chat_id = str(uuid.uuid4())
            st.session_state.messages = [{"role": "assistant", "content": "Conversation reset. How can I help?"}]
            st.rerun()
    with col2:
        if st.button("Cancel", key="confirm_no_sidebar"):
            st.session_state.confirm_delete_all = False
            st.rerun()

st.sidebar.markdown("### Chat History")

# Load history from Supabase
try:
    client = auth.get_client()
    history_res = client.table("chats").select("id, title").eq("user_email", st.session_state.user_email).order("created_at", desc=True).execute()
    if history_res.data:
        for past_chat in history_res.data:
            if st.sidebar.button(past_chat["title"], key=f"chat_{past_chat['id']}", use_container_width=True):
                chat_data = client.table("chats").select("messages").eq("id", past_chat["id"]).execute()
                if chat_data.data:
                    st.session_state.current_chat_id = past_chat["id"]
                    st.session_state.messages = chat_data.data[0]["messages"]
                    st.rerun()
    else:
        st.sidebar.caption("No past conversations yet.")
except Exception as e:
    st.sidebar.caption("Could not load history. (Ensure SQL table is created)")


# Chat Initialization
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am ELLI."}]


def show_chat() -> None:
    if "confirm_clear" not in st.session_state:
        st.session_state.confirm_clear = False

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

    # 3. AI GENERATION & DB SAVE
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
            
            try:
                title_text = st.session_state.messages[1]["content"] if len(st.session_state.messages) > 1 else "New Chat"
                chat_title = (title_text[:25] + "...") if len(title_text) > 25 else title_text
                
                client = auth.get_client()
                client.table("chats").upsert({
                    "id": st.session_state.current_chat_id,
                    "user_email": st.session_state.user_email,
                    "title": chat_title,
                    "messages": st.session_state.messages
                }).execute()
            except Exception as e:
                print(f"Database save error: {e}")
                
            st.rerun()


# --- 7. Render Header & Navigation Menu ---
st.markdown(
    '''
    <div class="elli-brand">
        <h1>ELLI</h1>
        <p>Evolving<br>Large<br>Language<br>Intelligence</p>
    </div>
    ''', 
    unsafe_allow_html=True
)

nav_col1, nav_col2, _ = st.columns([1, 7, 6])
with nav_col1:
    st.page_link("webpage.py", label="Chat")
with nav_col2:
    # Ensure this matches the exact filename in your repo!
    st.page_link("pages/InfoPage.py", label="Info Center")

show_chat()
