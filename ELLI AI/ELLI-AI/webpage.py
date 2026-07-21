import base64
import random
import time
from html import escape
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import auth

# --- PAGE CONFIGURATION ---
ROOT = Path(__file__).parent
st.set_page_config(
    page_title="ELLI | Evolving Large Language Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- 1. GLOBAL UI SETTINGS (Must load before Gatekeeper) ---
# Background Animation
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

# Main CSS Styling
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
        :root { --ink:#181b1a; --panel:#202523; --mint:#1ee5aa; --gold:#ffcb05; --soft:#b9c0bc; }
        .stApp { background:radial-gradient(circle at 25% 12%, #2a3530 0, #181b1a 32rem); color:#f5f7f5; }
        [data-testid="stHeader"] { background:transparent; } #MainMenu, footer { visibility:hidden; }
        .block-container { max-width:1400px; padding:2rem 3.5rem 2rem; position: relative; z-index: 10; }
        
        /* BRANDING LOGO */
        .elli-brand-link { text-decoration: none !important; display: inline-block; cursor: pointer; transition: transform 0.2s ease; }
        .elli-brand-link:hover { transform: scale(1.02); }
        .elli-brand { display:flex; align-items:flex-end; gap:0.8rem; margin:.2rem 0 2rem 0; }
        .elli-brand h1 { font:700 clamp(2.5rem,6vw,4rem)/.72 "Space Grotesk",sans-serif; letter-spacing:0; margin:0; color:#f2f4f2; }
        .elli-brand p { font:600 0.8rem/1.22 "Space Grotesk",sans-serif; color:#c5cbc7; margin:0 0 0.3rem 0; max-width:11rem; }
        
        .chat-shell { background:rgba(32,37,35,.88); border:2px solid var(--mint); border-radius:3.2rem; padding:1.5rem 1.6rem 1.2rem; min-height:32rem; box-shadow:0 0 32px rgba(30,229,170,.06); }
        .chat-title { display:flex; justify-content:space-between; align-items:center; color:#e9efea; font:500 .77rem "DM Mono",monospace; letter-spacing:.1em; text-transform:uppercase; margin:0 .5rem 1.2rem; }
        .online-dot { display:inline-block; width:.55rem; height:.55rem; background:var(--mint); border-radius:50%; margin-right:.45rem; box-shadow:0 0 12px var(--mint); }
        .message { width:fit-content; max-width:76%; padding:1rem 1.2rem; margin:.85rem .45rem; border-radius:1.35rem; font:500 1rem/1.45 "Space Grotesk",sans-serif; }
        .assistant-message { background:#29302d; border:1px solid var(--mint); border-bottom-left-radius:.35rem; color:#f4f7f4; }
        .user-message { background:transparent; border:1px solid #86aaa0; border-bottom-right-radius:.35rem; color:var(--gold); margin-left:auto; }
        .message-label { display:block; font:500 .65rem "DM Mono",monospace; letter-spacing:.1em; opacity:.72; text-transform:uppercase; margin-bottom:.38rem; }
        
        [data-testid="stChatInput"] { border:2px solid var(--mint)!important; background:#202523!important; border-radius:1.5rem!important; padding:.38rem .55rem!important; margin-top:1.3rem; }
        [data-testid="stChatInput"] textarea { color:var(--gold)!important; font:500 1.1rem "Space Grotesk",sans-serif!important; }
        [data-testid="stChatInput"] textarea::placeholder { color:#a8b0ab!important; }
        [data-testid="stChatInput"] button { background:var(--mint); border-radius:50%; }
        [data-testid="stChatInput"] button svg { fill:#13221b; }
        .clear-button button { border-color:#61716a!important; color:#b9c0bc!important; border-radius:1rem!important; font:.75rem "DM Mono",monospace!important; }
        
        @media (max-width:800px) { .block-container{padding:2rem 1rem;} .elli-brand h1{font-size:3rem;} .elli-brand p{font-size:.7rem;} .chat-shell{min-height:24rem;border-radius:2rem;} .message{max-width:92%;} }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- 2. AUTHENTICATION STATE ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""


# --- 3. LOGIN PAGE FUNCTION ---
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
                                    st.error("Rate limit exceeded. Please wait a moment or disable email confirmation in Supabase.")
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


# --- 4. GATEKEEPER ---
if not st.session_state.logged_in:
    show_login_page()
    st.stop()


# ==========================================
# --- 5. MAIN ELLI INTERFACE (LOGGED IN) ---
# ==========================================

# Sidebar Controls
st.sidebar.markdown(f"**Logged in as:**<br>{st.session_state.user_email}", unsafe_allow_html=True)
st.sidebar.divider()

# Change Password Expander
with st.sidebar.expander("Change Password"):
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

st.sidebar.divider()

# Sidebar Logout Button
if st.sidebar.button("Logout", key="logout_sidebar_btn"):
    st.session_state.logged_in = False
    st.session_state.user_email = ""
    auth.supabase.auth.sign_out()
    st.rerun()

# Chat Initialization & Functions
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am ELLI. What would you like to explore today?"}]

def show_chat() -> None:
    if "confirm_clear" not in st.session_state:
        st.session_state.confirm_clear = False

    conversation = '<div class="chat-shell"><div class="chat-title"><span><span class="online-dot"></span>ELLI conversation</span><span>v3.5.2</span></div>'
    for message in st.session_state.messages:
        style = "assistant-message" if message["role"] == "assistant" else "user-message"
        label = "ELLI reply" if message["role"] == "assistant" else "Your message"
        conversation += f'<div class="message {style}"><span class="message-label">{label}</span>{escape(message["content"])}</div>'
    st.markdown(conversation + "</div>", unsafe_allow_html=True)

    st.markdown('<div class="clear-button">', unsafe_allow_html=True)
    
    if not st.session_state.confirm_clear:
        if st.button("Clear conversation", key="clear_chat_init_btn"):
            st.session_state.confirm_clear = True
            st.rerun()
    else:
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

    if prompt := st.chat_input("Ask ELLI anything…"):
        st.session_state.confirm_clear = False
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("ELLI is thinking…"):
            time.sleep(0.7)
            prompt_lower = prompt.lower()
            if "hello" in prompt_lower or "hi" in prompt_lower:
                ai_reply = "Hello! I am ELLI, and I am currently running in a front-end demo mode. I am ready to help you explore the project concept."
            elif "team" in prompt_lower or "founder" in prompt_lower or "creator" in prompt_lower:
                ai_reply = "The project is led by Team Eightfold, with Eddie Franco guiding the data science side and the rest of the team contributing to architecture and AI data engineering."
            elif "learn" in prompt_lower or "about" in prompt_lower:
                ai_reply = "ELLI is built around efficient adaptation, spontaneous learning, and a lightweight architecture designed to make advanced AI interaction feel more responsive and flexible."
            else:
                ai_reply = random.choice(
                    [
                        "Acknowledged. ELLI is ready to guide you through the project concept and its goals.",
                        "That is a thoughtful question. The current demo keeps the experience focused on the interface and the project story.",
                        f"Your message, '{prompt}', has been received. The full model connection is intentionally left for later, so this section focuses on the experience itself.",
                    ]
                )

        st.session_state.messages.append({"role": "assistant", "content": ai_reply})
        st.rerun()

# Render Header & Chat
st.markdown(
    '''
    <a href="landingpage" target="_self" class="elli-brand-link">
        <div class="elli-brand">
            <h1>ELLI</h1>
            <p>Evolving<br>Large<br>Language<br>Intelligence</p>
        </div>
    </a>
    ''', 
    unsafe_allow_html=True
)

show_chat()