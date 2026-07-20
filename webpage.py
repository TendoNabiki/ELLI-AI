import streamlit as st
import streamlit.components.v1 as components
import random
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ELLI | The Future of AI",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- BACKGROUND ANIMATION FIX ---
components.html(
    """
    <script>
    const parentDoc = window.parent.document;
    
    // Check if we already injected it to avoid duplicates on rerun
    if (!parentDoc.getElementById('lottie-bg-container')) {
        
        // 1. Load the Lottie script dynamically
        const script = parentDoc.createElement('script');
        script.src = "https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js";
        parentDoc.head.appendChild(script);

        // 2. Create a stable wrapper div for the animation
        const container = parentDoc.createElement('div');
        container.id = 'lottie-bg-container';
        container.style.position = 'fixed';
        container.style.top = '0';
        container.style.left = '0';
        container.style.width = '100vw';
        container.style.height = '100vh';
        container.style.zIndex = '0'; /* Sits just above the base background */
        container.style.pointerEvents = 'none'; /* Allows you to click through it */
        container.style.opacity = '0.15'; /* Adjust visibility here */

        // 3. Once script loads, inject the web component
        script.onload = () => {
            container.innerHTML = `
                <lottie-player 
                    src="https://lottie.host/80f7602e-13cb-4a11-8ec8-8cf81e3c8ca4/4xJ1t2T0B8.json" 
                    background="transparent" 
                    speed="0.6" 
                    style="width: 100%; height: 100%;" 
                    loop 
                    autoplay>
                </lottie-player>
            `;
        };

        // 4. Inject into the deepest Streamlit wrapper so it doesn't get hidden
        const stApp = parentDoc.querySelector('[data-testid="stAppViewContainer"]') || parentDoc.body;
        stApp.appendChild(container);
    }
    </script>
    """,
    height=0,
    width=0
)

# --- CUSTOM CSS FOR FUTURISTIC UI ---
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
        color: #c9d1d9;
    }
    
    /* Force Streamlit's main content to sit ABOVE the Lottie animation */
    div[data-testid="stMain"] {
        position: relative;
        z-index: 10;
    }
    
    header[data-testid="stHeader"] {
        background: transparent !important;
        z-index: 10;
    }
    
    /* Hero section styling */
    .hero-container {
        text-align: center;
        padding: 3rem 1rem;
        background: linear-gradient(180deg, rgba(14,17,23,1) 0%, rgba(22,27,34,1) 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        border: 1px solid #30363d;
    }
    
    /* Gradient text for main title */
    .gradient-text {
        background: -webkit-linear-gradient(45deg, #a8c7fa, #d2e3fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Inter', sans-serif;
        font-size: 4rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    
    /* Subtitles and headers */
    .subtitle {
        font-size: 1.5rem;
        color: #8b949e;
        font-weight: 300;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    
    .section-header {
        color: #a8c7fa;
        font-family: 'Inter', sans-serif;
        border-bottom: 1px solid #30363d;
        padding-bottom: 10px;
    }

    /* Info cards */
    .info-card {
        background-color: #161b22;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #30363d;
        height: 100%;
    }
    
    /* Chat input styling to mimic Gemini */
    .stChatInputContainer {
        border-radius: 30px;
        border: 1px solid #555 !important;
        background-color: #1e1e1e !important;
    }
</style>
""", unsafe_allow_html=True)

# --- HERO SECTION ---
st.markdown("""
<div class="hero-container">
    <h1 class="gradient-text">ELLI</h1>
    <p class="subtitle">Evolving Language Learning Intelligence</p>
    <p style="max-width: 800px; margin: 0 auto; color: #b1bac4; font-size: 1.1rem; line-height: 1.6;">
        ELLI is designed to overcome the computational bottleneck of massive 70-Billion parameter models. 
        By utilizing a highly efficient, 300-million parameter architecture powered by a Mixture of Experts, 
        ELLI acts as a hyper-adaptable AI agent. It seamlessly bridges the gap between raw computation and intuitive 
        user adaptation through continuous, spontaneous learning.
    </p>
</div>
""", unsafe_allow_html=True)

# --- CORE ARCHITECTURE & AIMS (Columns) ---
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <h3 style="color: #a8c7fa;"> Spontaneous Learning</h3>
        <p>ELLI continuously fine-tunes itself. By automatically reviewing historical chats and data inputs, it adapts its weights and memory spontaneously without requiring massive, separate training loops.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h3 style="color: #a8c7fa;"> Cognition & Introspection</h3>
        <p>Operating on a dual-stage Transformer architecture, a separate, constantly-running 'Thinking Layer' processes context and pushes optimized instructions directly to the output generation layer.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <h3 style="color: #a8c7fa;"> Lightweight & Agile</h3>
        <p>Built as a lean 300-million parameter model using bf16 format. This allows ELLI to run its internal cognition loops 24/7, maximizing compute efficiency and ensuring lightning-fast responses.</p>
    </div>
    """, unsafe_allow_html=True)

st.write("") # Spacer

# --- THE FOUNDERS ---
with st.expander("Group Eightfold | The Founders"):
    st.markdown("""
    **Data Scientist:** Eddie Franco  
    **Architecture Engineers:** Roy Zhou, Brian Suh  
    **AI Data Engineers:** Kamesh Surapuraju, Vikranth Maddali  
    """)

st.divider()

# --- GEMINI-STYLE CHAT INTERFACE ---
st.markdown("<h3 style='text-align: center; color: #a8c7fa; margin-bottom: 2rem;'>Initialize ELLI Interface</h3>", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "System initialized. I am ELLI. Awaiting input..."}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask ELLI"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Processing via Mixture of Experts..."):
            time.sleep(1.0)
            
        message_placeholder = st.empty()
        full_response = ""
        assistant_response = random.choice(
            [
                "Acknowledged. Routing query through the Cognition & Introspection layer.",
                "System standing by. How can I assist?",
                f"Directive received: '{prompt}'. Executing parameters...",
                "I am ELLI. Fine-tuning spontaneous memory based on your input.",
            ]
        )
        
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})