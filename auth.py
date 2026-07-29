import streamlit as st
from supabase import create_client, Client

def get_client() -> Client:
    """
    Creates or retrieves a Supabase client that is completely 
    isolated to the current user's browser session.
    """
    if "supabase_client" not in st.session_state:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        st.session_state.supabase_client = create_client(url, key)
    
    return st.session_state.supabase_client

def verify_user(email, password):
    """Attempt to log in an existing user."""
    client = get_client()
    try:
        response = client.auth.sign_in_with_password({"email": email, "password": password})
        if response.user:
            return True, "Success"
        return False, "Unknown error during login."
    except Exception as e:
        return False, str(e)

def create_user(email, password):
    """Attempt to sign up a new user."""
    client = get_client()
    try:
        response = client.auth.sign_up({"email": email, "password": password})
        return True, "Account created successfully! You can now log in."
    except Exception as e:
        return False, str(e)

def send_password_reset(email):
    """Send a password reset link to the user's email."""
    client = get_client()
    try:
        client.auth.reset_password_email(email)
        return True, "Password reset link sent to your email!"
    except Exception as e:
        return False, str(e)

def update_password(new_password):
    """Update the password for the currently logged-in user."""
    client = get_client()
    try:
        client.auth.update_user({"password": new_password})
        return True, "Password updated successfully!"
    except Exception as e:
        return False, str(e)
