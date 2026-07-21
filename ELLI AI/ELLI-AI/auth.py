import streamlit as st
from supabase import create_client, Client

# Initialize the Supabase client using Streamlit secrets
@st.cache_resource
def init_supabase() -> Client:
    url: str = st.secrets["SUPABASE_URL"]
    key: str = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()

def create_user(email, password):
    try:
        # Supabase handles the password hashing automatically
        response = supabase.auth.sign_up({
            "email": email, 
            "password": password
        })
        return True, "Account created successfully! You can now log in."
    except Exception as e:
        # Will catch errors like "User already exists" or "Password too weak"
        return False, str(e)

def verify_user(email, password):
    try:
        # Attempts to log the user in and grab a session token
        response = supabase.auth.sign_in_with_password({
            "email": email, 
            "password": password
        })
        return True, response.user
    except Exception as e:
        return False, "Invalid email or password."

def send_password_reset(email):
    try:
        # Instructs Supabase to send a password reset magic link to the user's email
        supabase.auth.reset_password_email(email)
        return True, "Password reset link sent! Please check your inbox."
    except Exception as e:
        return False, str(e)

def update_password(new_password):
    try:
        # Updates the password for the currently authenticated user
        supabase.auth.update_user({"password": new_password})
        return True, "Password updated successfully!"
    except Exception as e:
        return False, str(e)