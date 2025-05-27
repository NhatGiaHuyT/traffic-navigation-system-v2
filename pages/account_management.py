import streamlit as st
import sqlite3
import bcrypt
import uuid
import time
import hashlib
import subprocess
import sys
from pathlib import Path


# Where we’ll keep our SQLite file
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_PATH = DATA_DIR / "users.db"

# Connect (and allow cross-thread usage if you really need it)
conn = sqlite3.connect(str(DATABASE_PATH), check_same_thread=False)


c = conn.cursor()

c.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    is_active INTEGER DEFAULT 0,
    activation_token TEXT,
    reset_token TEXT,
    preferences TEXT
)
''')
conn.commit()

# Utility functions

def hash_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def check_password(password: str, hashed: bytes) -> bool:
    return bcrypt.checkpw(password.encode(), hashed)

def generate_token() -> str:
    return str(uuid.uuid4())

def get_user_by_username(username: str):
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    return c.fetchone()

def get_user_by_email(email: str):
    c.execute("SELECT * FROM users WHERE email = ?", (email,))
    return c.fetchone()

def get_user_by_activation_token(token: str):
    c.execute("SELECT * FROM users WHERE activation_token = ?", (token,))
    return c.fetchone()

def get_user_by_reset_token(token: str):
    c.execute("SELECT * FROM users WHERE reset_token = ?", (token,))
    return c.fetchone()

def register_user(username: str, email: str, password: str):
    if get_user_by_username(username):
        return False, "Username already exists."
    if get_user_by_email(email):
        return False, "Email already registered."
    password_hash = hash_password(password)
    activation_token = generate_token()
    try:
        c.execute('''
        INSERT INTO users (username, email, password_hash, is_active, activation_token)
        VALUES (?, ?, ?, 0, ?)
        ''', (username, email, password_hash, activation_token))
        conn.commit()
        return True, activation_token
    except Exception as e:
        return False, str(e)

def activate_account(token: str):
    user = get_user_by_activation_token(token)
    if not user:
        return False, "Invalid activation token."
    if user[4] == 1:
        return False, "Account already activated."
    c.execute("UPDATE users SET is_active = 1, activation_token = NULL WHERE id = ?", (user[0],))
    conn.commit()
    return True, "Account activated successfully."

def login_user(username: str, password: str):
    user = get_user_by_username(username)
    if not user:
        return False, "User not found."
    if user[4] == 0:
        return False, "Account not activated."
    if not check_password(password, user[3]):
        return False, "Incorrect password."
    return True, user

def logout_user():
    st.session_state['logged_in'] = False
    st.session_state['user'] = None

def reset_password_request(email: str):
    user = get_user_by_email(email)
    if not user:
        return False, "Email not found."
    reset_token = generate_token()
    c.execute("UPDATE users SET reset_token = ? WHERE id = ?", (reset_token, user[0]))
    conn.commit()
    return True, reset_token

def reset_password(token: str, new_password: str):
    user = get_user_by_reset_token(token)
    if not user:
        return False, "Invalid reset token."
    password_hash = hash_password(new_password)
    c.execute("UPDATE users SET password_hash = ?, reset_token = NULL WHERE id = ?", (password_hash, user[0]))
    conn.commit()
    return True, "Password reset successfully."

def change_password(username: str, old_password: str, new_password: str):
    user = get_user_by_username(username)
    if not user:
        return False, "User not found."
    if not check_password(old_password, user[3]):
        return False, "Old password incorrect."
    password_hash = hash_password(new_password)
    c.execute("UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user[0]))
    conn.commit()
    return True, "Password changed successfully."

def get_user_preferences(username: str):
    user = get_user_by_username(username)
    if not user:
        return None
    return user[7]  # preferences column

def update_user_preferences(username: str, preferences: str):
    user = get_user_by_username(username)
    if not user:
        return False
    c.execute("UPDATE users SET preferences = ? WHERE id = ?", (preferences, user[0]))
    conn.commit()
    return True

# UI Colors (example, adjust to your project colors)
PRIMARY_COLOR = "#4f46e5"  # Indigo-600
SECONDARY_COLOR = "#6366f1"  # Indigo-500
BACKGROUND_COLOR = "#f3f4f6"  # Gray-100
TEXT_COLOR = "#111827"  # Gray-900


# Custom CSS for styling
st.markdown(f"""
    <style>
    .main {{
        background-color: {BACKGROUND_COLOR};
        color: {TEXT_COLOR};
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}
    .stButton>button {{
        background-color: {PRIMARY_COLOR};
        color: white;
        border-radius: 8px;
        height: 40px;
        width: 100%;
        font-weight: 600;
        font-size: 16px;
    }}
    .stTextInput>div>input {{
        height: 40px;
        font-size: 16px;
        border-radius: 8px;
        border: 1px solid #d1d5db;
        padding-left: 10px;
    }}
    .stTextInput>div>input:focus {{
        border-color: {PRIMARY_COLOR};
        box-shadow: 0 0 5px {PRIMARY_COLOR};
        outline: none;
    }}
    .stAlert {{
        border-radius: 8px;
    }}
    </style>
""", unsafe_allow_html=True)

# Session state initialization
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'user' not in st.session_state:
    st.session_state['user'] = None
if 'activation_token' not in st.session_state:
    st.session_state['activation_token'] = None
if 'reset_token' not in st.session_state:
    st.session_state['reset_token'] = None

import threading

def launch_homepage_subprocess():
    # Launch HomePage.py as a subprocess in a separate thread to avoid blocking
    def run_homepage():
        subprocess.Popen([sys.executable, "HomePage.py"])
    thread = threading.Thread(target=run_homepage, daemon=True)
    thread.start()

def redirect_to_homepage():
    # Removed to avoid subprocess launching and iframe embedding for seamless integration
    pass

def launch_homepage_subprocess():
    # Removed to avoid subprocess launching and iframe embedding for seamless integration
    pass

def signup():
    st.header("Sign Up")
    with st.form("signup_form"):
        username = st.text_input("Username", max_chars=20)
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        password_confirm = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Sign Up")
        if submitted:
            if not username or not email or not password or not password_confirm:
                st.error("Please fill in all fields.")
                return
            if password != password_confirm:
                st.error("Passwords do not match.")
                return
            success, result = register_user(username, email, password)
            if success:
                st.success("Registration successful! Please activate your account using the token below.")
                st.info(f"Activation Token: {result}")
                st.session_state['activation_token'] = result
            else:
                st.error(f"Registration failed: {result}")

def activate():
    st.header("Activate Account")
    token = st.text_input("Enter Activation Token")
    if st.button("Activate"):
        if not token:
            st.error("Please enter the activation token.")
            return
        success, message = activate_account(token)
        if success:
            st.success(message)
        else:
            st.error(message)

# account_management.py
import streamlit as st
import sqlite3
import bcrypt
import uuid

# … your existing utility functions here …

# Set session defaults at top‐level of this module
st.session_state.setdefault("logged_in", False)
st.session_state.setdefault("user", None)
st.session_state.setdefault("login_error", "")

def do_login():
    """Callback invoked when the user clicks “Login” in the form."""
    username = st.session_state.login_username
    password = st.session_state.login_password

    success, result = login_user(username, password)
    if success:
        st.session_state.user = result           # store the full user row
        st.session_state.logged_in = True        # flip login flag
        st.session_state.login_error = ""        # clear any prior error
    else:
        st.session_state.login_error = result    # show error below the form

def login_ui():
    """Renders the login form; one-click login via on_click."""
    st.header("Login")
    with st.form("login_form"):
        st.text_input("Username", key="login_username")
        st.text_input("Password", type="password", key="login_password")
        st.form_submit_button("Login", on_click=do_login)

    # If there was an error, display it
    if st.session_state.login_error:
        st.error(st.session_state.login_error)


def logout():
    if st.button("Logout"):
        logout_user()
        st.success("Logged out successfully.")

def password_reset_request():
    st.header("Password Reset Request")
    with st.form("reset_request_form"):
        email = st.text_input("Enter your registered email")
        submitted = st.form_submit_button("Send Reset Token")
        if submitted:
            if not email:
                st.error("Please enter your email.")
                return
            success, token = reset_password_request(email)
            if success:
                st.success("Password reset token generated. Use it to reset your password.")
                st.info(f"Reset Token: {token}")
                st.session_state['reset_token'] = token
            else:
                st.error(token)

def password_reset():
    st.header("Reset Password")
    token = st.text_input("Enter Reset Token")
    new_password = st.text_input("New Password", type="password")
    new_password_confirm = st.text_input("Confirm New Password", type="password")
    if st.button("Reset Password"):
        if not token or not new_password or not new_password_confirm:
            st.error("Please fill in all fields.")
            return
        if new_password != new_password_confirm:
            st.error("Passwords do not match.")
            return
        success, message = reset_password(token, new_password)
        if success:
            st.success(message)
        else:
            st.error(message)

def change_password_ui():
    st.header("Change Password")
    if not st.session_state['logged_in']:
        st.warning("You must be logged in to change your password.")
        return
    with st.form("change_password_form"):
        old_password = st.text_input("Old Password", type="password")
        new_password = st.text_input("New Password", type="password")
        new_password_confirm = st.text_input("Confirm New Password", type="password")
        submitted = st.form_submit_button("Change Password")
        if submitted:
            if not old_password or not new_password or not new_password_confirm:
                st.error("Please fill in all fields.")
                return
            if new_password != new_password_confirm:
                st.error("New passwords do not match.")
                return
            username = st.session_state['user'][1]
            success, message = change_password(username, old_password, new_password)
            if success:
                st.success(message)
            else:
                st.error(message)

def user_preferences_ui():
    st.header("User Preferences")
    if not st.session_state['logged_in']:
        st.warning("You must be logged in to view or edit preferences.")
        return
    username = st.session_state['user'][1]
    prefs = get_user_preferences(username)
    prefs_dict = {}
    if prefs:
        try:
            import json
            prefs_dict = json.loads(prefs)
        except:
            prefs_dict = {}
    with st.form("preferences_form"):
        theme = st.selectbox("Theme", options=["Light", "Dark"], index=0 if prefs_dict.get("theme") == "Light" else 1)
        notifications = st.checkbox("Enable Notifications", value=prefs_dict.get("notifications", True))
        submitted = st.form_submit_button("Save Preferences")
        if submitted:
            prefs_dict["theme"] = theme
            prefs_dict["notifications"] = notifications
            prefs_json = json.dumps(prefs_dict)
            if update_user_preferences(username, prefs_json):
                st.success("Preferences updated.")
            else:
                st.error("Failed to update preferences.")

def main():
    st.title("Account Management 2.0")
    menu = ["Login", "Sign Up", "Activate Account", "Password Reset Request", "Reset Password", "Change Password", "User Preferences", "Logout"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Login":
        login_ui()
    elif choice == "Sign Up":
        signup()
    elif choice == "Activate Account":
        activate()
    elif choice == "Password Reset Request":
        password_reset_request()
    elif choice == "Reset Password":
        password_reset()
    elif choice == "Change Password":
        change_password_ui()
    elif choice == "User Preferences":
        user_preferences_ui()
    elif choice == "Logout":
        logout()

if __name__ == "__main__":
    main()
