import streamlit as st
import streamlit.components.v1 as components
import os
import subprocess
import socket
import time

def is_backend_running(host='127.0.0.1', port=5000):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.settimeout(1)
        s.connect((host, port))
        s.close()
        return True
    except Exception:
        return False

def start_backend():
    # Start the backend Flask app in a subprocess
    # Use python executable and run backend/app.py
    # Set cwd to backend directory to ensure correct working directory
    # Keep stdout and stderr to console for debugging
    subprocess.Popen(
        ['python', 'app.py'],
        cwd='pages',
        shell=False
    )

# Check if backend is running, if not start it
if not is_backend_running():
    start_backend()
    # Wait a bit for backend to start
    time.sleep(3)

# Page configuration
st.set_page_config(
    page_title="RouteVision AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

from account_management import (
    login_ui,
    signup,
    activate,
    password_reset_request,
    password_reset,
    change_password_ui,
    user_preferences_ui,
    logout
)

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Note: session_state defaults (logged_in, user, login_error) are initialized in account_management.py
st.session_state.setdefault("logged_in", False)
st.session_state.setdefault("user", None)
# --- Unauthenticated account menu ---
if not st.session_state.logged_in:
    choice = st.sidebar.selectbox(
        "Account",
        [
            "Login",
            "Sign Up",
            "Activate Account",
            "Password Reset Request",
            "Reset Password"
        ]
    )
    if choice == "Login":

        if not st.session_state.logged_in:
            did_login = login_ui()             # just a bool
            if did_login:
                # pull the user out of wherever login_ui() put it
                st.session_state.user = st.session_state.get("login_username")
                st.session_state.logged_in = True
            else:
                st.stop()
        st.write(f"Welcome back, {st.session_state.user}!")
    elif choice == "Sign Up":
        signup()
    elif choice == "Activate Account":
        activate()
    elif choice == "Password Reset Request":
        password_reset_request()
    elif choice == "Reset Password":
        password_reset()
    st.stop()

# --- Authenticated app menu ---
choice = st.sidebar.selectbox(
    "App",
    ["Home", "Change Password"]
)

if choice == "Home":
    st.success(f"Welcome, {st.session_state.user[1]}!")
    chatbot_html = """
<style>
  #chat-toggle-btn {
    position: fixed;
    bottom: 20px;
    right: 30px;
    width: 60px;
    height: 60px;
    background: linear-gradient(90deg, #1e3c72, #2a5298);
    border-radius: 50%;
    box-shadow: 0 8px 30px rgba(0,0,0,0.15);
    border: none;
    color: white;
    font-size: 30px;
    cursor: pointer;
    z-index: 10000;
    display: flex;
    align-items: center;
    justify-content: center;
    user-select: none;
    transition: background 0.3s ease;
  }
  #chat-toggle-btn:hover {
    background: linear-gradient(1000deg, #1e3c68, #2a5296);
  }
  #chat-container {
    position: fixed;
    bottom: 90px;
    right: -0.2px;
    width: 360px;
    max-height: 520px;
    background: white;
    border-radius: 20px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.15);
    display: none;
    flex-direction: column;
    overflow: hidden;
    font-family: 'Poppins', sans-serif;
    z-index: 9999;
  }
  #chat-header {
    background: linear-gradient(90deg, #1e3c72, #2a5298);
    color: white;
    padding: 14px 20px;
    font-weight: 700;
    font-size: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    cursor: default;
    user-select: none;
    border-top-left-radius: 20px;
    border-top-right-radius: 20px;
  }
  #chat-header span.close-btn {
    font-weight: 900;
    font-size: 24px;
    cursor: pointer;
    user-select: none;
  }
  #chat-messages {
    flex: 1;
    padding: 200px 20px;
    overflow-y: auto;
    background: #f7f7f7;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .message {
    max-width: 75%;
    padding: 12px 18px;
    border-radius: 20px;
    font-size: 14px;
    line-height: 1.4;
    word-wrap: break-word;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    position: relative;
  }
  .message.user {
    background: #1e3c72;
    color: white;
    align-self: flex-end;
    border-bottom-right-radius: 6px;
  }
  .message.bot {
    background: #e9ecef;
    color: #333;
    align-self: flex-start;
    border-bottom-left-radius: 6px;
  }
  .timestamp {
    font-size: 10px;
    color: #999;
    margin-top: 4px;
    user-select: none;
  }
  #chat-input-container {
    display: flex;
    padding: 12px 20px;
    border-top: 1px solid #ddd;
    background: white;
    align-items: center;
    border-radius: 20px;
  }
  #chat-input {
    flex: 1;
    border: 1px solid #ddd;
    border-radius: 20px;
    padding: 10px 18px;
    font-size: 14px;
    outline: none;
    font-family: 'Poppins', sans-serif;
  }
  #chat-send-btn {
    background: linear-gradient(90deg, #1e3c72, #2a5298);
    border: none;
    color: white;
    padding: 10px 18px;
    margin-left: 12px;
    border-radius: 50%;
    cursor: pointer;
    font-weight: 700;
    transition: background 0.3s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
  }
  #chat-send-btn:hover {
    background: #1e3c72;
  }
  #chat-send-btn svg {
    fill: white;
  }
</style>
<button id="chat-toggle-btn" aria-label="Toggle chat">💬</button>
<div id="chat-container" role="region" aria-live="polite" aria-label="Chat window">
  <div id="chat-header">
    Virtual Assistant
    <span class="close-btn" role="button" aria-label="Close chat">&times;</span>
  </div>
  <div id="chat-messages" tabindex="0"></div>
  <div id="chat-input-container">
    <input type="text" id="chat-input" placeholder="Write your message..." aria-label="Chat input" />
    <button id="chat-send-btn" aria-label="Send message" title="Send message">
      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-send" viewBox="0 0 24 24"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
    </button>
  </div>
</div>
<script>
  const toggleBtn = document.getElementById('chat-toggle-btn');
  const chatContainer = document.getElementById('chat-container');
  const closeBtn = chatContainer.querySelector('.close-btn');
  const chatMessages = document.getElementById('chat-messages');
  const chatInput = document.getElementById('chat-input');
  const sendBtn = document.getElementById('chat-send-btn');

  let chatOpen = false;

  function toggleChat() {
    chatOpen = !chatOpen;
    const iframe = document.querySelector('iframe[srcdoc*="chat-toggle-btn"]');
    if (chatOpen) {
      chatContainer.style.display = 'flex';
      chatContainer.style.pointerEvents = 'auto';
      chatContainer.style.visibility = 'visible';
      if (iframe) {
        iframe.style.position = 'fixed';
        iframe.style.bottom = '20px';
        iframe.style.right = '30px';
        iframe.style.width = '360px';
        iframe.style.height = '700px';
        iframe.style.pointerEvents = 'auto';
        iframe.style.visibility = 'visible';
        iframe.style.display = 'block';
        iframe.style.left = '';
        iframe.style.top = '';
        iframe.style.zIndex = '-1';
      }
    } else {
      chatContainer.style.display = 'none';
      chatContainer.style.pointerEvents = 'none';
      chatContainer.style.visibility = 'hidden';
      if (iframe) {
        iframe.style.position = 'fixed';
        iframe.style.left = '-9999px';
        iframe.style.top = '-9999px';
        iframe.style.width = '0';
        iframe.style.height = '0';
        iframe.style.pointerEvents = 'none';
        iframe.style.visibility = 'hidden';
        iframe.style.display = 'none';
        iframe.style.zIndex = '-1';
      }
    }
    toggleBtn.textContent = chatOpen ? '×' : '💬';
    if (chatOpen) {
      chatInput.focus();
      scrollToBottom();
    }
  }

  toggleBtn.addEventListener('click', toggleChat);
  closeBtn.addEventListener('click', toggleChat);

  function addMessage(role, content) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', role);
    msgDiv.textContent = content;

    const timestampDiv = document.createElement('div');
    timestampDiv.classList.add('timestamp');
    const now = new Date();
    timestampDiv.textContent = now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    msgDiv.appendChild(timestampDiv);

    chatMessages.appendChild(msgDiv);
    scrollToBottom();
  }

  function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) return;
    addMessage('user', message);
    chatInput.value = '';
    try {
      const response = await fetch('http://localhost:5000/chatbot', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message})
      });
      if (response.ok) {
        const data = await response.json();
        addMessage('bot', data.reply || '🤖: (no reply)');
      } else {
        addMessage('bot', `Error ${response.status}: ${response.statusText}`);
      }
    } catch (error) {
      addMessage('bot', `Error: ${error.message}`);
    }
  }

  sendBtn.addEventListener('click', sendMessage);
  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      sendMessage();
    }
  });
</script>
"""
    components.html(chatbot_html, height=600, scrolling=True)

    st.markdown(
        """
        <style>
        /* Override Streamlit iframe container to fix chatbot position */
        iframe[srcdoc*="chat-toggle-btn"] {
            position: fixed !important;
            bottom: 20px !important;
            right: 30px !important;
            width: 360px !important;
            height: 700px !important;
            max-height: 700px !important;
            border-radius: 20px !important;
            box-shadow: none !important;
            background: transparent !important;
            z-index: 10000 !important;
            pointer-events: auto !important;
            visibility: visible !important;
            display: block !important;
        }
        /* When chat is closed, hide iframe container completely */
        iframe.chat-hidden {
            pointer-events: none !important;
            visibility: hidden !important;
            display: none !important;
            width: 0 !important;
            height: 0 !important;
        }
        /* Override parent containers to prevent clipping */
        .element-container, .block-container, .main {
            overflow: visible !important;
            height: auto !important;
            max-height: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

elif choice == "Change Password":
    change_password_ui()
elif choice == "Preferences":
    user_preferences_ui()
elif choice == "Chatbot":
    chatbot_html = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Chatbot Bubble UI</title>
<style>
  body {
    font-family: 'Poppins', sans-serif;
    margin: 0;
    padding: 0;
    background: #f0f2f5;
  }
  #chat-toggle-btn {
    position: fixed;
    bottom: 20px;
    right: 30px;
    width: 60px;
    height: 60px;
    background: #ff2e63;
    border-radius: 50%;
    box-shadow: 0 8px 30px rgba(0,0,0,0.15);
    border: none;
    color: white;
    font-size: 30px;
    cursor: pointer;
    z-index: 10000;
    display: flex;
    align-items: center;
    justify-content: center;
    user-select: none;
    transition: background 0.3s ease;
  }
  #chat-toggle-btn:hover {
    background: #ff4d7a;
  }
  #chat-container {
    position: fixed;
    bottom: 90px;
    right: 30px;
    width: 360px;
    max-height: 520px;
    background: white;
    border-radius: 20px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.15);
    display: none;
    flex-direction: column;
    overflow: hidden;
    font-family: 'Poppins', sans-serif;
    z-index: 9999;
  }
  #chat-header {
    background: #ff2e63;
    color: white;
    padding: 14px 20px;
    font-weight: 700;
    font-size: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    cursor: default;
    user-select: none;
    border-top-left-radius: 20px;
    border-top-right-radius: 20px;
  }
  #chat-header span.close-btn {
    font-weight: 900;
    font-size: 24px;
    cursor: pointer;
    user-select: none;
  }
  #chat-messages {
    flex: 1;
    padding: 15px 20px;
    overflow-y: auto;
    background: #f7f7f7;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .message {
    max-width: 75%;
    padding: 12px 18px;
    border-radius: 20px;
    font-size: 14px;
    line-height: 1.4;
    word-wrap: break-word;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    position: relative;
  }
  .message.user {
    background: #ff2e63;
    color: white;
    align-self: flex-end;
    border-bottom-right-radius: 6px;
  }
  .message.bot {
    background: #e9ecef;
    color: #333;
    align-self: flex-start;
    border-bottom-left-radius: 6px;
  }
  .timestamp {
    font-size: 10px;
    color: #999;
    margin-top: 4px;
    user-select: none;
  }
  #chat-input-container {
    display: flex;
    padding: 12px 20px;
    border-top: 1px solid #ddd;
    background: white;
    align-items: center;
  }
  #chat-input {
    flex: 1;
    border: 1px solid #ddd;
    border-radius: 20px;
    padding: 10px 18px;
    font-size: 14px;
    outline: none;
    font-family: 'Poppins', sans-serif;
  }
  #chat-send-btn {
    background: #ff2e63;
    border: none;
    color: white;
    padding: 10px 18px;
    margin-left: 12px;
    border-radius: 50%;
    cursor: pointer;
    font-weight: 700;
    transition: background 0.3s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
  }
  #chat-send-btn:hover {
    background: #ff4d7a;
  }
  #chat-send-btn svg {
    fill: white;
  }
</style>
</head>
<body>
<button id="chat-toggle-btn" aria-label="Toggle chat">💬</button>
<div id="chat-container" role="region" aria-live="polite" aria-label="Chat window">
  <div id="chat-header">
    Virtual Assistant
    <span class="close-btn" role="button" aria-label="Close chat">&times;</span>
  </div>
  <div id="chat-messages" tabindex="0"></div>
  <div id="chat-input-container">
    <input type="text" id="chat-input" placeholder="Write your message..." aria-label="Chat input" />
    <button id="chat-send-btn" aria-label="Send message" title="Send message">
      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-send" viewBox="0 0 24 24"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
    </button>
  </div>
</div>
<script>
  const toggleBtn = document.getElementById('chat-toggle-btn');
  const chatContainer = document.getElementById('chat-container');
  const closeBtn = chatContainer.querySelector('.close-btn');
  const chatMessages = document.getElementById('chat-messages');
  const chatInput = document.getElementById('chat-input');
  const sendBtn = document.getElementById('chat-send-btn');

  let chatOpen = false;

  function toggleChat() {
    chatOpen = !chatOpen;
    const iframe = document.querySelector('iframe[srcdoc*="chat-toggle-btn"]');
    if (chatOpen) {
      chatContainer.style.display = 'flex';
      chatContainer.style.pointerEvents = 'auto';
      chatContainer.style.visibility = 'visible';
      if (iframe) {
        iframe.style.position = 'fixed';
        iframe.style.bottom = '20px';
        iframe.style.right = '30px';
        iframe.style.width = '360px';
        iframe.style.height = '700px';
        iframe.style.pointerEvents = 'auto';
        iframe.style.visibility = 'visible';
        iframe.style.display = 'block';
        iframe.style.left = '';
        iframe.style.top = '';
        iframe.style.zIndex = '10000';
      }
    } else {
      chatContainer.style.display = 'none';
      chatContainer.style.pointerEvents = 'none';
      chatContainer.style.visibility = 'hidden';
      if (iframe) {
        iframe.style.position = 'fixed';
        iframe.style.left = '-9999px';
        iframe.style.top = '-9999px';
        iframe.style.width = '0';
        iframe.style.height = '0';
        iframe.style.pointerEvents = 'none';
        iframe.style.visibility = 'hidden';
        iframe.style.display = 'none';
        iframe.style.zIndex = '-1';
      }
    }
    toggleBtn.textContent = chatOpen ? '×' : '💬';
    if (chatOpen) {
      chatInput.focus();
      scrollToBottom();
    }
  }

  toggleBtn.addEventListener('click', toggleChat);
  closeBtn.addEventListener('click', toggleChat);

  function addMessage(role, content) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', role);
    msgDiv.textContent = content;

    const timestampDiv = document.createElement('div');
    timestampDiv.classList.add('timestamp');
    const now = new Date();
    timestampDiv.textContent = now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    msgDiv.appendChild(timestampDiv);

    chatMessages.appendChild(msgDiv);
    scrollToBottom();
  }

  function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) return;
    addMessage('user', message);
    chatInput.value = '';
    try {
      const response = await fetch('http://localhost:5000/chatbot', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message})
      });
      if (response.ok) {
        const data = await response.json();
        addMessage('bot', data.reply || '🤖: (no reply)');
      } else {
        addMessage('bot', `Error ${response.status}: ${response.statusText}`);
      }
    } catch (error) {
      addMessage('bot', `Error: ${error.message}`);
    }
  }

  sendBtn.addEventListener('click', sendMessage);
  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      sendMessage();
    }
  });
</script>
</body>
</html>
"""
    components.html(chatbot_html, height=600, scrolling=True)

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 1) Configure Chrome to run headlessly
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920,1080")

driver = webdriver.Chrome(options=options)

try:
    # 2) Navigate to your local Streamlit app
    driver.get("http://localhost:8501")

    # 3) Wait until the header (with class "stAppHeader") appears
    wait = WebDriverWait(driver, 10)
    header = wait.until(
        EC.presence_of_element_located((By.CSS_SELECTOR, ".stAppHeader"))
    )

    # 4) Read its computed background-color
    #
    #    You can also try "background" if you want the full shorthand (in case there is an image or gradient),
    #    but most Streamlit themes set a solid color, so "background-color" will give you a single rgba/hex.
    bg_color = header.value_of_css_property("background-color")


finally:
    driver.quit()

st.markdown(
    """
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    /* Global Styles */
    body {
        font-family: 'Poppins', sans-serif;
        color: #2c3e50;
        background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%);
        margin: 0;
        padding: 0;
    }

    .stApp {
        background: bg_color;
        min-height: 100vh;
        padding: 20px 40px;
    }
    /* Header Styles */
    .main-header {
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        padding: 40px 30px;
        border-radius: 20px;
        margin-bottom: 40px;
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.15);
        position: relative;
        overflow: hidden;
        color: #f0f4f8;
        text-align: center;
    }

    .main-header::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120" viewBox="0 0 100 100"><path d="M95,50 L85,35 L85,20 L75,20 L75,10 L25,10 L25,20 L15,20 L15,35 L5,50 L15,65 L15,80 L25,80 L25,90 L75,90 L75,80 L85,80 L85,65 Z" fill="none" stroke="rgba(255,255,255,0.12)" stroke-width="3"/></svg>');
        background-repeat: repeat;
        opacity: 0.12;
        z-index: 0;
    }

    .header-title {
        font-size: 48px;
        font-weight: 700;
        margin-bottom: 10px;
        position: relative;
        z-index: 1;
        text-shadow: 3px 3px 8px rgba(0, 0, 0, 0.3);
    }

    .header-subtitle {
        font-size: 20px;
        font-weight: 400;
        color: #d1d9e6;
        position: relative;
        z-index: 1;
        text-shadow: 2px 2px 6px rgba(0, 0, 0, 0.2);
    }

    /* Card Styles */
    .feature-card {
        background: bg_color;
        border-radius: 20px;
        padding: 30px 25px;
        height: 100%;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 10px 25px rgba(46, 82, 152, 0.15);
        position: relative;
        overflow: hidden;
        border-top: 6px solid #2a5298;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .feature-card:hover {
        transform: translateY(-12px);
        box-shadow: 0 20px 40px rgba(46, 82, 152, 0.25);
    }

    .feature-icon {
        font-size: 56px;
        text-align: center;
        margin-bottom: 20px;
        background: linear-gradient(45deg, #1e3c72, #4286f4);
        -webkit-background-clip: text;
        line-height: 1;
        user-select: none;
    }

    .feature-title {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 15px;
        color: #1e3c72;
        text-align: center;
        user-select: none;
    }

    .feature-description {
        font-size: 16px;
        color: #555555;
        margin-bottom: 25px;
        text-align: center;
        line-height: 1.6;
        user-select: none;
    }

    /* Button Styles */
    .stButton > button {
        background: linear-gradient(90deg, #1e3c72, #4286f4);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 16px 32px;
        font-size: 18px;
        font-weight: 700;
        width: 100%;
        cursor: pointer;
        transition: all 0.35s ease;
        text-align: center;
        margin-top: auto;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        box-shadow: 0 6px 18px rgba(46, 82, 152, 0.3);
        user-select: none;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #4286f4, #1e3c72);
        box-shadow: 0 8px 22px rgba(46, 82, 152, 0.45);
        transform: translateY(-3px);
    }

    /* Traffic theme elements */
    .traffic-line {
        height: 4px;
        background: repeating-linear-gradient(90deg, #ffcc00, #ffcc00 20px, transparent 20px, transparent 40px);
        margin: 40px 0;
        animation: moveLine 8s linear infinite;
        border-radius: 2px;
    }

    @keyframes moveLine {
        0% { background-position: 0 0; }
        100% { background-position: 120px 0; }
    }

    /* Footer Styles */
    .footer {
        background: bg_color;
        padding: 20px;
        border-radius: 15px;
        margin-top: 60px;
        text-align: center;
        box-shadow: 0 -8px 20px rgba(0, 0, 0, 0.07);
        font-size: 14px;
        color: #666666;
        user-select: none;
    }

    /* Home Button */
    .home-button {
        position: fixed;
        top: 70px;
        left: 30px;
        z-index: 999;
    }

    .home-button button {
        background: linear-gradient(90deg, #1e3c72, #4286f4);
        color: white;
        border: none;
        border-radius: 50%;
        width: 56px;
        height: 56px;
        font-size: 26px;
        cursor: pointer;
        box-shadow: 0 6px 14px rgba(46, 82, 152, 0.3);
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.3s ease;
        user-select: none;
    }

    .home-button button:hover {
        transform: scale(1.15);
        box-shadow: 0 10px 24px rgba(46, 82, 152, 0.45);
    }

    /* Road Animation */
    .road-container {
        height: 36px;
        overflow: hidden;
        position: relative;
        margin: 15px 0 40px 0;
    }

    .road {
        height: 100%;
        background-color: #2a5298;
        position: relative;
        border-radius: 8px;
        box-shadow: inset 0 0 10px rgba(0,0,0,0.3);
    }

    .road-line {
        position: absolute;
        top: 50%;
        transform: translateY(-50%);
        height: 6px;
        background: #ffcc00;
        width: 40px;
        border-radius: 3px;
        animation: roadLine 5s linear infinite;
        box-shadow: 0 0 8px #ffcc00;
    }

    @keyframes roadLine {
        0% { transform: translateX(-100px) translateY(-50%); }
        100% { transform: translateX(calc(100vw + 100px)) translateY(-50%); }
    }

    .road-line:nth-child(1) { animation-delay: 0s; }
    .road-line:nth-child(2) { animation-delay: 1s; }
    .road-line:nth-child(3) { animation-delay: 2s; }
    .road-line:nth-child(4) { animation-delay: 3s; }
    .road-line:nth-child(5) { animation-delay: 4s; }

    /* Card container to ensure equal height */
    .feature-container {
        display: flex;
        flex-direction: column;
        height: 100%;
    }

    /* Fix for button display inside cards */
    .button-wrapper {
        margin-top: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Show logout button
with st.sidebar:
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user = None
        components.html(
            """
            <script>
            window.location.reload(true);
            </script>
            """,
            height=0,
            width=0,
        )

# Set session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'home'

if 'lang' not in st.session_state:
    st.session_state.lang = 'vi'

lang_mapping = {
    "vi": {
        "home_title": "Chào mừng đến với RouteVision AI!",
        "home_subtitle": "Nền tảng Giám sát & Tối ưu hóa Giao thông Tiên tiến",
        "home_traffic_feed": "Thông Tin Giao Thông",
        "home_route_optimizer": "Lộ Trình Tối Ưu",
        "home_signal_simulation": "Mô Phỏng Tín Hiệu",
        "home_traffic_feed_desc": "Xem camera giao thông thời gian thực với phân tích AI. Giám sát ùn tắc, sự cố và lưu lượng giao thông trên toàn mạng lưới của bạn.",
        "home_route_optimizer_desc": "Tìm lộ trình nhanh nhất với phân tích dự đoán AI. Tránh ùn tắc và giảm thời gian di chuyển với thông tin giao thông thời gian thực.",
        "home_signal_simulation_desc": "Kiểm soát và mô phỏng tín hiệu giao thông thông minh. Kiểm tra chiến lược thời gian tín hiệu và tối ưu hóa lưu lượng giao thông với hệ thống AI của chúng tôi.",
        "home_feature_desc": "Tính năng nổi bật",
        "home_feature_traffic_feed": "KHÁM PHÁ THÔNG TIN GIAO THÔNG",
        "home_feature_route_optimizer": "TỐI ƯU LỘ TRÌNH",
        "home_feature_signal_simulation": "THỰC HIỆN MÔ PHỎNG",
        "home_dashboard_preview": "Làm cho các thành phố thông minh hơn",
        "home_dashboard_desc": "RouteVision AI giúp các nhà quản lý giao thông, nhà quy hoạch đô thị và người đi lại đưa ra quyết định tốt hơn thông qua phân tích dữ liệu thời gian thực và mô hình dự đoán. Nền tảng của chúng tôi giảm ùn tắc, khí thải và thời gian di chuyển.",
        "home_footer": "© 2025 RouteVision AI - Chuyển Đổi Giao Thông Đô Thị Thông Qua Trí Tuệ Nhân Tạo",
        "home_button": "🏠 Trang Chủ",
    },
    "en": {
        "home_title": "Welcome to RouteVision AI!",
        "home_subtitle": "Advanced Traffic Monitoring & Optimization Platform",
        "home_traffic_feed": "Traffic Feed",
        "home_route_optimizer": "Route Optimizer",
        "home_signal_simulation": "Signal Simulation",
        "home_traffic_feed_desc": "View real-time traffic cameras with AI-powered analytics. Monitor congestion, incidents, and traffic flow across your network.",
        "home_route_optimizer_desc": "Find the fastest routes with AI predictive analysis. Avoid congestion and reduce travel time with real-time traffic insights.",
        "home_signal_simulation_desc": "Smart traffic signal control and simulation. Test signal timing strategies and optimize traffic flow with our AI-powered system.",
        "home_feature_desc": "Featured",
        "home_feature_traffic_feed": "EXPLORE TRAFFIC FEED",
        "home_feature_route_optimizer": "OPTIMIZE ROUTES",
        "home_feature_signal_simulation": "RUN SIMULATION",
        "home_dashboard_preview": "Making Cities Smarter",
        "home_dashboard_desc": "RouteVision AI helps traffic managers, city planners, and commuters make better decisions through real-time data analysis and predictive modeling. Our platform reduces congestion, emissions, and travel times.",
        "home_footer": "© 2025 RouteVision AI - Transforming Urban Mobility Through Artificial Intelligence",
        "home_button": "🏠 Home",
    }
}

with st.sidebar:
    selected_lang = st.toggle("🇻🇳", value="vi", key="lang_selector") and "vi" or "en"
    if selected_lang != st.session_state.lang:
        st.session_state.lang = selected_lang

# Add home button when not on home page
if st.session_state.page != 'home':
    # Hidden button to trigger the action
    if st.button(lang_mapping[st.session_state.lang]["home_button"]
        , key="home-button-hidden", help="Return to homepage"):
        st.session_state.page = 'home'
    
    

# Logic to navigate to different Python files
if st.session_state.page == 'home':
    # Animated road at the top
    st.markdown(
        """
        <div class="road-container">
            <div class="road">
                <div class="road-line"></div>
                <div class="road-line"></div>
                <div class="road-line"></div>
                <div class="road-line"></div>
                <div class="road-line"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Header with traffic-themed design
    st.markdown(
        """
        <div class="main-header">
            <h1 class="header-title">
                {home_title}
            </h1>
            <h3 class="header-subtitle">
                {home_subtitle}
            </h3>
        </div>
        """.format(
            home_title=lang_mapping[st.session_state.lang]["home_title"],
            home_subtitle=lang_mapping[st.session_state.lang]["home_subtitle"]
        ),
        unsafe_allow_html=True
    )
    
    # Create horizontal layout with enhanced cards and WORKING BUTTONS
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-container">
                    <div class="feature-icon">🎥</div>
                    <div class="feature-title">{home_traffic_feed}</div>
                    <div class="feature-description">{home_traffic_feed_desc}</div>
                </div>
            </div>
            """
            .format(
                home_traffic_feed=lang_mapping[st.session_state.lang]["home_traffic_feed"],
                home_traffic_feed_desc=lang_mapping[st.session_state.lang]["home_traffic_feed_desc"]
            ),
            unsafe_allow_html=True
        )
        if st.button(lang_mapping[st.session_state.lang]["home_traffic_feed"], key="traffic_btn"):
            st.session_state.page = 'traffic_video'
    
    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-container">
                    <div class="feature-icon">🗺️</div>
                    <div class="feature-title">{home_route_optimizer}</div>
                    <div class="feature-description">{home_route_optimizer_desc}</div>
                </div>
            </div>
            """
            .format(
                home_route_optimizer=lang_mapping[st.session_state.lang]["home_route_optimizer"],
                home_route_optimizer_desc=lang_mapping[st.session_state.lang]["home_route_optimizer_desc"]
            ),
            unsafe_allow_html=True
        )
        if st.button(lang_mapping[st.session_state.lang]["home_route_optimizer"]
            , key="route_btn"):
            st.session_state.page = 'route_optimize_predictor'
    
    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-container">
                    <div class="feature-icon">🚦</div>
                    <div class="feature-title">{home_signal_simulation}</div>
                    <div class="feature-description">{home_signal_simulation_desc}</div>
                </div>
            </div>
            """.format(
                home_signal_simulation=lang_mapping[st.session_state.lang]["home_signal_simulation"],
                home_signal_simulation_desc=lang_mapping[st.session_state.lang]["home_signal_simulation_desc"]
            ),
            unsafe_allow_html=True
        )
        if st.button("RUN SIMULATION", key="signal_btn"):
            st.session_state.page = 'smart_signal'
    
    # Another road animation
    st.markdown(
        """
        <div class="traffic-line"></div>
        """,
        unsafe_allow_html=True
    )
    
    # Dashboard preview section (optional)
    st.markdown(
        """
        <div style="text-align: center; margin: 40px 0 20px 0;">
            <h2 style="color: #1e3c72; font-size: 28px; margin-bottom: 15px;">{home_dashboard_preview}</h2>
            <p style="color: #555; max-width: 800px; margin: 0 auto; font-size: 16px; line-height: 1.6;">
                {home_dashboard_desc}
            </p>
        </div>
        """.format(
            home_dashboard_preview=lang_mapping[st.session_state.lang]["home_dashboard_preview"],
            home_dashboard_desc=lang_mapping[st.session_state.lang]["home_dashboard_desc"]
        ),
        unsafe_allow_html=True
    )
    
    # Footer section
    st.markdown(
        """
        <div class="footer">
            <p style="margin: 0; color: #666; font-size: 14px;">{home_footer}</p>
        </div>
        """.format(
            home_footer=lang_mapping[st.session_state.lang]["home_footer"]
        )
        ,
        unsafe_allow_html=True
    )
    
else:
    # Simulate the redirection by running the corresponding Python file
    try:
        if st.session_state.page == 'traffic_video':
            exec(open(r"pages/Traffic_Video.py", encoding="utf-8").read())
        elif st.session_state.page == 'route_optimize_predictor':
            exec(open(r"pages/predictive_analysis_route_analysis.py", encoding="utf-8").read())
        elif st.session_state.page == 'smart_signal':
            exec(open(r"pages/SignalSimulation.py", encoding="utf-8").read())
    except Exception as e:
        st.error(f"Error loading page: {e}")
        st.button("Return to Home", on_click=lambda: setattr(st.session_state, 'page', 'home'))
