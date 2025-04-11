import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv
import sys
import re


from sidebar import sidebar

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Agent creation
from run_local import create_agent

class StreamCapture:
    def __init__(self, thinking_container):
        self.thinking_container = thinking_container
        with self.thinking_container:
            self.step_container = st.container()
        self.current_step = 0

    def clean_text(self, text):
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        cleaned = ansi_escape.sub('', text)
        cleaned = re.sub(r'\[.*?m', '', cleaned)
        markdown_chars = [
            '#', '*', '_', '`', '~', '-', '+', '>', '•',
            '|', '│', '├', '─', '━', '═', '║', '╔', '╗', '╚', '╝',
            '▌', '⠀', '⠅', '⣿', '\u200b', '\u200c', '\u200d', '\ufeff'
        ]
        for char in markdown_chars:
            cleaned = cleaned.replace(char, '')
        cleaned = re.sub(r'\[.*?\]', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()

    def write(self, text):
        clean_text = self.clean_text(text)
        if not clean_text:
            return
        if "Step" in clean_text:
            with self.step_container:
                st.info(f"{clean_text} 🔍")
        elif "Final answer" in clean_text:
            with self.step_container:
                st.info(f"{clean_text} ✅")
        else:
            with self.step_container:
                st.info(clean_text)

    def flush(self):
        pass

# --- UI Title ---
st.set_page_config(page_title="History Deep Research", page_icon="💬", layout="wide")
st.title("💬 History Deep Research Chatbot")
st.caption("🚀 Let's chat! Upload a file to enhance responses.")

sidebar()

# --- System prompt for OpenAI fallback ---
SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are a helpful AI assistant. If a file is uploaded, use its content to enhance responses."
}

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [SYSTEM_PROMPT]
    st.session_state["file_content"] = None

if "agent" not in st.session_state:
    st.session_state["agent"] = create_agent(model_id="gpt-4o")

# --- Display Chat History ---
for msg in st.session_state.messages[1:]:  # skip system prompt
    st.chat_message(msg["role"]).write(msg["content"])

# --- File Upload (right above input box) ---
# uploaded_file = st.file_uploader("📎 Upload a `.txt` file (optional)", type=["txt"])
# if uploaded_file:
#     file_content = uploaded_file.read()
#     try:
#         decoded_content = file_content.decode()
#         st.session_state["file_content"] = decoded_content
#         st.toast("📄 File uploaded successfully! AI will use it for context.")
#     except UnicodeDecodeError:
#         st.warning("⚠️ This file is not a readable text file.")
# else:
#     st.session_state["file_content"] = None

import tempfile

uploaded_file = st.file_uploader("📎 Upload any file (optional)", type=None)  # allow all file types
if uploaded_file:
    try:
        # 保存上传的文件到临时目录
        suffix = os.path.splitext(uploaded_file.name)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name

        st.session_state["file_path"] = tmp_file_path
        st.session_state["file_name"] = uploaded_file.name
        st.toast(f"📄 File '{uploaded_file.name}' uploaded and saved to {tmp_file_path}")
    except Exception as e:
        st.warning(f"⚠️ Failed to save uploaded file: {e}")
        st.session_state["file_path"] = None
        st.session_state["file_name"] = None
else:
    st.session_state["file_path"] = None
    st.session_state["file_name"] = None


# --- Chat Input ---
if user_input := st.chat_input("Type your message here..."):
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    try:
        with st.spinner("Researching your question..."):
            user_message = user_input
            if st.session_state.get("file_path"):
                user_message += f"\n\nThe user has uploaded a file named '{st.session_state['file_name']}' located at:\n{st.session_state['file_path']}\n"
            
            if st.session_state["file_content"]:
                user_message += "\n\nHere is the uploaded file content:\n"
                user_message += st.session_state["file_content"]

            with st.expander("💭 Agent's Thinking Process", expanded=True):
                thinking_container = st.container()
                stream_capture = StreamCapture(thinking_container)

                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = stream_capture
                sys.stderr = stream_capture

                try:
                    agent_response = st.session_state["agent"].run(user_message)
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr

            with st.chat_message("assistant"):
                st.write(agent_response)

            st.session_state.messages.append({"role": "assistant", "content": agent_response})

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        context = user_input
        if st.session_state["file_content"]:
            context = (
                f"The user has uploaded a file. Here is its content:\n\n"
                f"{st.session_state['file_content']}\n\nUser's message: {user_input}"
            )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=st.session_state.messages[-10:] + [{"role": "user", "content": context}],
            temperature=0.2,
        )
        response_text = response.choices[0].message.content

        with st.chat_message("assistant"):
            st.write(response_text)

        st.session_state.messages.append({"role": "assistant", "content": response_text})

