import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv
import sys
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

from run import create_agent

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
        markdown_chars = ['#', '*', '_', '`', '~', '-', '+', '>', '•', '|', '│', '├', '─', '━', '═', '║', '╔', '╗', '╚', '╝', '▌', '⠀', '⠅', '⣿', '\u200b', '\u200c', '\u200d', '\ufeff']
        for char in markdown_chars:
            cleaned = cleaned.replace(char, '')
        cleaned = re.sub(r'\[.*?\]', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()

    def write(self, text):
        clean_text = self.clean_text(text)
        if not clean_text:
            return
        with self.step_container:
            if "Step" in clean_text:
                st.info(f"{clean_text} 🔍")
            elif "Final answer" in clean_text:
                st.info(f"{clean_text} ✅")
            else:
                st.info(clean_text)

    def flush(self):
        pass
    
st.write("✅ Starting app")

try:
    import run
    st.write("✅ Imported run module")
except Exception as e:
    st.error(f"❌ Failed to import run: {e}")

# UI
st.title("💬 History Deep Research Chatbot")
st.caption("🚀 Let's chat! Upload a file to enhance responses.")
st.write("✅ App started")

uploaded_file = st.file_uploader("Upload a file (optional)")
client = OpenAI(api_key=openai_api_key)

SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are a helpful AI assistant. If a file is uploaded, use its content to enhance responses."
}

if "messages" not in st.session_state:
    st.session_state["messages"] = [SYSTEM_PROMPT]
    st.session_state["file_content"] = None

if uploaded_file:
    file_content = uploaded_file.read()
    try:
        decoded_content = file_content.decode()
        st.session_state["file_content"] = decoded_content
        st.toast("📄 File uploaded successfully! AI will use it for context.")
    except UnicodeDecodeError:
        st.warning("This file is not a readable text file. AI may not be able to use it.")

# Display past chat messages
for msg in st.session_state.messages[1:]:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat input
if user_input := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    try:
        with st.spinner("Researching your question..."):
            # Lazy agent creation
            if "agent" not in st.session_state:
                try:
                    with st.spinner("Initializing agent..."):
                        st.session_state["agent"] = create_agent(model_id="gpt-4o")
                        st.success("🤖 Agent initialized successfully.")
                except Exception as init_err:
                    st.error(f"Failed to create agent: {init_err}")
                    st.stop()

            user_message = user_input
            if st.session_state["file_content"]:
                user_message += "\n\nHere is the uploaded file content:\n"
                user_message += st.session_state["file_content"]

            with st.expander("💭 Agent's Thinking Process", expanded=True):
                thinking_container = st.container()
                stream_capture = StreamCapture(thinking_container)
                old_stdout, old_stderr = sys.stdout, sys.stderr
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
        st.error(f"Agent failed: {str(e)}")
        context = user_input
        if st.session_state["file_content"]:
            context = f"The user uploaded a file. Content:\n\n{st.session_state['file_content']}\n\nUser: {user_input}"

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=st.session_state.messages[-10:] + [{"role": "user", "content": context}],
            temperature=0.2,
        )
        response_text = response.choices[0].message.content
        with st.chat_message("assistant"):
            st.write(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
