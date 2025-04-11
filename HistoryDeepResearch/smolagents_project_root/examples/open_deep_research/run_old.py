import argparse
import os
import threading
import sys
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from dotenv import load_dotenv
from scripts.text_inspector_tool import TextInspectorTool
from scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SimpleTextBrowser,
    VisitTool,
)
from scripts.visual_qa import visualizer
from huggingface_hub import login

from smolagents import (
    CodeAgent,
    GoogleSearchTool,
    LiteLLMModel,
    ToolCallingAgent,
)

AUTHORIZED_IMPORTS = [
    "requests", "zipfile", "os", "pandas", "numpy", "sympy", "json", "bs4",
    "pubchempy", "xml", "yahoo_finance", "Bio", "sklearn", "scipy", "pydub",
    "io", "PIL", "chess", "PyPDF2", "pptx", "torch", "datetime", "fractions", "csv"
]

load_dotenv(override=True)

append_answer_lock = threading.Lock()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str, help="e.g., 'How many albums did Mercedes Sosa release before 2007?'")
    parser.add_argument("--model-id", type=str, default="gpt-4o")
    return parser.parse_args()

custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

user_agent = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
)

BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

def create_agent(model_id="o1"):
    print("🚩 Step A: calling huggingface login")
    from huggingface_hub import login
    hf_token = st.secrets.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("❌ HF_TOKEN is missing in environment!")
    login(hf_token)

    print("🚩 Step B: making downloads folder")
    os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)

    print("🚩 Step C: selecting model token limits")
    if model_id == "gpt-3.5-turbo":
        max_tokens = 4096
        tool_max_tokens = 4096
    elif model_id == "o1":
        max_tokens = 8192
        tool_max_tokens = 8192
    elif model_id == "gpt-4o":
        max_tokens = 16000
        tool_max_tokens = 16000
    else:
        max_tokens = 4096
        tool_max_tokens = 4096

    print("🚩 Step D: initializing main model")
    model = LiteLLMModel(
        model_id,
        custom_role_conversions=custom_role_conversions,
        max_completion_tokens=max_tokens,
    )

    print("🚩 Step E: initializing tool model")
    tool_model = LiteLLMModel(
        "gpt-4o",
        custom_role_conversions=custom_role_conversions,
        max_completion_tokens=tool_max_tokens,
    )

    print("🚩 Step F: initializing browser")
    browser = SimpleTextBrowser(**BROWSER_CONFIG)

    print("🚩 Step G: creating web tools")
    WEB_TOOLS = [
        GoogleSearchTool(provider="serpapi"),
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(model, 100000),
    ]

    print("🚩 Step H: creating ToolCallingAgent")
    text_webbrowser_agent = ToolCallingAgent(
        model=tool_model,
        tools=WEB_TOOLS,
        max_steps=20,
        verbosity_level=2,
        planning_interval=4,
        name="search_agent",
        description="Search agent with browser tools",
        provide_run_summary=True,
    )

    print("🚩 Step I: setting prompt extensions")
    text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += (
        "You can navigate to .txt files. Use 'inspect_file_as_text' for PDF/Youtube."
    )

    print("🚩 Step J: creating CodeAgent (manager)")
    manager_agent = CodeAgent(
        model=model,
        tools=[visualizer, TextInspectorTool(model, 100000)],
        max_steps=12,
        verbosity_level=2,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        planning_interval=4,
        managed_agents=[text_webbrowser_agent],
    )

    print("✅ Step K: returning agent")
    return manager_agent


def main():
    args = parse_args()
    agent = create_agent(model_id=args.model_id)
    answer = agent.run(args.question)
    print(f"Got this answer: {answer}")

if __name__ == "__main__":
    main()