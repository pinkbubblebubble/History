import streamlit as st

from dotenv import load_dotenv
import os

load_dotenv()


def sidebar():
    with st.sidebar:
        st.markdown(
            "## How to use\n"
            "1. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) below🔑\n" \
            "**However, it's now a test version, so we have already set the OpenAI API key for you**.\n"  # noqa: E501
            "2. Ask a question about the history💬\n"
            "3. If needed, you can upload a pdf, docx, or txt file📄\n"
        )
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="We have already set it for you since it's a test mode",
            help="You can get your API key from https://platform.openai.com/account/api-keys.",  # noqa: E501
            # value=os.environ.get("OPENAI_API_KEY", None)
            # or st.session_state.get("OPENAI_API_KEY", ""),
        )

        st.session_state["OPENAI_API_KEY"] = api_key_input
        st.markdown("---")
        st.markdown("# About")
        st.markdown(
            "📚 **History Deep Research** is a multimodal AI platform that combines "
            "large language models (LLMs) with a multi-agent tool framework "
            "to assist with in-depth historical material analysis. \n\n"
            "The system supports multiple functions:\n"
            "- Text Web Browser\n"
            "- Literature Finder\n"
            "- OCR\n"
            "- Speech Recognition\n"
            "- Translation \n"
            "- Reverse Image Search\n"
            "- File Processing\n\n"
            "The platform emphasizes multi-source verification, context-aware reasoning, and the use "
            "of expert-level references. It has shown significantly better performance than traditional baselines."
        )
        st.markdown("---")



        st.markdown(
                """
        # FAQ
        ## How does History Deep Research work?
        When you upload a document, image, or audio file, the system uses multiple AI tools—like OCR, reverse image search, and LLM-based summarization—to process and extract information.

        A central agent called **CodeAgent** coordinates tasks between sub-agents such as the Web Browser Agent, File Processor, Literature Finder, and Image Information Agent. These tools help analyze historical content across different media formats.

        ## What makes it different from a regular search engine?
        Unlike a regular search engine, this system uses reasoning steps powered by LLMs to analyze the content—such as comparing bronze artifacts with historical databases or identifying painting themes using reverse image search and cultural cross-referencing.

        ## Is my data safe?
        Yes, your uploaded data is only used during the session. It is not stored on our servers after the session ends.

        ## Why does it take a while to generate answers?
        Complex questions often involve multiple tools (OCR, translation, image analysis) and multiple LLM calls. If you are using a free OpenAI API key, rate limits may also slow down the response.

        ## How accurate are the answers?
        The answers are based on the best available sources and AI reasoning. However, like any AI system, it can make mistakes, especially if the data is ambiguous or missing context. You should always verify critical findings with primary sources.

        ## What types of files can I upload?
        You can upload PDF, DOCX, PPTX, image files (JPG, PNG), and audio files (MP3, WAV). The system will automatically route them to the appropriate agents for processing.

        ## What is CodeAgent?
        CodeAgent is the manager that coordinates different tools in the SmolAgent framework. It ensures that each task—like image recognition, document analysis, or web search—is assigned to the right agent.
        """
            )
