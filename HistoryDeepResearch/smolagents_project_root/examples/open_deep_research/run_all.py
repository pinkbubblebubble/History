import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import time
import sys
import logging
from scripts.reverse_image import GoogleLensSearchTool
import re
import traceback
import numpy as np
import pandas as pd
import streamlit as st
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
import datasets
from datasets import Dataset
from dotenv import load_dotenv
from smolagents import (
    CodeAgent,
    GoogleSearchTool,
    LiteLLMModel,
    ToolCallingAgent,
)
from scripts.reformulator import prepare_response
from scripts.run_agents import (
    get_single_file_description,
    get_zip_description,
)
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
from scripts.image_web_browser import (
    SimpleImageBrowser,
    SearchInformationTool_Image,
    VisitTool_Image,
    ArchiveSearchTool_Image,
    PageUpTool_Image,
    PageDownTool_Image,
    FinderTool_Image,
    FindNextTool_Image,
    ImageContextExtractorTool,
    VisitImageSearchResultsTool,
    SaveHTMLTool,
)
from scripts.file_processing import (
    FileProcessor,
    OCRTool,
    PDFTool,
    DOCXTool,
    XLSXTool,
    PPTXTool,
    ImageAnalysisTool,
)

# from scripts.web_tools import (
#     LiteratureSearchingTool,
#     GeneralBrowserTool,
#     RelevantLiteratureFinderTool,
# )

from scripts.visual_qa import visualizer
from tqdm import tqdm

from smolagents import (
    # HfApiModel,
    LiteLLMModel,
    Model,
)
from scripts.translator import TranslatorTool
from scripts.speech_recognition import SpeechRecognitionTool
from scripts.ocr import OCRTool

import openai  # 用于调用GPT API
import base64
from smolagents.models import MessageRole
from dataset_loader import load_custom_dataset


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

SHARED_CONFIG = {
    "downloads_folder": "downloads",
    "ocr_languages": ["en", "ch_sim"],
    "speech_model": "google",
    "translation_url": "http://127.0.0.1:5000/translate",
    "imgbb_api_key": os.getenv("IMGBB_API_KEY"),
    "serpapi_api_key": os.getenv("SERPAPI_API_KEY")
}

BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

BROWSER_CONFIG_IMAGE = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "image_downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),  # 保留这个因为SimpleTextBrowser支持
}

# 将Google Lens相关的配置单独保存
GOOGLE_LENS_CONFIG = {
    "imgbb_api_key": os.getenv("IMGBB_API_KEY"),
    "serpapi_api_key": os.getenv("SERPAPI_API_KEY")
}

def create_agent(model_id="o1"):
    text_limit = 100000
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
    browser_image = SimpleImageBrowser(**BROWSER_CONFIG_IMAGE)
    Image_Reverse_Search_Tool = GoogleLensSearchTool(
        imgbb_api_key=GOOGLE_LENS_CONFIG["imgbb_api_key"],
        serpapi_api_key=GOOGLE_LENS_CONFIG["serpapi_api_key"]
    )
    Image_Reverse_Search_Tool.name = "Image_Reverse_Search_Tool" 

    file_processor = FileProcessor(
        ocr_languages=SHARED_CONFIG["ocr_languages"],
        model=tool_model
    )

    pdf_tool = PDFTool(file_processor)
    xlsx_tool = XLSXTool(file_processor)
    docx_tool = DOCXTool(file_processor)
    pptx_tool = PPTXTool(file_processor)

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

    IMAGE_SEARCH_TOOLS = [
        SearchInformationTool_Image(browser_image),
        VisitTool_Image(browser_image),
        PageUpTool_Image(browser_image),
        PageDownTool_Image(browser_image),
        FinderTool_Image(browser_image),
        FindNextTool_Image(browser_image),
        ArchiveSearchTool_Image(browser_image),
        TextInspectorTool(model, text_limit),
        ImageContextExtractorTool(browser_image),
        VisitImageSearchResultsTool(browser_image),
        SaveHTMLTool(browser_image),
    ]

    # LITERATURE_SEARCH_TOOLS = [
    #     LiteratureSearchingTool(api_key=os.getenv("OPENAI_API_KEY"), download_path="downloads_folder"),
    #     GeneralBrowserTool(api_key=os.getenv("OPENAI_API_KEY"), download_path="downloads_folder"),
    #     RelevantLiteratureFinderTool(api_key=os.getenv("OPENAI_API_KEY"), download_path="downloads_folder"),
    # ]
    FILE_TOOLS = [
        ImageAnalysisTool(file_processor, model),  # 新增：图像分析工具
        pdf_tool,
        docx_tool,
        xlsx_tool,
        pptx_tool
    ]

    # FILE_TOOLS = [
    #     ImageAnalysisTool(file_processor, model),  # 新增：图像分析工具
    #     pdf_tool,
    #     docx_tool,
    #     xlsx_tool,
    #     pptx_tool
    # ]

    ocr_tool = OCRTool(model)
    ocr_agent = ToolCallingAgent(
        model=model,
        tools=[ocr_tool],
        max_steps=5,
        verbosity_level=2,
        planning_interval=2,
        name="ocr_agent",
        description="""Agent specialized in image text recognition.
        
Features:
1. Extract text content from images
2. Automatically detect languages in images
3. Support multi-language OCR processing
4. Provide image content description when OCR fails

Use cases:
- Extract text from screenshots, scanned documents, or photos
- Process charts, images, or documents containing text
- Recognize mixed multi-language content in images
        """,
        provide_run_summary=True,
    )

    speech_tool = SpeechRecognitionTool(model)
    speech_recognition_agent = ToolCallingAgent(
        model=model,
        tools=[speech_tool],
        max_steps=3,
        verbosity_level=2,
        planning_interval=1,
        name="speech_recognition_agent",
        description="""Agent specialized in speech recognition.
        
Features:
1. Convert speech in audio files to text
2. Support processing of multiple audio formats
3. Use Google Speech Recognition API for transcription

Use cases:
- Transcribe recordings, voice notes, or audio meetings
- Process voice commands or voice messages
- Analyze audio content
        """,
        provide_run_summary=True,
    )

    translator_tool = TranslatorTool()
    translator_agent = ToolCallingAgent(
        model=model,
        tools=[translator_tool],
        max_steps=3,
        verbosity_level=2,
        planning_interval=1,
        name="translator_agent",
        description="""Agent specialized in text translation.
        
Features:
1. Translate text to different languages
2. Automatically detect source language
3. Support conversion between multiple languages
4. Use specialized translation methods for special languages

Use cases:
- Translate foreign language text
- Process multilingual content
- Cross-language communication and understanding
        """,
        provide_run_summary=True,
    )

    print("🚩 Step H: creating ToolCallingAgent")
    text_webbrowser_agent = ToolCallingAgent(
        model=model,
        tools=WEB_TOOLS,
        max_steps=12,
        verbosity_level=2,
        planning_interval=4,
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
    Ask him for all your questions that require browsing the web.
    
    IMPORTANT LANGUAGE GUIDELINES:
    - When searching content in a specific language (e.g., German, Chinese), try to use search terms in that original language first and then use English as a backup.
    - For German content, try to formulate your search query in German
    - For Chinese content, try to formulate your search query in Chinese
    - Using the original language often yields better and more relevant results
    
    Provide him as much context as possible, in particular if you need to search on a specific timeframe!
    And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
    Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
        """,
        provide_run_summary=True,
    )

    text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
    If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
    Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""


    image_information_agent = ToolCallingAgent(
        model=model,
        tools=[*IMAGE_SEARCH_TOOLS, Image_Reverse_Search_Tool], 
        max_steps=20,
        verbosity_level=2,
        planning_interval=4,
        name="image_information_agent",
        description="""I am the essential first step for processing all images! I'm responsible for extracting and analyzing all information in images.
    I am the primary agent responsible for comprehensive image information. My key functions include:

    1. **Image_Reverse_Search_Tool**:
    - This tool performs reverse image search to identify the origin and related web pages of the image.
    - It helps find where an image appears online and what information exists about it.
    - Use this tool first when analyzing any image to discover its online presence.

    2. **VisitImageSearchResultsTool**:
    - This tool specifically visits the top results from reverse image search.
    - It automatically processes multiple links found during reverse image search.
    - Use this after Image_Reverse_Search_Tool to quickly visit multiple search results.

    3. **VisitTool**:
    - This tool allows you to visit any specific web page by its URL.
    - Use it to access particular web pages of interest that may contain information about the image.
    - Helpful for visiting specific pages mentioned in search results or for deeper investigation.

    4. **ImageContextExtractorTool**:
    - This tool extracts text and visual information from the surroundings of image URLs found during reverse image search.
    - It helps analyze the relationship between the image and its surrounding content on web pages.
    - Use this tool to get specific context about how the image is presented online.
    
    Trigger conditions:
    - Any question containing an image file path
    - For example: "What is this image: xxx.jpg"
    - For example: "Analyze this image: xxx.png"
    - For example: "What is written on this bamboo slip: xxx.png"
    - For example: "What building is in this photo: xxx.jpg"
    
    Important notes:
    - I should be the first step in processing any image
    - Even if the image contains Chinese text, I should process it first
    - In analysis, distinguish between the content of the image itself and related information found online
    - Other tools (such as file_processor) should only be used after I complete my analysis
    """,
        provide_run_summary=True
    )

    image_information_agent.prompt_templates["managed_agent"]["task"] += """
You are the `image_information_agent`, responsible for extracting and analyzing information from images. You have four primary tools at your disposal:

1. **Image_Reverse_Search_Tool**:
   - Purpose: Find where an image appears online and discover related information.
   - Usage: `Image_Reverse_Search_Tool: /path/to/image.jpg`
   - When to use: This should be your first step when analyzing any image.
   - Output: Provides links to web pages where the image or similar images appear.

2. **VisitImageSearchResultsTool**:
   - Purpose: Automatically visit multiple top results from reverse image search.
   - Usage: `VisitImageSearchResultsTool: [search results from Image_Reverse_Search_Tool]`
   - When to use: After running Image_Reverse_Search_Tool, to efficiently visit multiple results at once.
   - What to look for: Overview information from multiple sources about the image.
   - Advantage: Saves time by visiting multiple pages in one operation.

3. **VisitTool**:
   - Purpose: Visit a specific web page to gather detailed information.
   - Usage: `VisitTool: https://example.com/page-with-image`
   - When to use: When you need to examine a particular web page in detail.
   - What to look for: Detailed information such as:
     * Product descriptions and specifications
     * Historical context and background information
     * Auction details and provenance information
     * Artist or creator information
     * Dating and authentication details
     * Any other relevant contextual information
   - Advantage: Allows focused analysis of a single important page.

4. **ImageContextExtractorTool**:
   - Purpose: Extract specific information around the image on a web page.
   - Usage: `ImageContextExtractorTool: [image filename or URL]`
   - When to use: When you need to focus on the immediate context of an image on a web page.
   - Output: Provides text and information surrounding the image on the page.
   - Advantage: Focuses specifically on content directly related to the image.

**Recommended Workflow**:
1. Start with `Image_Reverse_Search_Tool` to find where the image appears online.
2. Use `VisitImageSearchResultsTool` to quickly visit multiple top search results.
3. Use `VisitTool` to examine specific important pages in more detail.
4. Use `ImageContextExtractorTool` to extract specific context around the image when necessary.
5. Integrate all findings into a comprehensive report about the image.

**IMPORTANT: DISTINGUISHING EXAMPLES FROM ACTUAL TASKS**
The following is just an EXAMPLE to illustrate the workflow. DO NOT process 'historical_document.png' unless it's specifically mentioned in the actual task:

   - *Example Task*: Analyze 'historical_document.png'.
   - *Example Process*:
     - Use `Image_Reverse_Search_Tool: historical_document.png` to find online sources
     - Use `VisitImageSearchResultsTool: [search results]` to visit multiple top results
     - Use `VisitTool: https://specific-page.com` for any specific page that needs detailed examination
     - Use `ImageContextExtractorTool: historical_document.png` to extract specific context
     - Integrate findings into a report

Your objective is to process only the actual images mentioned in the current task, not any examples used for illustration.

Your task is:
{task}

Begin by identifying any image file paths in this task and using Image_Reverse_Search_Tool.
"""

    file_processor_agent = ToolCallingAgent(
        model=model,
        tools=FILE_TOOLS,
        max_steps=20,
        verbosity_level=2,
        planning_interval=4,
        name="file_processor",
        description="""A specialized team member for processing various types of files:
1. Automatic File Type Detection: 
   - Files are automatically analyzed to determine their type
   - No need to specify file type in your requests
   - Just provide the file path and the appropriate tool will be selected

2. OCR: Extract ONLY the plain text from images using EasyOCR
   - Returns EXACTLY the text content with no analysis or additional information
   - Input: image file path
   - Output: extracted text only
   
3. Image Analysis: Analyze and describe image content in detail
   - Provides detailed descriptions of what appears in the image
   - Input: image file path
   - Output: comprehensive description of the image content
   
4. Speech Recognition: Convert speech to text
   - Input: audio file path (.wav, .mp3, etc.)
   - Output: transcribed text
   
5. Translation: Translate text between languages
   - Input: text and target language code (e.g., 'en', 'zh')
   - Output: translated text
""",
        provide_run_summary=True,
    )
    file_processor_agent.prompt_templates["managed_agent"]["task"] += """
File Type Detection:
- The system automatically detects file types based on file extension or content analysis
- Simply provide the file path without specifying the file type
- Example: "Extract content from this file: /path/to/file.ext" instead of "Extract text from this image: /path/to/image.png"

For image files (detected automatically):
- Supported formats: .png, .jpg, .jpeg, .bmp
- Two processing options:
  1. Text extraction using OCR - when you need to extract text from the image
  2. Image analysis - when you need to understand the image content and get a detailed description
- Example: "Extract text from this image: /path/to/image.jpg" for OCR
- Example: "Analyze this image: /path/to/image.jpg" for visual description

For document files (detected automatically):
- Supported formats: .pdf, .docx, .xlsx, .pptx
- Text extraction is applied based on document type

For text translation:
- Use TranslatorTool with appropriate language codes
- Common codes: 'en' (English), 'zh' (Chinese), 'ja' (Japanese), 'ko' (Korean)

If you encounter any issues:
- Check if file exists
- Verify file path is correct
- Use `final_answer` with error description if file is inaccessible or format unsupported
"""

    print("🚩 Step J: creating CodeAgent (manager)")
    manager_agent = CodeAgent(
        model=model,
        tools=[visualizer, TextInspectorTool(model, 100000)],
        max_steps=12,
        verbosity_level=2,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        planning_interval=4,
        managed_agents=[text_webbrowser_agent, image_information_agent, file_processor_agent, ocr_agent, speech_recognition_agent, translator_agent],
    )
    manager_agent.description = """Team manager responsible for coordinating specialized agents to solve complex tasks.

You have access to the following agents:
1. text_webbrowser_agent - For web searches and browsing
2. image_information_agent - For image-related web information and reverse image search
3. file_processor_agent - For processing various file types including PDFs, DOCX, Excel, PPTX, and visual/image/audio content
4. ocr_agent - For extracting plain text from images using OCR
5. translator_agent - For translating text between languages
6. speech_recognition_agent - For transcribing speech content from audio files

CRITICAL RULES:

🖼️ IMAGE PROCESSING:
- For any image file (e.g., .jpg, .png, .jpeg), you MUST FIRST delegate to `image_information_agent`
- NEVER use `visualizer`, `ocr_agent`, or `file_processor_agent` directly on images
- This rule is NON-NEGOTIABLE and applies to:
  - Historical documents (e.g., bamboo slips)
  - Diagrams, symbols, or natural scenes
- All follow-up actions must build upon the result from `image_information_agent`

🔊 AUDIO FILES:
- You MUST use `speech_recognition_agent` for .mp3, .wav, .flac, or .m4a files
- DO NOT use `inspect_file_as_text` or other tools on audio files

📄 TEXT FILES:
- Use `inspect_file_as_text` ONLY for .txt plain text files
- DO NOT misuse it for structured documents or scanned content

📄 MULTIMODAL TASKS:
- When multiple file types are present (e.g., an image and a PDF), delegate to the proper agent per file type
- Coordinate results across agents to form a final answer
"""

    manager_agent.prompt_templates["task"] = """You are the manager of a team of specialized agents. Your job is to coordinate their work to solve complex tasks.

You have access to the following agents:
1. text_webbrowser_agent - For web searches and browsing
2. image_information_agent - For image-related web information and reverse image search
3. file_processor_agent - For processing various file types including PDFs, DOCX, Excel, PPTX, and visual/image/audio content
4. ocr_agent - For extracting plain text from images using OCR
5. translator_agent - For translating text between languages
6. speech_recognition_agent - For transcribing speech content from audio files

========================
CRITICAL RULES FOR FILES
========================

🖼️ IMAGE PROCESSING
- You MUST FIRST delegate to `image_information_agent` for any image file (.jpg, .png, .jpeg, etc.)
- You are FORBIDDEN from using `visualizer`, `file_processor_agent`, or `ocr_agent` directly on image files
- NO EXCEPTIONS, even for images containing Chinese characters or historical content

🔊 AUDIO FILE PROCESSING
- You MUST use `speech_recognition_agent` for .mp3, .wav, .flac, etc.
- NEVER use `inspect_file_as_text` or `file_processor_agent` on audio files

📄 TEXT FILES
- Only use `inspect_file_as_text` for .txt files
- DO NOT use it for structured or non-text formats (PDFs, images, audio, etc.)

===============================
MANDATORY IMAGE PROCESSING WORKFLOW
===============================

Example:
Question: What is the content of 'document.png'?

- Step 1: `image_information_agent: document.png`
- Step 2: `visit_image_search_results: document.png`
- Step 3: `visualizer: document.png` (only after step 1)
- Step 4: `file_processor: document.png` (only after step 1)

Do not skip step 1. Do not use visualizer first.

Your task is:
{task}

🛑 REMINDER:
- For ANY image file path: Start with `image_information_agent`
- For ANY audio file: Use `speech_recognition_agent`
- For ANY structured file: Use `file_processor_agent`
- For translation: Use `translator_agent`
- Use each agent exactly as assigned above.
"""


    print("✅ Step K: returning agent")
    return manager_agent


def main():
    args = parse_args()
    agent = create_agent(model_id=args.model_id)
    answer = agent.run(args.question)
    print(f"Got this answer: {answer}")

def test_huggingface_login_and_folder():
    print("🚩 Step A: huggingface login")
    hf_token = os.getenv("HF_TOKEN") or (st.secrets.get("HF_TOKEN") if hasattr(st, "secrets") else None)
    if not hf_token:
        raise ValueError("❌ HF_TOKEN is missing in environment or Streamlit secrets!")

    login(hf_token)
    print("✅ Huggingface login successful")

    print("🚩 Step B: make downloads folder")
    downloads_folder = "downloads_folder"
    os.makedirs(downloads_folder, exist_ok=True)
    assert os.path.exists(downloads_folder), "❌ Failed to create downloads folder"
    print("✅ Downloads folder ready")

if __name__ == "__main__":
    test_huggingface_login_and_folder()