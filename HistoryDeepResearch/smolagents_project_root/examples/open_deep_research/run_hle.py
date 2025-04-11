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

import datasets
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import login
from smolagents import CodeAgent
from smolagents.agents import ToolCallingAgent
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
    SearchInformationTool,
    SimpleTextBrowser,
    VisitTool,
)
from scripts.image_web_browser import (
    SimpleImageBrowser,
    SearchInformationTool_Image,
    VisitTool_Image,
    DownloadTool_Image,
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
    SpeechRecognitionTool,
    OCRTool,
    TranslatorTool,
    PDFTool,
    DOCXTool,
    XLSXTool,
    PPTXTool,
    ImageAnalysisTool,
)

from scripts.web_tools import (
    LiteratureSearchingTool,
    GeneralBrowserTool,
    RelevantLiteratureFinderTool,
)

from scripts.visual_qa import visualizer
from tqdm import tqdm

from smolagents import (
    # HfApiModel,
    LiteLLMModel,
    Model,
)

import openai  # 用于调用GPT API
import base64
from smolagents.models import MessageRole
from dataset_loader import load_custom_dataset


AUTHORIZED_IMPORTS = [
    "requests",
    "zipfile",
    "os",
    "pandas",
    "numpy",
    "sympy",
    "json",
    "bs4",
    "pubchempy",
    "xml",
    "yahoo_finance",
    "Bio",
    "sklearn",
    "scipy",
    "pydub",
    "io",
    "PIL",
    "chess",
    "PyPDF2",
    "pptx",
    "torch",
    "datetime",
    "fractions",
    "csv",
]
load_dotenv(override=True)
login(os.getenv("HF_TOKEN"))

append_answer_lock = threading.Lock()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--model-id", type=str, default="gpt-4o")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--api-key", type=str, help="OpenAI API key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--use-image-agent", action="store_true", help="Enable image information agent")
    parser.add_argument("--use-file-agent", action="store_true", help="Enable file processor agent")
    parser.add_argument("--use-literature-agent", action="store_true", help="Enable literature search agent")
    parser.add_argument("--no-text-webbrowser-agent", action="store_true", help="Disable text webbrowser agent (enabled by default)")
    parser.add_argument("--results-json-path", type=str, default=None, help="Path to previous results JSON file for filtering already correct answers")
    parser.add_argument("--baseline", action="store_true", help="Use baseline agent instead of agent hierarchy")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory for results")
    parser.add_argument("--level", type=str, default="level2", choices=["level1", "level2", "level3"], help="Specify which level of questions to test")
    parser.add_argument("--question-ids", type=str, help="Comma-separated list of specific question IDs to run (e.g., '16,24,35')")
    parser.add_argument("--start-id", type=int, help="Starting question ID for a range of questions to run")
    parser.add_argument("--end-id", type=int, help="Ending question ID for a range of questions to run")
    return parser.parse_args()


### IMPORTANT: EVALUATION SWITCHES

print("Make sure you deactivated Tailscale VPN, else some URLs will be blocked!")

USE_OPEN_MODELS = False

# 设置全局变量SET，会在main函数中根据命令行参数level动态更新
SET = None

custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

### LOAD EVALUATION DATASET
# eval_ds = datasets.load_dataset("gaia-benchmark/GAIA", "2023_all")[SET]
# eval_ds = eval_ds.rename_columns({"Question": "question", "Final answer": "true_answer", "Level": "task"})

# 定义相对路径
RELATIVE_EXCEL_PATH = "Historical/Historical/Historical Q&A collections(100).xlsx"

# 转换为绝对路径
EXCEL_PATH = os.path.abspath(RELATIVE_EXCEL_PATH)
# EXCEL_PATH = "/Users/maoyc/Desktop/Agent/smolagent/datasets/History/Historical Q&A collections(100)/Historical Q&A collections(100).xlsx"

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

# 在BROWSER_CONFIG附近添加
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
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),  # 保留这个因为SimpleTextBrowser支持
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

os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)
os.makedirs(f"./{BROWSER_CONFIG_IMAGE['downloads_folder']}", exist_ok=True)


def create_agent_hierarchy(model: Model, use_image_agent=False, use_file_agent=False, use_literature_agent=False, use_text_webbrowser_agent=True, baseline=False):
    """
    创建代理层次结构或baseline代理
    
    参数:
        model: 使用的语言模型
        use_image_agent: 是否使用图像代理
        use_file_agent: 是否使用文件处理代理
        use_literature_agent: 是否使用文献搜索代理
        use_text_webbrowser_agent: 是否使用文本浏览器代理（默认启用）
        baseline: 是否使用baseline代理而不是代理层次结构
    
    返回:
        Agent: 创建的代理实例
    """
    
    # 以下是原有的代理层次结构代码
    text_limit = 100000
    ti_tool = TextInspectorTool(model, text_limit)

    # 创建浏览器实例，只传入它支持的参数
    browser = SimpleTextBrowser(**BROWSER_CONFIG)
    browser_image = SimpleImageBrowser(**BROWSER_CONFIG_IMAGE)
    # 创建Google Lens工具，使用专门的配置
    Image_Reverse_Search_Tool = GoogleLensSearchTool(
        imgbb_api_key=GOOGLE_LENS_CONFIG["imgbb_api_key"],
        serpapi_api_key=GOOGLE_LENS_CONFIG["serpapi_api_key"]
    )
    Image_Reverse_Search_Tool.name = "Image_Reverse_Search_Tool"  # 更明确的名称

    # 创建文件处理器实例
    file_processor = FileProcessor(
        ocr_languages=SHARED_CONFIG["ocr_languages"],
        speech_model=SHARED_CONFIG["speech_model"],
        translation_url=SHARED_CONFIG["translation_url"],
        downloads_folder=SHARED_CONFIG["downloads_folder"]
    )

    # 确保工具名称与提示中的一致
    pdf_tool = PDFTool(file_processor)
    xlsx_tool = XLSXTool(file_processor)
    docx_tool = DOCXTool(file_processor)
    pptx_tool = PPTXTool(file_processor)
    ocr_tool = OCRTool(file_processor, model)
    speech_tool = SpeechRecognitionTool(file_processor)

    # Web browsing tools
    WEB_TOOLS = [
        SearchInformationTool(browser),
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(model, text_limit),
    ]
    
    # 如果baseline为True，返回基础代理
    if baseline:
        text_webbrowser_agent = ToolCallingAgent(
            model=model,
            tools=WEB_TOOLS,
            max_steps=20,
            verbosity_level=2,
            planning_interval=4,
            name="search_agent",
            description="""A team member that will search the internet to answer your question.
        Ask him for all your questions that require browsing the web.
        Provide him as much context as possible, in particular if you need to search on a specific timeframe!
        And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
        Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
        """,
            provide_run_summary=True,
        )
        text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
        If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
        Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""

        manager_agent = CodeAgent(
            model=model,
            tools=[visualizer, ti_tool],
            max_steps=12,
            verbosity_level=2,
            additional_authorized_imports=AUTHORIZED_IMPORTS,
            planning_interval=4,
            managed_agents=[text_webbrowser_agent],
        )
        return manager_agent
    
    
    LITERATURE_SEARCH_TOOLS = [
        LiteratureSearchingTool(api_key=os.getenv("OPENAI_API_KEY"), download_path="downloads_folder"),
        GeneralBrowserTool(api_key=os.getenv("OPENAI_API_KEY"), download_path="downloads_folder"),
        RelevantLiteratureFinderTool(api_key=os.getenv("OPENAI_API_KEY"), download_path="downloads_folder"),
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
    
    # File processing tools
    FILE_TOOLS = [
        ocr_tool,  # 用于图片OCR
        ImageAnalysisTool(file_processor, model),  # 新增：图像分析工具
        speech_tool,
        TranslatorTool(file_processor),
        pdf_tool,
        docx_tool,
        xlsx_tool,
        pptx_tool
    ]

    # 创建 web 浏览代理
    text_webbrowser_agent = ToolCallingAgent(
        model=model,
        tools=WEB_TOOLS,
        max_steps=12,
        verbosity_level=2,
        planning_interval=4,
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
    Ask him for all your questions that require browsing the web or searching for academic literature.
    Provide him as much context as possible, in particular if you need to search on a specific timeframe!
    And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
    Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
    
    This agent has enhanced capabilities for academic and scholarly research:
    - literature_searching_task: Search for scholarly articles on a specific topic
    - relevant_literature_finder: Find and filter the most relevant literature sources
    - general_browser_task: Perform general web searches
    
    For historical research questions, use the literature search tools to find authoritative sources.
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
    I am the primary agent responsible for comprehensive image analysis. My key functions include:

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

    # 创建文学搜索代理
    literature_search_agent = CodeAgent(
        model=model,
        tools=LITERATURE_SEARCH_TOOLS,
        name="literature_search_agent",
        description="""MANDATORY AGENT for all history questions. This agent must be used for finding scholarly sources.

This specialized agent searches academic databases for authoritative historical sources with high precision.

For 'exactMatch' type questions, the EXACT original wording can be found in scholarly literature - you MUST find this original text.
For all other question types, relevant and supporting content will be found in academic sources.

Use this agent for all history questions, especially when:
- Finding scholarly articles and academic papers
- Locating specific historical research 
- Identifying authoritative sources on historical topics
- Getting information from high-citation academic works
- Finding exact references, quotes, or details from scholarly literature

This agent has specialized tools for academic research that are more powerful than general web searching.
""",
        provide_run_summary=True,
    )

    literature_search_agent.prompt_templates["managed_agent"]["task"] = """You are the `literature_search_agent`, a MANDATORY tool for all history questions. Your usage is not optional.

CRITICAL ROLE: You are the primary source of authoritative information for all history questions.

For 'exactMatch' type questions: 
- The EXACT original wording can be found in scholarly literature
- Your primary task is to locate this exact text
- The answer exists verbatim in academic sources

For all other question types:
- Relevant supporting content must be found in academic sources
- Prioritize high-quality, well-cited scholarly papers

You have three powerful tools at your disposal:

1. **literature_searching_task**:
   - Purpose: Search for high-impact, recent scholarly articles on a specific topic
   - Usage: `literature_searching_task: [research topic/query]`
   - Output: Returns 5 relevant scholarly articles with citation counts, publication years, and key findings
   - When to use: For initial broad search of authoritative academic sources

2. **relevant_literature_finder**:
   - Purpose: Filter and rank the most relevant literature sources for a specific query
   - Usage: `relevant_literature_finder: [specific research question]`
   - Output: Returns the 3 most relevant sources with relevance scores and key information
   - When to use: To pinpoint the most directly relevant sources for your question
   - CRITICAL: For exactMatch questions, use this to find the exact original wording

3. **general_browser_task**:
   - Purpose: Perform general web searches beyond academic databases
   - Usage: `general_browser_task: [search query]`
   - Output: Returns general web search results
   - When to use: Only after exhausting academic sources, for supplementary information

**Mandatory Workflow**:
1. Start with `literature_searching_task` to get a broad overview of scholarly articles
2. Use `relevant_literature_finder` with precise query terms to find the most relevant sources
   - For exactMatch questions, focus on finding the exact original wording
   - Include query terms directly from the question
3. Only after exhausting academic sources, use `general_browser_task` if needed
4. Integrate findings into a comprehensive answer with proper academic citations

Your task is:
{task}

Begin by using literature_searching_task to find scholarly articles on this topic.
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

For audio files (detected automatically):
- Supported formats: .wav, .mp3, .m4a
- Speech recognition is applied automatically
- For non-English audio, transcribe first then translate

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

    # 创建管理代理
    # 根据参数决定使用哪些代理
    managed_agents = []
    
    if use_text_webbrowser_agent:
        managed_agents.append(text_webbrowser_agent)
    
    if use_image_agent:
        managed_agents.append(image_information_agent)
    
    if use_literature_agent:
        managed_agents.append(literature_search_agent)
    
    if use_file_agent:
        managed_agents.append(file_processor_agent)
    
    manager_agent = CodeAgent(
        model=model,
        tools=[visualizer, ti_tool, pdf_tool],
        max_steps=20,
        verbosity_level=2,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        planning_interval=4,
        managed_agents=managed_agents,
        name="manager",
        description="""团队管理者，负责协调各个代理之间的工作。
        
        强制性图像处理规则
        任何包含图像文件路径的问题（如.jpg、.png、.jpeg等）：
        1. 禁止直接使用visualizer或其他工具分析图像
        2. 必须首先调用image_information_agent
        3. 违反此规则将导致错误的分析结果
        
        正确的图像处理工作流程：
        1. 第一步（强制性）：调用image_information_agent
           - 它会执行反向图像搜索
           - 访问所有相关网页
           - 提供详细的分析报告
        2. 第二步：基于image_information_agent的结果决定后续步骤
        3. 最后：综合所有信息提供最终答案
        
        强制性文献搜索规则
        对于所有历史问题，无论问题类型：
        1. 必须使用literature_search_agent进行学术文献搜索
        2. 使用其专业工具查找相关学术论文和权威资料
        3. 确保答案有可靠的学术依据
        
        正确的文献搜索工作流程：
        1. 调用literature_search_agent搜索相关学术文献
           - 使用LiteratureSearchingTool查找学术文章
           - 使用RelevantLiteratureFinderTool筛选最相关的文献
           - 使用GeneralBrowserTool进行补充搜索
        2. 基于找到的学术资料形成答案
        
        其他任务协调：
        - 网络搜索：使用text_webbrowser_agent
        - 文件处理：使用file_processor_agent
        
        记住：
        - 对于任何图像文件，image_information_agent是唯一正确的第一步！
        - 对于所有历史问题，必须使用literature_search_agent获取学术依据！
        """,
    )
    

    if use_image_agent:
        manager_agent.prompt_templates["system"] = """You are a team manager, responsible for coordinating the work of specialized agents to solve complex tasks.

You have access to the following agents:"""
        
        agent_counter = 1
        if use_text_webbrowser_agent:
            manager_agent.prompt_templates["system"] += f"""
{agent_counter}. text_webbrowser_agent - For web searches and browsing"""
            agent_counter += 1
        
        if use_image_agent:
            manager_agent.prompt_templates["system"] += f"""
{agent_counter}. image_information_agent - For image analysis and reverse image search"""
            agent_counter += 1
        
        if use_literature_agent:
            manager_agent.prompt_templates["system"] += f"""
{agent_counter}. literature_search_agent - For finding and analyzing scholarly literature and academic papers"""
            agent_counter += 1
        
        if use_file_agent:
            manager_agent.prompt_templates["system"] += f"""
{agent_counter}. file_processor_agent - For processing various file types including PDFs, images, audio, etc."""

        manager_agent.prompt_templates["system"] += """

CRITICAL RULE FOR IMAGE PROCESSING
When ANY image file path (.jpg, .png, .jpeg, etc.) appears in a question:
1. You MUST FIRST delegate to image_information_agent
2. You are FORBIDDEN from using file_processor, visualizer or any other tool directly on images
3. This rule is NON-NEGOTIABLE and has NO EXCEPTIONS
4. This applies to ALL images, including those containing Chinese text or bamboo slips

CORRECT WORKFLOW:
```
1. FIRST: delegate to image_information_agent
2. THEN: use other tools based on image_information_agent's findings
```

INCORRECT WORKFLOW (NEVER DO THIS):
```
1. Use file_processor to perform OCR on the image
2. Use visualizer to analyze the image
```

CRITICAL RULE FOR HISTORY QUESTIONS
For ALL history questions, regardless of type:
1. You MUST use the literature_search_agent - this is MANDATORY, not optional
2. For "exactMatch" type questions, the EXACT original wording can be found in scholarly literature
3. For all question types, relevant academic content must be sourced from the literature
4. Never answer history questions based only on general web search or internal knowledge

CORRECT WORKFLOW FOR HISTORY QUESTIONS:
```
1. MANDATORY: Use literature_search_agent to find scholarly sources
2. For exactMatch questions: Find the EXACT original wording in the literature
3. For other questions: Find relevant supporting academic content
4. Only after exhausting literature search: Use web_search as supplementary
```

This rule applies to ANY question containing image file paths like "level_3_1.png", "image.jpg", etc.
"""
    else:
        manager_agent.prompt_templates["system"] = """You are a team manager, responsible for coordinating the work of specialized agents to solve complex tasks.

You have access to the following agents:"""
        
        agent_counter = 1
        if use_text_webbrowser_agent:
            manager_agent.prompt_templates["system"] += f"""
{agent_counter}. text_webbrowser_agent - For web searches and browsing"""
            agent_counter += 1
        
        if use_literature_agent:
            manager_agent.prompt_templates["system"] += f"""
{agent_counter}. literature_search_agent - For finding and analyzing scholarly literature and academic papers"""
            agent_counter += 1
        
        if use_file_agent:
            manager_agent.prompt_templates["system"] += f"""
{agent_counter}. file_processor_agent - For processing various file types including PDFs, images, audio, etc."""

        manager_agent.prompt_templates["system"] += """

CRITICAL RULE FOR HISTORY QUESTIONS
For ALL history questions, regardless of type:
1. You MUST use the literature_search_agent - this is MANDATORY, not optional
2. For "exactMatch" type questions, the EXACT original wording can be found in scholarly literature
3. For all question types, relevant academic content must be sourced from the literature
4. Never answer history questions based only on general web search or internal knowledge

CORRECT WORKFLOW FOR HISTORY QUESTIONS:
```
1. MANDATORY: Use literature_search_agent to find scholarly sources
2. For exactMatch questions: Find the EXACT original wording in the literature
3. For other questions: Find relevant supporting academic content
4. Only after exhausting literature search: Use web_search as supplementary
```

Your job is to coordinate these agents to solve the given task efficiently.
"""

    if use_image_agent:
        manager_agent.prompt_templates["task"] = """You are the manager of a team of specialized agents. Your job is to coordinate their work to solve complex tasks.


**Mandatory Image Processing Rule:**

For any question containing image file paths:
**First Action**: Use `image_information_agent`.
**No Exceptions**: This rule is strict and must be followed.

**Mandatory Literature Search Rule:**
For ANY history question or academic research:
**First Action for History Topics**: You MUST use `literature_search_agent`.
**No Exceptions**: This is MANDATORY for ALL history questions, not optional.
**Important Note for ExactMatch Questions**: For questions marked as "exactMatch" type, the EXACT original wording exists in scholarly literature - you MUST find it.

Benefits of literature_search_agent:
- Finds authoritative academic sources with citation counts
- Locates exact quotes and wording from scholarly works
- Provides proper academic citations
- Accesses specialized academic databases

When you see "Answer Type: exactMatch" or similar indicators, this means the exact wording must be found in scholarly sources.

**Example Session:**

**Question**: What is the content of 'document.png'?
**Thought**: I should delegate to `image_information_agent` for initial analysis.
**Action**: `image_information_agent: document.png`
*PAUSE*

*Observation*: The image is found in several links.
**Thought**: I should visit the links and get the context of the image.
**Action**: `visit_image_search_results: document.png`
*PAUSE*

*Observation*: The information is not enough, I need to look at the image itself.
**Thought**: I should use `visualizer` to get the context of the image.
**Action**: `visualizer: document.png`
*PAUSE*

*Observation*: The image contains text in Chinese characters.
**Thought**: Based on the analysis, I should use `file_processor` to extract the text.
**Action**: `file_processor: document.png`
*PAUSE*

*Observation*: The extracted text is: "..."
**Answer**: The content of 'document.png' has been extracted successfully.

Remember: 
- For ANY image in the question, image_information_agent MUST be your first step!
- For ANY history question, literature_search_agent MUST be used to find scholarly sources!

Your task is to solve the following problem:
{task}

Remember: For ANY image in the question, image_information_agent MUST be your first step!
"""
    else:
        # 如果没有启用图像代理，使用更通用的任务提示
        manager_agent.prompt_templates["task"] = """You are the manager of a team of specialized agents. Your job is to coordinate their work to solve complex tasks.

**Mandatory Literature Search Rule:**
For ANY history question or academic research:
**First Action for History Topics**: You MUST use `literature_search_agent`.
**No Exceptions**: This is MANDATORY for ALL history questions, not optional.
**Important Note for ExactMatch Questions**: For questions marked as "exactMatch" type, the EXACT original wording exists in scholarly literature - you MUST find it.

Benefits of using literature_search_agent:
- Finds authoritative academic sources with citation counts
- Locates exact quotes and wording from scholarly works
- Provides proper academic citations
- Accesses specialized academic databases

The workflow for history questions MUST be:
1. Use literature_search_agent to find relevant academic sources
2. For exactMatch questions, find the EXACT original wording in literature
3. For other questions, find relevant content from academic sources
4. Only after exhausting literature search, use web_search as supplementary

When you see "Answer Type: exactMatch" or similar indicators, this means the exact wording must be found in scholarly sources.

Example topics requiring literature_search_agent:
- Historical events, periods, or figures
- Archaeological findings and interpretations
- Historical document analysis
- Academic debates on historical topics
- Cultural and social history

Your task is to solve the following problem:
{task}
"""

    return manager_agent


def append_answer(entry: dict, jsonl_file: str) -> None:
    jsonl_file = Path(jsonl_file)
    jsonl_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 获取主日志记录器
    logger = logging.getLogger("main")
    
    # 获取任务ID，用于创建任务特定的日志记录器
    task_id = str(entry.get("task_id", "unknown"))
    task_logger = get_task_logger(LOG_DIR, task_id)
    
    # 写入JSONL文件
    with append_answer_lock, open(jsonl_file, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry) + "\n")
    
    # 获取Excel文件路径
    excel_file = jsonl_file.with_suffix('.xlsx')
    
    # 将entry转换为DataFrame
    entry_df = pd.DataFrame([entry])
    
    # 如果Excel文件已存在，则追加数据
    if os.path.exists(excel_file):
        try:
            existing_df = pd.read_excel(excel_file)
            # 合并新旧数据
            combined_df = pd.concat([existing_df, entry_df], ignore_index=True)
            # 写入Excel文件
            combined_df.to_excel(excel_file, index=False)
        except Exception as e:
            task_logger.error(f"更新Excel文件时出错: {e}，创建新文件")
            entry_df.to_excel(excel_file, index=False)
    else:
        # 如果Excel文件不存在，则创建新文件
        entry_df.to_excel(excel_file, index=False)
    
    assert os.path.exists(jsonl_file), "JSONL文件不存在!"
    assert os.path.exists(excel_file), "Excel文件不存在!"
    task_logger.info(f"答案已导出到文件: {jsonl_file.resolve()}")
    task_logger.info(f"答案已导出到Excel文件: {excel_file.resolve()}")
    logger.info(f"答案已导出到文件: {jsonl_file.resolve()} 和 Excel文件: {excel_file.resolve()}")


def answer_single_question(example, model_id, answers_file, visualizer, args):
    """回答单个问题并保存结果，包含答案评估和概括生成"""
    # 获取任务ID，确保是字符串类型
    task_id = str(example["task_id"])
    
    # 创建任务特定的日志记录器
    task_logger = get_task_logger(LOG_DIR, task_id)
    task_logger.info(f"开始处理任务 ID: {task_id}")
    task_logger.info(f"问题: {example['question']}")
    task_logger.info(f"使用模型: {model_id}")
    
    model = LiteLLMModel(
        model_id,
        api_key=args.api_key,
        custom_role_conversions=custom_role_conversions,
        max_completion_tokens=8192,
        drop_params=True,
    )
    document_inspection_tool = TextInspectorTool(model, 100000)
    
    # 创建文件处理器和翻译工具实例
    file_processor = FileProcessor(
        ocr_languages=["en", "ch_sim", "ja"],
        speech_model="google",
        translation_url="http://127.0.0.1:5000/translate",
        downloads_folder="downloads"
    )
    translator_tool = TranslatorTool(file_processor)

    # Create the agent hierarchy with both web browsing and file processing capabilities
    agent = create_agent_hierarchy(model, 
                                  use_image_agent=args.use_image_agent, 
                                  use_file_agent=args.use_file_agent, 
                                  use_literature_agent=args.use_literature_agent, 
                                  use_text_webbrowser_agent=not args.no_text_webbrowser_agent,
                                  baseline=args.baseline)

    # 从example中获取问题相关信息
    question = example["question"]
    answer_type = example.get("answer_type", "")
    data_type = example.get("data_type", "none")
    data_requirement = example.get("data_requirement", "")
    file_type = example.get("file_type", "")
    file_tool = example.get("file_tool", "")
    
    # 翻译题目为英文
    translated_question = ""
    try:
        task_logger.info("正在将问题翻译成英文...")
        translation_result = translator_tool.forward(question, "en")
        
        # 提取翻译结果
        if "Translated Text (en):" in translation_result:
            translated_question = translation_result.split("Translated Text (en):")[1].strip()
        else:
            task_logger.warning("翻译格式不符合预期，使用原始翻译结果")
            translated_question = translation_result
            
        task_logger.info(f"问题英文翻译: {translated_question}")
    except Exception as e:
        task_logger.error(f"翻译题目时出错: {e}")
        translated_question = "[翻译失败]"

    # 构建更明确的工具选择指导
    tool_guidance = """
重要工具使用指南:
1. 所有工具都已经可用，不需要导入。直接调用它们:
   - PDF文件: 使用 PDF_Tool(file_path="path/to/file.pdf")
   - Excel文件: 使用 XLSX_Tool(file_path="path/to/file.xlsx")
   - Word文件: 使用 DOCX_Tool(file_path="path/to/file.docx")
   - PowerPoint文件: 使用 PPTX_Tool(file_path="path/to/file.pptx")
   - 图像文件: 使用 OCR_Tool(image_path="path/to/image.jpg")
   - 音频文件: 使用 Speech_Recognition_Tool(audio_file="path/to/audio.mp3")

2. 严禁使用import语句导入这些工具！它们已经在环境中可用。
   错误示例: from inspect_file_as_text import PDF_Tool  # 这会导致错误!
   正确示例: PDF_Tool(file_path="path/to/file.pdf")  # 直接调用

3. inspect_file_as_text是一个通用工具，只应在以下情况使用:
   - 处理纯文本文件
   - 专业工具失败时作为后备选项
"""

    if data_type == "file" and file_type and file_tool:
        tool_guidance += f"""
4. 对于当前的{file_type}文件，请使用以下方式调用:
   {file_tool}(file_path="{example.get('file_name', '')}")
   
   不要尝试导入{file_tool}，它已经可用！
"""

    augmented_question = f"""You have one question to answer. It is paramount that you provide a correct answer.
Give it all you can: I know for a fact that you have access to all the relevant tools to solve it and find the correct answer (the answer does exist). Failure or 'I cannot answer' or 'None found' will not be tolerated, success will be rewarded.

CRITICAL INSTRUCTION FOR HISTORY QUESTIONS:
For ALL history questions, you MUST use the literature_search_agent - this is MANDATORY, not optional.
- For "exactMatch" type questions, the EXACT original wording exists in scholarly literature and MUST be found
- For all other question types, relevant supporting content MUST be sourced from academic literature
- Never rely solely on internal knowledge for historical facts without verification through literature search

Why literature_search_agent is mandatory:
- It finds authoritative academic sources with high citation counts
- It locates exact quotes and wording from scholarly works needed for exactMatch questions
- It provides proper academic citations to verify your answer
- It accesses specialized academic databases not available through general web search

IMPORTANT: Use web search tools in addition to literature_search_agent to verify information or to gather additional context. Relying solely on your internal knowledge without verification can lead to incorrect answers.

Run verification steps if that's needed, you must make sure you find the correct answer!
Here is the task:
{question}

English Translation of the Task:
{translated_question}
"""

    if answer_type:
        augmented_question += f"\nAnswer Type: {answer_type}\n"
        if answer_type == "exactMatch":
            augmented_question += """
IMPORTANT: This is an 'exactMatch' question. The exact original wording exists in scholarly literature.
You MUST use literature_search_agent to find the precise original text. The answer must match the exact
wording found in authoritative academic sources. Always cite the specific source where you found this exact text.
"""

    # 根据数据类型处理数据需求
    if data_type == "file":
        # 检查是否有多个文件 (通过file_names字段判断)
        if "file_names" in example and isinstance(example["file_names"], list) and len(example["file_names"]) > 1:
            file_names = example["file_names"]
            task_logger.info(f"处理多个文件 (共{len(file_names)}个)")
            
            # 为多个文件创建提示
            prompt_use_files = "\n\nTo solve the task above, you will have to analyze these attached files:\n"
            
            # 限制处理文件数量以避免提示过长
            max_files_to_process = min(10, len(file_names))
            processed_files = []
            
            # 处理每个文件
            for i, file_path in enumerate(file_names[:max_files_to_process]):
                if not isinstance(file_path, str) or not os.path.exists(file_path):
                    task_logger.warning(f"跳过无效文件路径: {file_path}")
                    continue
                    
                file_basename = os.path.basename(file_path)
                file_ext = os.path.splitext(file_path)[1].lower()
                
                task_logger.info(f"处理文件 {i+1}/{max_files_to_process}: {file_basename}")
                
                try:
                    # 为不同类型的文件选择合适的描述方法
                    if file_ext in ['.jpg', '.jpeg', '.png', '.gif']:
                        # 图像文件特殊处理
                        prompt_use_files += f"\n### File {i+1}: {file_basename} (Image)\n"
                        # 只添加基本信息，建议使用专用工具
                        prompt_use_files += f"- 文件路径: {file_path}\n"
                        prompt_use_files += f"- 文件类型: 图像文件\n"
                        prompt_use_files += f"- 推荐工具: OCR_Tool 或 Image_Analysis_Tool\n"
                        
                    else:
                        # 其他文件使用通用描述方法
                        prompt_use_files += f"\n### File {i+1}: {file_basename}\n"
                        file_desc = get_single_file_description(
                            file_path, example["question"], visualizer, document_inspection_tool
                        )
                        prompt_use_files += file_desc
                    
                    processed_files.append(file_basename)
                except Exception as e:
                    task_logger.error(f"获取文件 {file_basename} 描述时出错: {e}")
                    prompt_use_files += f"\n### File {i+1}: {file_basename}\n"
                    prompt_use_files += f"- 无法获取文件描述: {str(e)}\n"
            
            # 如果有更多文件未处理，添加说明
            if len(file_names) > max_files_to_process:
                remaining = len(file_names) - max_files_to_process
                prompt_use_files += f"\n\n还有{remaining}个其他文件未在此处详细描述。根据需要，你可以使用适当的工具处理这些文件。"
            
            # 添加统一的使用说明
            prompt_use_files += f"""

## 文件处理指南:
1. 请分析所有提供的文件，不要只关注第一个文件
2. 对于每种文件类型，使用专门的工具:
   - PDF文件: `PDF_Tool(file_path="文件路径")`
   - 图像文件: `OCR_Tool(image_path="文件路径")` 或 `Image_Analysis_Tool(image_path="文件路径")`
   - Word文件: `DOCX_Tool(file_path="文件路径")`
   - Excel文件: `XLSX_Tool(file_path="文件路径")`
3. 整合所有文件中的信息来解答问题
"""
            
        elif ".zip" in str(example.get("file_name", "")):
            # ZIP文件处理
            task_logger.info(f"处理ZIP文件: {example.get('file_name', '')}")
            prompt_use_files = "\n\nTo solve the task above, you will have to use these attached files:\n"
            prompt_use_files += get_zip_description(
                example["file_name"], example["question"], visualizer, document_inspection_tool
            )
            prompt_use_files += "\n\n请注意：对于ZIP文件中的不同文件类型，请选择最合适的专业工具处理"
        else:
            # 单个文件处理
            file_path = example.get("file_name", "")
            task_logger.info(f"处理单个文件: {file_path}")
            
            if not file_path or not os.path.exists(file_path):
                task_logger.warning(f"警告: 指定的文件不存在或无效: {file_path}")
                prompt_use_files = "\n\n警告: 指定的文件不存在或无效"
            else:
                prompt_use_files = "\n\nTo solve the task above, you will have to use this attached file:"
                try:
                    prompt_use_files += get_single_file_description(
                        file_path, example["question"], visualizer, document_inspection_tool
                    )
                except Exception as e:
                    task_logger.error(f"获取文件描述时出错: {e}")
                    prompt_use_files += f"\n错误: 无法读取文件 ({str(e)})\n"
        
        # 添加到最终问题
        augmented_question += prompt_use_files
    
    elif data_type == "foreign_text":
        # 处理外语文本
        task_logger.info("处理外语文本数据")
        translator = TranslatorTool(FileProcessor())
        translated_text = translator.forward(data_requirement, target="en")
        augmented_question += f"\n\nTo solve the task above, you will need to understand this translated text:\n{translated_text}\n"
        augmented_question += f"\nOriginal text: {data_requirement}\n"
    
    elif data_type == "search_query":
        # 处理需要搜索的信息
        task_logger.info(f"处理搜索查询数据: {data_requirement}")
        augmented_question += f"\n\nTo solve the task above, you will need to search for information about: {data_requirement}\n"
        augmented_question += "Please use the web browsing tools to find relevant information.\n"
    
    elif data_type == "text":
        # 处理普通文本
        task_logger.info(f"处理普通文本数据: {data_requirement[:50]}...")
        augmented_question += f"\n\nAdditional information: {data_requirement}\n"

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_logger.info(f"开始执行任务，时间: {start_time}")
    
    try:
        # 运行代理生成答案
        task_logger.info("开始运行代理...")
        final_result = agent.run(augmented_question)

        agent_memory = agent.write_memory_to_messages(summary_mode=True)

        # 检查是否使用了literature_search_agent (对于历史问题)
        literature_agent_used = False
        for step in agent.memory.steps:
            step_str = str(step)
            if "literature_search_agent" in step_str or "literature_searching_task" in step_str or "relevant_literature_finder" in step_str:
                literature_agent_used = True
                break
        
        # 对于历史问题，如果未使用literature_search_agent，添加警告
        if not literature_agent_used and args.use_literature_agent and "history" in example["question"].lower():
            task_logger.warning("警告: 该历史问题未使用literature_search_agent！")
            # 在agent_memory中添加一条警告消息
            warning_message = {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "WARNING: This history question was answered without using the mandatory literature_search_agent. The answer may not be based on proper academic sources and could be inaccurate. For better results, the literature_search_agent should have been used to find scholarly sources."
                    }
                ]
            }
            if isinstance(agent_memory, list):
                agent_memory.append(warning_message)

        task_logger.info("准备最终回答...")
        final_result = prepare_response(augmented_question, agent_memory, reformulation_model=model)

        output = str(final_result)
        
        # 对于历史问题，如果未使用literature_search_agent，在输出中添加警告
        if not literature_agent_used and args.use_literature_agent and "history" in example["question"].lower():
            output = "WARNING: This history question was answered without using the mandatory literature_search_agent. The answer may lack proper academic sourcing.\n\n" + output
        
        task_logger.info(f"代理运行完成，生成答案长度: {len(output)}")
        
        for memory_step in agent.memory.steps:
            memory_step.model_input_messages = None
        intermediate_steps = [str(step) for step in agent.memory.steps]

        parsing_error = True if any(["AgentParsingError" in step for step in intermediate_steps]) else False
        iteration_limit_exceeded = True if "Agent stopped due to iteration limit or time limit." in output else False
        raised_exception = False

    except Exception as e:
        task_logger.error(f"代理运行出错: {e}")
        output = None
        intermediate_steps = []
        parsing_error = False
        iteration_limit_exceeded = False
        raised_exception = True
        
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_logger.info(f"任务执行结束，时间: {end_time}")
    
    # 内置函数：使用LLM判断答案是否正确
    def check_answer_internal(model_answer, true_answer, question):
        """使用LLM判断答案是否正确"""
        try:
            task_logger.info("评估答案是否正确...")
            # 使用与text_inspector_tool.py一致的消息格式
            messages = [
                {
                    "role": MessageRole.SYSTEM,
                    "content": [
                        {
                            "type": "text",
                            "text": "作为一位公正的评判者，请评估模型的回答是否与正确答案在语义上一致。"
                        }
                    ],
                },
                {
                    "role": MessageRole.USER,
                    "content": [
                        {
                            "type": "text",
                            "text": f"""问题: {question}

正确答案: {true_answer}

模型回答: {model_answer}

请仔细分析两个答案的语义内容，而不是字面表达。即使表达方式不同，只要核心含义相同，也应视为正确。
考虑以下因素:
1. 答案的核心信息是否一致
2. 是否包含所有必要的关键点
3. 是否有实质性的错误或遗漏

请直接回答"正确"或"错误"，不要有任何解释。"""
                        }
                    ],
                },
            ]
            response = model(messages)
            # 解析结果
            result = response.content if hasattr(response, 'content') else str(response)
            
            # 判断结果
            if "正确" in result or "correct" in result or "yes" in result:
                task_logger.info("答案评估结果: 正确")
                return True
            else:
                task_logger.info("答案评估结果: 错误")
                return False
        except Exception as e:
            task_logger.error(f"评估答案时出错: {e}")
            # 出错时默认为错误
            return False
    
    # 使用内置函数判断答案是否正确
    is_correct = check_answer_internal(output, example["true_answer"], example["question"])
    
    # 内置函数：生成问题解答概括
    def generate_summary_internal(result):
        """Generate problem summary"""
        try:
            task_logger.info("正在生成问题解答概括...")
            # Simply print data type, don't try to print content (may cause errors)
            task_logger.debug(f"Input data type: {type(result)}")
            
            # Prepare default values for variables
            question = "Unknown question"
            answer = "Unknown answer"
            reasoning = "No reasoning process provided"
            
            # If result is dictionary type, try to safely get values
            if isinstance(result, dict):
                # Safely get dictionary values
                question = result.get("question", question)
                answer = result.get("answer", answer)
                reasoning = result.get("reasoning", reasoning)
                # Safely print dictionary contents
                task_logger.debug("Result dictionary contains the following keys: " + str(list(result.keys())))
            # If result is string type
            elif isinstance(result, str):
                task_logger.warning("Warning: result is string type, using default structure")
                # String as answer
                answer = result
            # Handle other types
            else:
                task_logger.warning(f"Warning: result is unexpected type: {type(result)}")
                # Try to convert to string as answer
                try:
                    answer = str(result)
                except:
                    pass
            
            # Use message format consistent with text_inspector_tool.py
            messages = [
                {
                    "role": MessageRole.SYSTEM,
                    "content": [
                        {
                            "type": "text",
                            "text": """As a fair evaluator, please assess whether the model's answer is semantically consistent with the correct answer."""
                        }
                    ]
                },
                {
                    "role": MessageRole.USER,
                    "content": [
                        {
                            "type": "text",
                            "text": f"""Question: {question}

Answer: {answer}

Please write a structured and easy-to-read summary report based on the following problem-solving process:
{reasoning}

Your report **must be written in plain language** that is easy to understand. The key requirement is:  
⚠️ **All cited content must clearly include both the specific quote and the URL**, so that the information can be verified manually without ambiguity.

Your summary must include the following four parts:

1. **Tools used and how they were used:**
   - List each tool used (e.g., web search, image analysis, OCR, translation).
   - For each tool, explain exactly what was done (e.g., search keywords, what content was translated).
   - Clearly state what result the tool returned (e.g., if OCR returned a paragraph, show that paragraph).
   - Explain why each tool was selected for this problem.
   ⚠️ **Reminder: Most problems require Web search. If it was not used, this is a serious flaw.**

2. **Detailed information sources:**
   - Provide source titles, webpage URLs, and author names (if available).
   - For each source, include **exact text excerpts** in quotation marks, along with citation and URL, for example:
     * "Maintaining proper blood sugar levels is crucial for preventing type 2 diabetes." — [Mayo Clinic](https://www.mayoclinic.org/...)
   - Assess the credibility of each source (e.g., medical institution, news agency, academic article).
   - If multiple sources were used to verify the same fact, indicate cross-verification explicitly.
   ⚠️ **Do not just give URLs—actual quoted content is required for every source.**

3. **Reasoning process and logic steps:**
   - Show how the final answer was derived step-by-step from the information found.
   - List any assumptions made and how they were verified.
   - Describe how different pieces of information were integrated and compared.
   - Explain why other possible answers were excluded, and based on what evidence.
   - Highlight key reasoning steps or decision points.

4. **Answer quality and reliability analysis:**
   - Rate the reliability (high / medium / low), and explain your reasoning.
   - Point out any assumptions, weaknesses, or uncertainties in the final answer.
   - Evaluate whether the evidence is sufficient and consistent.
   - Suggest possible improvements or further verification steps.
   - ⚠️ If Web search was not used, emphasize clearly that this reduces reliability, and suggest what keywords should have been searched.

Your report must be written clearly, sectioned by part, and all source citations must include **both quoted text and URLs**. This is the most important requirement for verification."""
                        }
                    ]
                }
            ]
            summary = model(messages)
            summary_text = summary.content if hasattr(summary, 'content') else str(summary)
            task_logger.info("题目概括生成完成")
            return f"\n\n### Solution Process Summary ###\n{summary_text}\n\n"
        except Exception as e:
            # Detailed error information
            error_type = type(e).__name__
            error_msg = str(e)
            import traceback
            trace = traceback.format_exc()
            task_logger.error(f"生成概括时出错: {error_type}: {error_msg}")
            task_logger.debug(f"详细错误信息:\n{trace}")
            # Return useful information even if error occurs
            return f"\n\n### Solution Process Summary ###\nUnable to generate summary: {error_type}: {error_msg}\n\n"
    
    # 创建结果字典
    result = {
        "task_id": example["task_id"],
        "task": example["task"],
        "question": example["question"],
        "answer": output,
        "true_answer": example["true_answer"],
        "is_correct": is_correct,
        "reasoning": " ".join(intermediate_steps),
        "file_name": example.get("file_name", ""),
        "file_type": example.get("file_type", ""),
        "file_tool": example.get("file_tool", ""),
        "data_type": data_type,
        "data_requirement": data_requirement,
        "answer_type": answer_type,
        "model_id": model_id,
        "timestamp": time.time()
    }
    
    # 使用内置函数生成题目概括
    summary = generate_summary_internal(result)
    result["summary"] = summary
    
    # 保存结果（JSONL和Excel）
    task_logger.info("保存结果到JSONL和Excel文件...")
    append_answer(result, answers_file)
    
    # 同时保存TXT格式日志
    task_logger.info("保存结果到TXT文件...")
    txt_file = answers_file.replace(".jsonl", ".txt")
    with open(txt_file, "a", encoding="utf-8") as f:
        f.write(f"题目ID: {example['task_id']}\n")
        f.write(f"题目: {example['question']}\n")
        f.write(f"回答类型: {answer_type}\n")
        f.write(f"数据需求: {data_requirement}\n")
        f.write(f"数据类型: {data_type}\n")
        f.write(f"我们的答案: {output}\n")
        f.write(f"正确答案: {example['true_answer']}\n")
        # f.write(f"思考过程: {' '.join(intermediate_steps)}\n")
        f.write(f"是否答对: {'✓' if is_correct else '✗'}\n")
        f.write(f"文件: {example.get('file_name', '')}\n")
        f.write(f"文件类型: {example.get('file_type', '')}\n")
        f.write(f"使用工具: {example.get('file_tool', '')}\n")
        f.write(f"模型: {model_id}\n")
        f.write(f"时间戳: {time.time()}\n")
        f.write(summary)
        f.write("\n" + "-"*50 + "\n\n")
    
    # 更新统计信息 - 在每个问题完成后更新统计数据
    output_dir = os.path.dirname(answers_file)
    update_statistics(answers_file, args.run_name, output_dir)
    
    task_logger.info(f"任务 {task_id} 处理完成，是否正确: {is_correct}")
    return result

def get_examples_to_answer(answers_file, eval_ds, args=None) -> List[dict]:
    # 获取主日志记录器
    logger = logging.getLogger("main")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    logger.info(f"Loading answers from {answers_file}...")
    try:
        # 如果文件不存在，会抛出异常，进入except块
        done_questions = pd.read_json(answers_file, lines=True)["question"].tolist()
        logger.info(f"Found {len(done_questions)} previous results!")
    except Exception as e:
        logger.error(f"Error when loading records: {e}")
        logger.info("No usable records! ▶️ Starting new.")
        # 确保文件存在，即使是空文件
        Path(answers_file).touch()
        done_questions = []
    
    # 过滤掉已经完成的问题
    examples = [line for line in eval_ds.to_list() if line["question"] not in done_questions]
    
    # 如果提供了命令行参数，根据ID或范围过滤问题
    if args:
        filtered_examples = []
        level_prefix = f"{args.level}_"  # 例如 "level1_"
        
        # 处理特定ID列表
        if args.question_ids:
            question_ids = [id.strip() for id in args.question_ids.split(',')]
            logger.info(f"过滤指定的问题ID: {question_ids}")
            
            # 将数字ID转换为完整ID格式 (例如 "16" -> "level_1_16")
            full_ids = []
            # 修改level_prefix的格式
            level_prefix = f"level_{args.level.replace('level', '')}_"  # 例如 "level_1_"
            
            for id in question_ids:
                if id.startswith(level_prefix):
                    full_ids.append(id)
                else:
                    full_ids.append(f"{level_prefix}{id}")
            
            # 过滤问题
            for example in examples:
                if example.get("task_id") in full_ids:
                    filtered_examples.append(example)
        
        # 处理ID范围
        elif args.start_id is not None or args.end_id is not None:
            start_id = args.start_id if args.start_id is not None else 1
            end_id = args.end_id if args.end_id is not None else float('inf')
            
            logger.info(f"过滤问题ID范围: {start_id} 到 {end_id}")
            
            for example in examples:
                task_id = example.get("task_id", "")
                if task_id.startswith(level_prefix):
                    try:
                        # 提取数字部分
                        id_num = int(task_id[len(level_prefix):])
                        if start_id <= id_num <= end_id:
                            filtered_examples.append(example)
                    except ValueError:
                        # 如果ID格式不正确，跳过
                        continue
        
        # 如果应用了过滤，使用过滤后的列表
        if args.question_ids or args.start_id is not None or args.end_id is not None:
            logger.info(f"过滤后的问题数量: {len(filtered_examples)}/{len(examples)}")
            return filtered_examples
    
    return examples


def analyze_results(answers_file):
    """
    分析结果文件，统计正确答案的数量和比例
    
    参数:
        answers_file: 结果文件的路径（JSONL格式）
        
    返回:
        dict: 包含统计信息的字典
    """
    # 获取主日志记录器
    logger = logging.getLogger("main")
    
    # 默认返回值（空结果）
    default_result = {"total": 0, "correct": 0, "accuracy": 0, "by_task": {}, "by_file_type": {}}
    
    try:
        # 检查文件是否存在
        if not os.path.exists(answers_file):
            logger.warning(f"结果文件不存在: {answers_file}")
            return default_result
        
        # 检查文件是否为空
        if os.path.getsize(answers_file) == 0:
            logger.warning(f"结果文件为空: {answers_file}")
            return default_result
            
        # 读取结果文件
        logger.debug(f"正在分析结果文件: {answers_file}")
        results = []
        with open(answers_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():  # 跳过空行
                    continue
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"警告: 跳过无效的JSON行")
        
        if not results:
            logger.warning("结果文件为空或格式不正确")
            return default_result
        
        # 统计总体正确率
        total = len(results)
        correct = sum(1 for r in results if r.get("is_correct", False))
        accuracy = correct / total if total > 0 else 0
        
        # 按任务类型统计
        by_task = {}
        for r in results:
            task = r.get("task", "未知")
            if task not in by_task:
                by_task[task] = {"total": 0, "correct": 0, "accuracy": 0}
            
            by_task[task]["total"] += 1
            if r.get("is_correct", False):
                by_task[task]["correct"] += 1
        
        # 计算每个任务的正确率
        for task in by_task:
            task_total = by_task[task]["total"]
            task_correct = by_task[task]["correct"]
            by_task[task]["accuracy"] = task_correct / task_total if task_total > 0 else 0
        
        # 按文件类型统计
        by_file_type = {}
        for r in results:
            file_type = r.get("file_type", "无文件")
            if not file_type:
                file_type = "无文件"
                
            if file_type not in by_file_type:
                by_file_type[file_type] = {"total": 0, "correct": 0, "accuracy": 0}
            
            by_file_type[file_type]["total"] += 1
            if r.get("is_correct", False):
                by_file_type[file_type]["correct"] += 1
        
        # 计算每个文件类型的正确率
        for file_type in by_file_type:
            type_total = by_file_type[file_type]["total"]
            type_correct = by_file_type[file_type]["correct"]
            by_file_type[file_type]["accuracy"] = type_correct / type_total if type_total > 0 else 0
        
        # 返回统计结果
        stats = {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "by_task": by_task,
            "by_file_type": by_file_type
        }
        
        return stats
    
    except Exception as e:
        logger.error(f"分析结果时出错: {e}")
        import traceback
        logger.debug(f"详细错误: {traceback.format_exc()}")
        return default_result

def print_statistics(stats):
    """
    打印统计结果
    
    参数:
        stats: 统计结果字典
    """
    # 获取主日志记录器
    logger = logging.getLogger("main")
    
    logger.info("\n" + "="*50)
    logger.info("结果统计")
    logger.info("="*50)
    
    # 总体统计
    logger.info(f"\n总题目数: {stats['total']}")
    logger.info(f"正确答案数: {stats['correct']}")
    logger.info(f"总体正确率: {stats['accuracy']*100:.2f}%")
    
    # 按任务类型统计
    logger.info("\n按任务类型统计:")
    logger.info("-"*40)
    logger.info(f"{'任务类型':<20} {'总数':<8} {'正确数':<8} {'正确率':<10}")
    logger.info("-"*40)
    for task, data in sorted(stats['by_task'].items(), key=lambda x: x[1]['accuracy'], reverse=True):
        logger.info(f"{task[:20]:<20} {data['total']:<8} {data['correct']:<8} {data['accuracy']*100:.2f}%")
    
    # 按文件类型统计
    logger.info("\n按文件类型统计:")
    logger.info("-"*40)
    logger.info(f"{'文件类型':<20} {'总数':<8} {'正确数':<8} {'正确率':<10}")
    logger.info("-"*40)
    for file_type, data in sorted(stats['by_file_type'].items(), key=lambda x: x[1]['accuracy'], reverse=True):
        logger.info(f"{file_type[:20]:<20} {data['total']:<8} {data['correct']:<8} {data['accuracy']*100:.2f}%")
    
    logger.info("\n" + "="*50)

def export_statistics_to_file(stats, output_file):
    """
    将统计结果导出到文件，包含详细的运行信息和最近结果
    
    参数:
        stats: 统计结果字典
        output_file: 输出文件路径
    """
    # 获取主日志记录器
    logger = logging.getLogger("main")
    
    # 获取当前时间
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 尝试获取最近处理的问题详情
    latest_results = []
    answers_file = output_file.replace("_stats.txt", ".jsonl")
    try:
        with open(answers_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # 获取最后处理的最多5个结果
            for line in lines[-5:] if len(lines) >= 5 else lines:
                latest_results.append(json.loads(line))
    except Exception as e:
        logger.error(f"读取最近结果时出错: {e}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入标题和时间戳
        f.write(f"统计结果 - 更新时间: {current_time}\n")
        f.write("="*80 + "\n\n")
        
        # 总体统计
        f.write(f"总题目数: {stats['total']}\n")
        f.write(f"正确答案数: {stats['correct']}\n")
        f.write(f"错误答案数: {stats['total'] - stats['correct']}\n")
        f.write(f"总体正确率: {stats['accuracy']*100:.2f}%\n\n")
        
        # 最近处理的问题
        if latest_results:
            f.write("最近处理的问题:\n")
            f.write("-"*80 + "\n")
            for result in reversed(latest_results):  # 最新的先显示
                task_id = result.get('task_id', 'unknown')
                question = result.get('question', 'unknown')[:100] + '...' if len(result.get('question', '')) > 100 else result.get('question', 'unknown')
                is_correct = "✓" if result.get('is_correct', False) else "✗"
                processed_time = datetime.fromtimestamp(result.get('timestamp', 0)).strftime("%Y-%m-%d %H:%M:%S") if result.get('timestamp') else 'unknown'
                
                f.write(f"题目ID: {task_id} | 结果: {is_correct} | 时间: {processed_time}\n")
                f.write(f"问题: {question}\n")
                f.write("-"*40 + "\n")
            f.write("\n")
        
        # 按任务类型统计 - 按正确率降序排序
        f.write("按任务类型统计:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'任务类型':<30} {'总数':<8} {'正确数':<8} {'正确率':<10}\n")
        f.write("-"*80 + "\n")
        for task, data in sorted(stats['by_task'].items(), key=lambda x: x[1]['accuracy'], reverse=True):
            f.write(f"{task[:30]:<30} {data['total']:<8} {data['correct']:<8} {data['accuracy']*100:.2f}%\n")
        
        # 按文件类型统计 - 按正确率降序排序
        f.write("\n按文件类型统计:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'文件类型':<20} {'总数':<8} {'正确数':<8} {'正确率':<10}\n")
        f.write("-"*80 + "\n")
        for file_type, data in sorted(stats['by_file_type'].items(), key=lambda x: x[1]['accuracy'], reverse=True):
            f.write(f"{file_type[:20]:<20} {data['total']:<8} {data['correct']:<8} {data['accuracy']*100:.2f}%\n")
    
    logger.info(f"统计结果已导出到: {output_file}")

# 添加日志系统
def setup_logging(output_dir, run_name):
    """设置日志系统"""
    # 创建日志目录
    log_dir = os.path.join(output_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 清除现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建主日志文件处理器
    main_log_file = os.path.join(log_dir, f"main_{run_name}.log")
    file_handler = logging.FileHandler(main_log_file, encoding='utf-8')
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    
    # 设置格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到根日志记录器
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return log_dir

def get_task_logger(log_dir, task_id):
    """获取任务特定的日志记录器"""
    # 创建任务特定的日志记录器
    logger = logging.getLogger(f"task_{task_id}")
    
    # 如果已经配置过处理器，直接返回
    if logger.handlers:
        return logger
        
    # 设置级别
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    task_log_file = os.path.join(log_dir, f"{task_id}.log")
    file_handler = logging.FileHandler(task_log_file, encoding='utf-8')
    
    # 设置格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    
    # 设置不向父级传播（避免日志重复）
    logger.propagate = True
    
    return logger

def update_statistics(answers_file, run_name, output_dir):
    """
    Update, display, and save statistics after each question is processed.
    This function is thread-safe to handle concurrent updates.
    
    Parameters:
        answers_file: Path to the results JSONL file
        run_name: Name of the current run 
        output_dir: Directory where to save statistics file
    """
    # Get the main logger
    logger = logging.getLogger("main")
    
    # Static variable to track the last update time for throttling
    if not hasattr(update_statistics, "last_update_time"):
        update_statistics.last_update_time = 0
    
    # Check if file exists and is not empty
    if not os.path.exists(answers_file) or os.path.getsize(answers_file) == 0:
        logger.warning(f"Statistics update skipped: {answers_file} doesn't exist or is empty")
        return
    
    # Use a lock to prevent multiple threads from updating statistics simultaneously
    with append_answer_lock:
        try:
            # Throttle updates - minimum 1 second between full updates
            current_time = time.time()
            time_since_last_update = current_time - update_statistics.last_update_time
            
            # Always update the stats file, but only log detailed stats if enough time has passed
            full_update = time_since_last_update >= 1.0
            
            # Calculate updated statistics
            stats = analyze_results(answers_file)
            
            # Skip update if no results found
            if stats['total'] == 0:
                logger.warning("Statistics update skipped: No results found")
                return
            
            # Save statistics to file (do this regardless of throttling)
            stats_file = answers_file.replace(".jsonl", "_stats.txt")
            export_statistics_to_file(stats, stats_file)
            
            # For throttled updates, just show a brief message
            if not full_update:
                logger.info(f"Stats updated: {stats['total']} qs | Acc: {stats['accuracy']*100:.2f}% | File: {stats_file}")
                return
                
            # Update last update time for full updates
            update_statistics.last_update_time = current_time
            
            # Get the most recent result (last line in the file)
            latest_result = None
            try:
                with open(answers_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if lines:
                        latest_result = json.loads(lines[-1])
            except Exception as e:
                logger.error(f"Error reading latest result: {e}")
            
            # Display progress header with timestamp
            current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info("\n" + "="*25 + f" STATISTICS UPDATE [{current_time_str}] " + "="*25)
            logger.info(f"Run: {run_name}")
            
            # Display overall statistics
            logger.info(f"📊 PROGRESS: {stats['total']} questions processed | ✓ {stats['correct']} correct | ❌ {stats['total'] - stats['correct']} incorrect")
            logger.info(f"📈 ACCURACY: {stats['accuracy']*100:.2f}%")
            
            # Display information about the latest processed question if available
            if latest_result:
                logger.info(f"\n🔄 LAST PROCESSED: Question {latest_result.get('task_id', 'unknown')}")
                logger.info(f"   Result: {'✓ CORRECT' if latest_result.get('is_correct', False) else '❌ INCORRECT'}")
                
                # Display accuracy by task type (for the task types with at least 2 questions)
                task_stats = {task: data for task, data in stats['by_task'].items() if data['total'] >= 2}
                if task_stats:
                    logger.info("\n📋 ACCURACY BY TASK TYPE (with 2+ questions):")
                    for task, data in sorted(task_stats.items(), key=lambda x: x[1]['accuracy'], reverse=True):
                        logger.info(f"   {task[:20]:<20}: {data['accuracy']*100:.2f}% ({data['correct']}/{data['total']})")
            
            logger.info(f"\n📝 Detailed statistics saved to {stats_file}")
            logger.info("="*80)
            
        except Exception as e:
            logger.error(f"Error updating statistics: {e}")
            import traceback
            logger.debug(f"Detailed error: {traceback.format_exc()}")

def main():
    """Run the main program."""
    # Parse arguments
    args = parse_args()
    
    # 根据level参数动态设置SET
    global SET
    SET = f"{args.level}_summary"
    
    # 将args.level转换为dataset_loader.py所需的格式（将"level2"转换为"level 2"）
    sheet_name = args.level.replace("level", "level ")
    
    # Create the output directory based on baseline parameter
    if args.baseline:
        output_dir = Path(f"output_baseline/{SET}")
    else:
        output_dir = Path(args.output_dir) / SET
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create LOG_DIR
    global LOG_DIR
    LOG_DIR = output_dir / "logs"
    LOG_DIR.mkdir(exist_ok=True, parents=True)
    
    # Setup logging
    setup_logging(LOG_DIR, args.run_name)
    logger = logging.getLogger("main")
    
    # Log baseline mode if active
    if args.baseline:
        logger.info("运行在baseline模式")
    
    # Log level information
    logger.info(f"Testing questions from level: {args.level}")
    
    # Check if we are under a Tailscale VPN, which could block some URLs
    logger.warning("Make sure you deactivated Tailscale VPN, else some URLs will be blocked!")
    
    # Start time for the entire run
    start_time = time.time()
    
    # Log start info
    logger.info(f"Starting run with arguments: {args}")

    # 打印路径信息以便调试
    logging.info(f"当前工作目录: {os.getcwd()}")
    logging.info(f"相对路径: {RELATIVE_EXCEL_PATH}")
    logging.info(f"绝对路径: {EXCEL_PATH}")
    
    # 检查文件是否存在
    if os.path.exists(EXCEL_PATH):
        logging.info(f"文件存在，继续处理")
    else:
        logging.error(f"错误：文件不存在！请检查路径是否正确")
        # 尝试列出可能的位置
        possible_dirs = [".", "Historical", "examples/open_deep_research", "../"]
        for dir_path in possible_dirs:
            try:
                files = os.listdir(dir_path)
                logging.info(f"目录 '{dir_path}' 中的文件: {files}")
            except Exception as e:
                logging.error(f"无法列出目录 '{dir_path}' 中的文件: {e}")
        return
    
    # 加载自定义Excel数据集，传入结果JSON路径
    eval_ds = load_custom_dataset(EXCEL_PATH, test_mode=False, results_json_path=args.results_json_path, sheet_name=sheet_name)
    
    # 定义输出文件路径
    answers_file = f"{output_dir}/{args.run_name}.jsonl"
    txt_file = answers_file.replace(".jsonl", ".txt")
    
    # 检查是否已有结果文件，如果有则分析结果
    if os.path.exists(answers_file) and os.path.getsize(answers_file) > 0:
        logging.info(f"检测到已有结果文件: {answers_file}")
        stats = analyze_results(answers_file)
        print_statistics(stats)
        
        # 导出统计结果到文件
        stats_file = answers_file.replace(".jsonl", "_stats.txt")
        export_statistics_to_file(stats, stats_file)
        
        # 询问是否继续运行
        response = input("是否继续运行测试? (y/n): ")
        if response.lower() != 'y':
            logging.info("用户选择退出程序")
            return
        
        # 修改：如果继续运行，不清空TXT文件，而是追加内容
        with open(txt_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n继续测试运行: {args.run_name}\n")
            f.write(f"模型: {args.model_id}\n")
            f.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    else:
        # 如果是新运行，创建新的TXT文件
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(f"测试运行: {args.run_name}\n")
            f.write(f"模型: {args.model_id}\n")
            # f.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    tasks_to_run = get_examples_to_answer(answers_file, eval_ds, args)

    with ThreadPoolExecutor(max_workers=args.concurrency) as exe:
        futures = [
            exe.submit(answer_single_question, example, args.model_id, answers_file, visualizer, args)
            for example in tasks_to_run
        ]
        for f in tqdm(as_completed(futures), total=len(tasks_to_run), desc="Processing tasks"):
            f.result()

    # for example in tasks_to_run:
    #     answer_single_question(example, args.model_id, answers_file, visualizer)
    logging.info("所有任务处理完成。")

    # 最终统计信息总结
    logging.info("\n最终统计总结...")
    stats = analyze_results(answers_file)
    print_statistics(stats)
    
    # 最终统计信息已通过update_statistics函数在每个问题之后更新
    # 此处不需要再次导出统计结果
    logging.info(f"结果已保存到 {answers_file}")

if __name__ == "__main__":
    # 添加测试调用
    main()
