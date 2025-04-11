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
from scripts.reverse_image import GoogleLensSearchTool
from scripts.visual_qa import visualizer
from huggingface_hub import login

from smolagents import (
    CodeAgent,
    GoogleSearchTool,
    LiteLLMModel,
    ToolCallingAgent,
)

from scripts.translator import TranslatorTool
# from scripts.speech_recognition import SpeechRecognitionTool
from scripts.ocr import OCRTool

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

BROWSER_CONFIG_IMAGE = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "image_downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

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

    print("🚩 Step J: creating CodeAgent (manager)")
    manager_agent = CodeAgent(
        model=model,
        tools=[visualizer, TextInspectorTool(model, 100000)],
        max_steps=12,
        verbosity_level=2,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        planning_interval=4,
        managed_agents=[text_webbrowser_agent, image_information_agent, ocr_agent, translator_agent],
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

# import argparse
# import os
# import threading
# import sys
# import streamlit as st

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

# from dotenv import load_dotenv
# from scripts.text_inspector_tool import TextInspectorTool
# from scripts.text_web_browser import (
#     ArchiveSearchTool,
#     FinderTool,
#     FindNextTool,
#     PageDownTool,
#     PageUpTool,
#     SimpleTextBrowser,
#     VisitTool,
# )
# from scripts.visual_qa import visualizer
# from huggingface_hub import login

# from smolagents import (
#     CodeAgent,
#     GoogleSearchTool,
#     LiteLLMModel,
#     ToolCallingAgent,
# )

# AUTHORIZED_IMPORTS = [
#     "requests", "zipfile", "os", "pandas", "numpy", "sympy", "json", "bs4",
#     "pubchempy", "xml", "yahoo_finance", "Bio", "sklearn", "scipy", "pydub",
#     "io", "PIL", "chess", "PyPDF2", "pptx", "torch", "datetime", "fractions", "csv"
# ]

# load_dotenv(override=True)

# append_answer_lock = threading.Lock()

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("question", type=str, help="e.g., 'How many albums did Mercedes Sosa release before 2007?'")
#     parser.add_argument("--model-id", type=str, default="gpt-4o")
#     return parser.parse_args()

# custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

# user_agent = (
#     "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
#     "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
# )

# BROWSER_CONFIG = {
#     "viewport_size": 1024 * 5,
#     "downloads_folder": "downloads_folder",
#     "request_kwargs": {
#         "headers": {"User-Agent": user_agent},
#         "timeout": 300,
#     },
#     "serpapi_key": os.getenv("SERPAPI_API_KEY"),
# }

# def create_agent(model_id="o1"):
#     print("🚩 Step A: calling huggingface login")
#     from huggingface_hub import login
#     hf_token = st.secrets.get("HF_TOKEN")
#     if not hf_token:
#         raise ValueError("❌ HF_TOKEN is missing in environment!")
#     login(hf_token)

#     print("🚩 Step B: making downloads folder")
#     os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)

#     print("🚩 Step C: selecting model token limits")
#     if model_id == "gpt-3.5-turbo":
#         max_tokens = 4096
#         tool_max_tokens = 4096
#     elif model_id == "o1":
#         max_tokens = 8192
#         tool_max_tokens = 8192
#     elif model_id == "gpt-4o":
#         max_tokens = 16000
#         tool_max_tokens = 16000
#     else:
#         max_tokens = 4096
#         tool_max_tokens = 4096

#     print("🚩 Step D: initializing main model")
#     model = LiteLLMModel(
#         model_id,
#         custom_role_conversions=custom_role_conversions,
#         max_completion_tokens=max_tokens,
#     )

#     print("🚩 Step E: initializing tool model")
#     tool_model = LiteLLMModel(
#         "gpt-4o",
#         custom_role_conversions=custom_role_conversions,
#         max_completion_tokens=tool_max_tokens,
#     )

#     print("🚩 Step F: initializing browser")
#     browser = SimpleTextBrowser(**BROWSER_CONFIG)

#     print("🚩 Step G: creating web tools")
#     WEB_TOOLS = [
#         GoogleSearchTool(provider="serpapi"),
#         VisitTool(browser),
#         PageUpTool(browser),
#         PageDownTool(browser),
#         FinderTool(browser),
#         FindNextTool(browser),
#         ArchiveSearchTool(browser),
#         TextInspectorTool(model, 100000),
#     ]

#     print("🚩 Step H: creating ToolCallingAgent")
#     text_webbrowser_agent = ToolCallingAgent(
#         model=tool_model,
#         tools=WEB_TOOLS,
#         max_steps=20,
#         verbosity_level=2,
#         planning_interval=4,
#         name="search_agent",
#         description="Search agent with browser tools",
#         provide_run_summary=True,
#     )

#     print("🚩 Step I: setting prompt extensions")
#     text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += (
#         "You can navigate to .txt files. Use 'inspect_file_as_text' for PDF/Youtube."
#     )

#     print("🚩 Step J: creating CodeAgent (manager)")
#     manager_agent = CodeAgent(
#         model=model,
#         tools=[visualizer, TextInspectorTool(model, 100000)],
#         max_steps=12,
#         verbosity_level=2,
#         additional_authorized_imports=AUTHORIZED_IMPORTS,
#         planning_interval=4,
#         managed_agents=[text_webbrowser_agent],
#     )

#     print("✅ Step K: returning agent")
#     return manager_agent


# def main():
#     args = parse_args()
#     agent = create_agent(model_id=args.model_id)
#     answer = agent.run(args.question)
#     print(f"Got this answer: {answer}")

# if __name__ == "__main__":
#     main()