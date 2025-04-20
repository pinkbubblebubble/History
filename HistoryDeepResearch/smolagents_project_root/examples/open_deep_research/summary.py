import streamlit as st

from smolagents.models import MessageRole
from smolagents import (
    CodeAgent,
    GoogleSearchTool,
    LiteLLMModel,
    ToolCallingAgent,
)
import logging  # 用于日志记录
import traceback  

# 配置日志记录
task_logger = logging.getLogger("TaskLogger")
task_logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
task_logger.addHandler(handler)

def generate_summary_from_messages(messages, user_input, answer, model):
    """Generate a summary based on the conversation messages history."""
    try:
        task_logger.info("正在生成会话消息历史的概括...")
        question = user_input
        answer = answer

        # 默认值
        default_summary = "No summary available."
        reasoning = "No reasoning process provided."
        
        # # 初始化结构
        # user_messages = []
        # assistant_messages = []

        # # 遍历消息历史
        # for message in messages:
        #     if message["role"] == "user":
        #         user_messages.append(message["content"])
        #     elif message["role"] == "assistant":
        #         assistant_messages.append(message["content"])
        
        # # 处理用户消息和助手消息
        # question = user_messages[-1] if user_messages else "Unknown question"
        # answer = assistant_messages[-1] if assistant_messages else "Unknown answer"

        # # 构造模型输入
        # reasoning = "\n\n".join([f"User: {msg}" for msg in user_messages] + 
        #                          [f"Assistant: {msg}" for msg in assistant_messages])
        reasoning = messages
        
        # 定义消息结构
        messages_for_model = [
            {
                "role": "system",
                "content": "You are a helpful assistant summarizing a conversation history."
            },
            {
                "role": "user",
                "content": f"""Question: {question}

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

Your report must be written clearly, sectioned by part, and all source citations must include **both quoted text and URLs**. This is the most important requirement for verification.
"""
            }
        ]

        # 调用模型生成总结
        summary = model(messages_for_model)
        summary_text = summary.content if hasattr(summary, 'content') else str(summary)
        task_logger.info("会话概括生成完成")
        return f"\n\n### Conversation Summary ###\n{summary_text}\n\n"
    except Exception as e:
        # 捕获错误并记录
        error_type = type(e).__name__
        error_msg = str(e)
        import traceback
        trace = traceback.format_exc()
        task_logger.error(f"生成会话总结时出错: {error_type}: {error_msg}")
        task_logger.debug(f"详细错误信息:\n{trace}")
        return f"\n\n### Conversation Summary ###\nUnable to generate summary: {error_type}: {error_msg}\n\n"
