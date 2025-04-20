

def generate_summary_from_messages(messages):
    """Generate a summary based on the conversation messages history."""
    try:
        task_logger.info("正在生成会话消息历史的概括...")

        # 默认值
        default_summary = "No summary available."
        reasoning = "No reasoning process provided."
        
        # 初始化结构
        user_messages = []
        assistant_messages = []

        # 遍历消息历史
        for message in messages:
            if message["role"] == "user":
                user_messages.append(message["content"])
            elif message["role"] == "assistant":
                assistant_messages.append(message["content"])
        
        # 处理用户消息和助手消息
        question = user_messages[-1] if user_messages else "Unknown question"
        answer = assistant_messages[-1] if assistant_messages else "Unknown answer"

        # 构造模型输入
        reasoning = "\n\n".join([f"User: {msg}" for msg in user_messages] + 
                                 [f"Assistant: {msg}" for msg in assistant_messages])
        
        # 定义消息结构
        messages_for_model = [
            {
                "role": "system",
                "content": "You are a helpful assistant summarizing a conversation history."
            },
            {
                "role": "user",
                "content": f"""Please summarize the following conversation history:

**User Question:**  
{question}

**Assistant Answer:**  
{answer}

**Conversation History:**  
{reasoning}

Your summary must include:
1. The user's question.
2. The assistant's response.
3. Key points discussed during the entire conversation.
4. Suggestions for improvement or missing information, if applicable.
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
