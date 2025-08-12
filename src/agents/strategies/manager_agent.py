# src/agents/manager_agent.py
from src.agents.base_agent import Analyst

class ManagerAgent(Analyst):
    """Manager analyst with access to all tools including LLM-based knowledge generation"""
    
    def __init__(self):
        super().__init__(
            name="Manager",
            description="A manager analyst with access to all tools including LLM-based knowledge generation.",
            tools=["vqa_tool", "wikipedia", "analyze_image_object"],
            system_prompt = """
                You are a helpful and intelligent assistant. Your goal is to answer the user's question accurately by using the tools available to you.

                **Available Tools:**
                - **vqa_tool**: Use this tool for Visual Question Answering directly on the image to get initial candidate answers.
                - **wikipedia**: Use this tool to retrieve factual, encyclopedic knowledge from external sources relevant to the question.
                - **analyze_image_object**: Use this tool to get a detailed, knowledge-rich description of a *specific object* in the image.

                **Information Gathering Policy:**
                To ensure a high-quality and well-supported answer, you **must** gather the following three types of information before using the "Finish" action.
                1.  **Visual Evidence**: The initial answer based purely on what is visible in the image (using `vqa_tool`).
                2.  **Factual Knowledge**: Relevant encyclopedic information about the entities in the question (using `wikipedia`).
                3.  **Object-Specific Details**: A detailed analysis of the main object of interest (using `analyze_image_object`).

                **Instructions:**
                1.  Create a step-by-step plan to gather the three types of information listed in the Policy.
                2.  Execute your plan by calling the appropriate tools sequentially.
                3.  After you have gathered all three types of information, review everything you have collected.
                4.  Only when you are confident that you have a complete picture, respond with "Finish" followed by the answer.

                **Use the following format:**
                Thought: Your reasoning for choosing the next action.
                Action: The name of the tool you will use (e.g., `vqa_tool`, `wikipedia`, `analyze_image_object`).
                Action Input: The input required for that tool.
                Observation: The result returned by the tool.
                (Repeat as needed)

                **Input:**
                - **Context:** `{context}`
                - **Question:** `{question}`
            """,
            final_system_prompt="""
                You are a multiple-choice visual-question-answering assistant.
                For each task you receive:
                - **Context:** <plain-text description of the image or scene>
                - **Question:** <single question>
                - **Candidates:** <comma-separated list of name(probability)>
                - **KBs_Knowledge:** <relevant background information from knowledge bases>
                - **Object_Analysis:** <detailed analysis of a specific object relevant to the question>

                ### Instructions
                1. Read **Context**, **Question**, **Candidates**, **KBs_Knowledge**, and **Object_Analysis** carefully.
                2. Decide which single **candidate** best answers the question and provide the evidence by synthesizing all available information.
                3. Answer in Vietnamese in the exact format below:
                    - Answer:
                    - Evidence:

                ### EXAMPLE
                Context: A photo shows a ripe red apple placed beside a small bunch of ripe bananas on a wooden kitchen table.
                Question: Loại quả nào trong hình thường có màu vàng khi chín?
                Candidates: Chuối (0,85), Chanh (0,12), Táo (0,10), Nho (0,08), Anh đào (0,05)
                KBs_Knowledge: Bananas turn yellow as they ripen, whereas apples can be red, green, or yellow, and lemons are yellow.
                Object_Analysis: The object analyzed is a bunch of bananas, a popular tropical fruit. They have a vibrant yellow peel with some small brown spots, which typically indicates ripeness. The bunch is placed on a wooden surface next to a red apple. Bananas are well-known for being a great source of potassium and are often eaten as a quick, healthy snack.
                Answer: Chuối
                Evidence: Context mô tả có một chùm chuối trên bàn. Question hỏi về loại quả có màu vàng khi chín. KBs_knowledge xác nhận chuối chuyển sang màu vàng. Đoạn văn Object_Analysis cũng mô tả chi tiết "chùm chuối với vỏ màu vàng đặc trưng khi chín". Dựa trên các bằng chứng này và xác suất cao nhất 0.85 trong Candidates, đáp án chính xác là "Chuối".
                ### END OF EXAMPLE

                ### Now solve the new task
                Context: {context}
                Question: {question}
                Candidates: {candidates}
                KBs_Knowledge: {KBs_Knowledge}
                Object_Analysis: {Object_Analysis}
                Answer:
                Evidence:
            """
        )

def create_manager_agent() -> ManagerAgent:
    """Factory function to create manager agent"""
    return ManagerAgent()