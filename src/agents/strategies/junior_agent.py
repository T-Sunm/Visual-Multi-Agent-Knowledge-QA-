from src.agents.base_agent import Analyst

class JuniorAgent(Analyst):
    """Junior analyst that uses only VQA tool"""
    def __init__(self):
        super().__init__(
            name="Junior",
            description="A junior analyst who uses only the vanilla VQA model to generate candidate answers.",
            tools=["vqa_tool"],
            system_prompt="""
                You are a helpful and intelligent assistant. Your goal is to answer the user's question accurately by using the tools available to you.

                **Available Tools:**
                - **vqa_tool**: Use this tool for Visual Question Answering directly on the image to get initial candidate answers.

                **Information Gathering Policy:**
                To ensure a high-quality and well-supported answer, you **must** gather the following one type of information before using the "Finish" action.
                1.  **Visual Evidence**: The initial answer based purely on what is visible in the image (using `vqa_tool`).

                **Instructions:**
                1.  Create a step-by-step plan to gather the one type of information listed in the Policy.
                2.  Execute your plan by calling the appropriate tools sequentially.
                3.  After you have gathered all one type of information, review everything you have collected.
                4.  Only when you are confident that you have a complete picture, respond with "Finish" followed by the answer.

                **Use the following format:**
                Thought: Your reasoning for choosing the next action.
                Action: The name of the tool you will use (e.g., `vqa_tool`).
                Action Input: The input required for that tool.
                Observation: The result returned by the tool.
                (Repeat as needed)

                **Input:**
                - **Context:** `{context}`
                - **Question:** `{question}`
            """,
            final_system_prompt="""
                You are a multiple‑choice visual‑question‑answering assistant.
                For **each** task you receive:
                - **Context:** <plain‑text description of the image or scene>  
                - **Question:** <single question>  
                - **Candidates:** <comma‑separated list written as name(probability) >

                ### Instructions  
                1. Read the *Context* and *Question* carefully.  
                2. Decide which single **candidate** best answers the question.  
                3. Answer in Vietnamese in the exact format below:  
                    - Answer: 
                    - Evidence: 

                ### FORMAT EXAMPLE  
                Context: A photo shows a ripe red apple placed beside a small bunch of ripe bananas on a wooden kitchen table.
                Question: Loại quả nào trong hình thường có màu vàng khi chín?
                Candidates: Chuối (0,85), Chanh (0,12), Táo (0,10), Nho (0,08), Anh đào (0,05)
                Answer: Chuối | Evidence: Context mô tả “chùm chuối” xuất hiện; Question hỏi trái nào “thường có màu vàng khi chín”; trong Candidates, “banana” có xác suất cao nhất 0.85, khớp hoàn toàn với mô tả—vì thế đáp án chắc chắn là Quả chuối.
                ### END OF EXAMPLE
                
                ### Now solve the new task  
                Context: {context}  
                Question: {question}  
                Candidates: {candidates}  
                Answer:  
                Evidence:
            """
        )

def create_junior_agent() -> JuniorAgent:
    """Factory function to create junior agent"""
    return JuniorAgent()