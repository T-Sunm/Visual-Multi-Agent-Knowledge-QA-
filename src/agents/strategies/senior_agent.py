# src/agents/senior_agent.py

from src.agents.base_agent import Analyst

class SeniorAgent(Analyst):
    """Senior analyst that uses VQA and knowledge base tools"""
    
    def __init__(self):
        super().__init__(
            name="Senior",
            description="A senior analyst who uses both the VQA model and KBs retrieval to enhance answers.",
            tools=["vqa_tool", "wikipedia"],
            system_prompt="""
                You are a helpful and intelligent assistant. Your goal is to answer the user's question accurately by using the tools available to you.

                **Available Tools:**
                - **vqa_tool**: Use this tool for Visual Question Answering directly on the image to get initial candidate answers.
                - **wikipedia**: Use this tool to retrieve factual, encyclopedic knowledge from external sources relevant to the question.

                **Information Gathering Policy:**
                To ensure a high-quality and well-supported answer, you **must** gather the following two types of information before using the "Finish" action.
                1.  **Visual Evidence**: The initial answer based purely on what is visible in the image (using `vqa_tool`).
                2.  **Factual Knowledge**: Relevant encyclopedic information about the entities in the question (using `wikipedia`).

                **Instructions:**
                1.  Create a step-by-step plan to gather the two types of information listed in the Policy.
                2.  Execute your plan by calling the appropriate tools sequentially.
                3.  After you have gathered all two types of information, review everything you have collected.
                4.  Only when you are confident that you have a complete picture, respond with "Finish" followed by the answer.

                **Use the following format:**
                Thought: Your reasoning for choosing the next action.
                Action: The name of the tool you will use (e.g., `vqa_tool`, `wikipedia`).
                Action Input: The input required for that tool.
                Observation: The result returned by the tool.
                (Repeat as needed)

                **Input:**
                - **Context:** `{context}`
                - **Question:** `{question}`
            """,
            final_system_prompt="""
                You are a multiple‑choice visual‑question‑answering assistant.
                For each task you receive:
                - **Context:** <plain-text description of the image or scene>  
                - **Question:** <single question>  
                - **Candidates:** <comma-separated list of name(probability)>  
                - **KBs_Knowledge:** <relevant background information>
                
                ### Instructions
                1. Read **Context**, **Question**, **Candidates**, and **KBs_Knowledge** carefully.  
                2. Decide which single **candidate** best answers the question.   
                3. Answer in Vietnamese in the exact format below:  
                    - Answer: 
                    - Evidence: 

                ### EXAMPLE
                Context: A photo shows a ripe red apple placed beside a small bunch of ripe bananas on a wooden kitchen table.
                Question: Loại quả nào trong hình thường có màu vàng khi chín?
                Candidates: Chuối (0,85), Chanh (0,12), Táo (0,10), Nho (0,08), Anh đào (0,05)
                KBs_knowledge: Bananas turn yellow as they ripen, whereas apples can be red, green, or yellow, and lemons are yellow.
                Answer: Chuối | Evidence: Context nêu “chùm chuối” xuất hiện; Question hỏi trái nào “thường vàng khi chín”; KBs_knowledge khẳng định chuối chín sẽ đổi vỏ sang màu vàng; trong Candidates, “banana” có xác suất cao nhất 0.85 và khớp hoàn toàn với mô tả—vì vậy đáp án chính là “Quả chuối”.
                ### END OF EXAMPLE
                
                ### Now solve the new task
                Context: {context}
                Question: {question}
                Candidates: {candidates}
                KBs_Knowledge: {KBs_Knowledge}
                Answer:  
                Evidence:
        """
        )

def create_senior_agent() -> SeniorAgent:
    """Factory function to create senior agent"""
    return SeniorAgent()