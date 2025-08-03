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
                You are **Senior Planner**, an agent that decides which actions to take for image-based Q&A tasks, optionally augmented with external knowledge retrieval.

               **Available Actions:**  
                - Action_1: Perform Visual Question Answering (VQA) on the image by first translating the Vietnamese question to English to ensure correct understanding, then analyzing the image content in detail to find the candidate answer.  
                - Action_2: Retrieve factual knowledge from external sources (Wikipedia) relevant to the question, using the English translation of the question as the query to gather accurate and up-to-date information.  

                **Use the following format:**
                Thought: your chain-of-thought for this step  
                Action: the action you will take (Action_1 / Action_2 / “finish”)  
                Action Input: the input for that action  
                Observation: the result returned by that action   
                (Repeat any number of Thought/Action/Action Input/Observation blocks as needed.)

                **Mandatory Execution Rules
                1. **You MUST invoke Action_1, then Action_2 — at least once each — before you may output “Finish”.**  
                2. After the two required actions, you MAY take extra actions (including calling any action again except Action_1) if helpful, but only output `Finish` once you are satisfied.  
                3. When all two required actions have been executed, finish with: Finish

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
                3. Translate that candidate name into Vietnamese.  
                4. Respond on one line in the exact format:  `Answer: <Vietnamese_candidate_name> | Evidence: <Vietnamese_sentence>`

                ### EXAMPLE
                Context: A photo shows a ripe red apple placed beside a small bunch of ripe bananas on a wooden kitchen table.
                Question: Loại quả nào trong hình thường có màu vàng khi chín?
                Candidates: apple (0.10), banana (0.85), cherry (0.05), lemon (0.12), grape (0.08)
                KBs_knowledge: Bananas turn yellow as they ripen, whereas apples can be red, green, or yellow, and lemons are yellow.
                Answer: Quả chuối | Evidence: Context nêu “chùm chuối” xuất hiện; Question hỏi trái nào “thường vàng khi chín”; KBs_knowledge khẳng định chuối chín sẽ đổi vỏ sang màu vàng; trong Candidates, “banana” có xác suất cao nhất 0.85 và khớp hoàn toàn với mô tả—vì vậy đáp án chính là “Quả chuối”.
                ### END OF EXAMPLE
                
                ### Now solve the new task
                Context: {context}
                Question: {question}
                Candidates: {candidates}
                KBs_Knowledge: {KBs_Knowledge}
                Answer: | Evidence:
        """
        )

def create_senior_agent() -> SeniorAgent:
    """Factory function to create senior agent"""
    return SeniorAgent()