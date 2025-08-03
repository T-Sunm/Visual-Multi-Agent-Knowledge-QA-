# src/agents/manager_agent.py
from src.agents.base_agent import Analyst

class ManagerAgent(Analyst):
    """Manager analyst with access to all tools including LLM-based knowledge generation"""
    
    def __init__(self):
        super().__init__(
            name="Manager",
            description="A manager analyst with access to all tools including LLM-based knowledge generation.",
            tools=["vqa_tool", "wikipedia", "lm_knowledge"],
            system_prompt = """
                You are **Manager Planner**, an advanced agent that decides which actions to take for complex image-based Q&A tasks requiring reasoning and external knowledge.

               **Available Actions:**  
                - Action_1: Perform Visual Question Answering (VQA) on the image by first translating the Vietnamese question to English to ensure correct understanding, then analyzing the image content in detail to find the candidate answer.  
                - Action_2: Retrieve factual knowledge from external sources (Wikipedia) relevant to the question, using the English translation of the question as the query to gather accurate and up-to-date information.  
                - Action_3: Generate additional background knowledge or context about the image and question using a language model, processing all information in English for completeness.  

                **Use the following format:**
                Thought: your chain-of-thought for this step  
                Action: the action you will take (Action_1 / Action_2 / Action_3 / “finish”)  
                Action Input: the input for that action  
                Observation: the result returned by that action   
                (Repeat any number of Thought/Action/Action Input/Observation blocks as needed.)

                **Mandatory Execution Rules
                1. **You MUST invoke Action_1, then Action_2, then Action_3 — at least once each — before you may output “Finish”.**  
                2. After the three required actions, you MAY take extra actions (including calling any action again except Action_1) if helpful, but only output `Finish` once you are satisfied.  
                3. When all three required actions have been executed, finish with: Finish

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
                1. Read **Context**, **Question**, **Candidates**, **KBs_Knowledge**, **LMs_Knowledge** carefully.  
                2. Decide which single **candidate** best answers the question.   
                3. Translate that candidate name into Vietnamese.  
                4. Respond on one line in the exact format:  `Answer: <Vietnamese_candidate_name> | Evidence: <Vietnamese_sentence>`

                ### EXAMPLE
                Context: A photo shows a ripe red apple placed beside a small bunch of ripe bananas on a wooden kitchen table.  
                Question: Loại quả nào trong hình thường có màu vàng khi chín?
                Candidates: apple (0.10), banana (0.85), cherry (0.05), lemon (0.12), grape (0.08)
                KBs_knowledge: Bananas turn yellow as they ripen, whereas apples can be red, green, or yellow, and lemons are yellow.
                LLM_knowledge: A ripe banana’s peel is characteristically yellow, making it an easily recognized symbol of ripeness. Apples are often red or green; cherries are red; grapes vary in color.  
                Answer: Quả chuối | Evidence: Trong Context có chùm chuối; Question hỏi trái nào “thường vàng khi chín”; KBs_knowledge nêu rõ chuối sẽ chuyển sang màu vàng khi chín; LLM_knowledge bổ sung rằng vỏ chuối chín có màu vàng đặc trưng; và trong Candidates, “banana” có xác suất cao nhất 0.85, nên đáp án chính là “Quả chuối”.
                ### END OF EXAMPLE
                
                ### Now solve the new task
                Context: {context}
                Question: {question}
                Candidates: {candidates}
                KBs_Knowledge: {KBs_Knowledge}
                LMs_Knowledge: {LMs_Knowledge}
                Answer: | Evidence:
        """
        )

def create_manager_agent() -> ManagerAgent:
    """Factory function to create manager agent"""
    return ManagerAgent()