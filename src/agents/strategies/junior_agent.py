from src.agents.base_agent import Analyst

class JuniorAgent(Analyst):
    """Junior analyst that uses only VQA tool"""
    def __init__(self):
        super().__init__(
            name="Junior",
            description="A junior analyst who uses only the vanilla VQA model to generate candidate answers.",
            tools=["vqa_tool"],
            system_prompt="""
                You are **Junior Planner**, a lightweight agent that decides which actions to take for basic image-based Q&A tasks.

                **Available Actions**  
                - Action_1: Perform Visual Question Answering (VQA) on the image by first translating the Vietnamese question to English to ensure correct understanding, then analyzing the image content in detail to find the candidate answer.  

                **Use the following format:**
                Thought: your chain-of-thought for this step  
                Action: the action you will take (Action_1 / “finish”)  
                Action Input: the input for that action  
                Observation: the result returned by that action   
                (Repeat any number of Thought/Action/Action Input/Observation blocks as needed.)

                **Mandatory Execution Rules**  
                1. **You MUST invoke Action_1 — at least once each — before you may output “Finish”.**  
                2. After the required action, you MAY take extra actions (including calling any action again except Action_1) if helpful, but only output `Finish` once you are satisfied.  
                3. When all required actions have been executed, finish with: Finish    

                **Input**  
                - **Context:** `{context}`  
                - **Question:** {question}
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
                3. Translate that candidate name into Vietnamese.  
                4. Respond on one line in the exact format:  `Answer: <Vietnamese_candidate_name> | Evidence: <Vietnamese_sentence>`

                ### FORMAT EXAMPLE  
                Context: A photo shows a ripe red apple placed beside a small bunch of ripe bananas on a wooden kitchen table.
                Question: Loại quả nào trong hình thường có màu vàng khi chín?
                Candidates: apple (0.10), banana (0.85), cherry (0.05), lemon (0.12), grape (0.08)
                Answer: Quả chuối | Evidence: Context mô tả “chùm chuối” xuất hiện; Question hỏi trái nào “thường có màu vàng khi chín”; trong Candidates, “banana” có xác suất cao nhất 0.85, khớp hoàn toàn với mô tả—vì thế đáp án chắc chắn là Quả chuối.
                ### END OF EXAMPLE
                
                ### Now solve the new task  
                Context: {context}  
                Question: {question}  
                Candidates: {candidates}  
                Answer: | Evidence:
            """
        )

def create_junior_agent() -> JuniorAgent:
    """Factory function to create junior agent"""
    return JuniorAgent()