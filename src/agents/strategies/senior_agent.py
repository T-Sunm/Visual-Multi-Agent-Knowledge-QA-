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

                When all two required actions have been executed, finish with: Finish

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
                4. Respond **in Vietnamese** on one line in the format: `Answer: <Vietnamese_candidate_name>`

                ### EXAMPLE
                Context: A close-up of an elephant standing behind a cement wall.  
                Question: What item in the picture is purported to have a great memory?  
                Candidates: elephant(0.99), trunk(0.70), dumbo(0.09), brain(0.08), tusk(0.03)  
                KBs_Knowledge: Elephants are renowned for their excellent memory and are often housed in zoos and sanctuaries.  
                Answer: Con voi
                ### END OF EXAMPLE
                
                ### Now solve the new task
                Context: {context}
                Question: {question}
                Candidates: {candidates}
                KBs_Knowledge: {KBs_Knowledge}
                Answer:
        """
        )

def create_senior_agent() -> SeniorAgent:
    """Factory function to create senior agent"""
    return SeniorAgent()