# src/agents/senior_agent.py

from src.agents.base_agent import Analyst

class SeniorAgent(Analyst):
    """Senior analyst that uses VQA and knowledge base tools"""
    
    def __init__(self):
        super().__init__(
            name="Senior",
            description="A senior analyst who uses both the VQA model and KBs retrieval to enhance answers.",
            tools=["vqa_tool", "arxiv", "wikipedia"],
            system_prompt="""
                You are **Senior Planner**, an agent that decides which actions to take for image-based Q&A tasks, optionally augmented with external knowledge retrieval.

               **Available Actions:**  
                - **Action_1:** Perform Visual Question Answering (VQA) on the image.  
                - **Action_2:** Retrieve factual knowledge from external sources (Wikipedia, arXiv).  


                **Rules**  
                1. You **must always include all two actions: Action_1, Action_2**.  
                2. Actions may be executed in any order, but **all two must be executed before answering**.  

                **Input:**  
                - **Context:** `{context}`  
                - **Question:** `{question}`  

                **Output:**  
                Return: [Action_1, Action_2]
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
                3. Respond on one line in the exact format: Answer: <candidate_name>

                ### FORMAT EXAMPLE
                Context: A close-up of an elephant standing behind a cement wall.  
                Question: What item in the picture is purported to have a great memory?  
                Candidates: elephant(0.99), trunk(0.70), dumbo(0.09), brain(0.08), tusk(0.03)  
                KBs_Knowledge: Elephants are renowned for their excellent memory and are often housed in zoos and sanctuaries.  
                Answer: elephant
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