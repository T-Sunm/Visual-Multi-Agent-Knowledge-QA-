# src/agents/manager_agent.py
from src.agents.base_agent import Analyst

class ManagerAgent(Analyst):
    """Manager analyst with access to all tools including LLM-based knowledge generation"""
    
    def __init__(self):
        super().__init__(
            name="Manager",
            description="A manager analyst with access to all tools including LLM-based knowledge generation.",
            tools=["vqa_tool", "arxiv", "wikipedia", "lm_knowledge"],
            system_prompt = """
                You are **Manager Planner**, an advanced agent that decides which actions to take for complex image-based Q&A tasks requiring reasoning and external knowledge.

               **Available Actions:**  
                - **Action_1:** Perform Visual Question Answering (VQA) on the image.  
                - **Action_2:** Retrieve factual knowledge from external sources (Wikipedia, arXiv).  
                - **Action_3:** Generate background knowledge from the image using a language model DAM.  

                **Rules**  
                1. You **must always include all three actions: Action_1, Action_2, Action_3**.  
                2. Actions may be executed in any order, but **all three must be executed before answering**.  

                **Input:**  
                - **Context:** `{context}`  
                - **Question:** `{question}`  

                **Output:**  
                Return: [Action_1, Action_2, Action_3]
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
                KBs_knowledge: Elephants are renowned for their excellent memory and are often housed in zoos and sanctuaries.
                LLM_knowledge: A cement wall is a wall made of cement. Cement is a mixture of sand, gravel, and other xxxxxx. Great memory is a memory that is very good at remembering things xxxxxx.  
                Answer: elephant
                ### END OF EXAMPLE
                
                ### Now solve the new task
                Context: {context}
                Question: {question}
                Candidates: {candidates}
                KBs_Knowledge: {KBs_Knowledge}
                LMs_Knowledge: {LMs_Knowledge}
                Answer:
        """
        )

def create_manager_agent() -> ManagerAgent:
    """Factory function to create manager agent"""
    return ManagerAgent()