# src/agents/manager_agent.py
from src.agents.base_agent import Analyst

class ManagerAgent(Analyst):
    """Manager analyst with access to all tools including LLM-based knowledge generation"""
    
    def __init__(self):
        super().__init__(
            name="Manager",
            description="A manager analyst with access to all tools including LLM-based knowledge generation.",
            tools=["vqa_tool", "arxiv", "wikipedia"],
            system_prompt = """
                You are **Manager Planner**, an advanced agent that decides which actions to take for complex image-based Q&A tasks requiring reasoning and external knowledge.

                **Available Actions**  
                - **Action_1:** Perform Visual Question Answering (VQA) on the image using vqa_tool.  
                - **Action_2:** Retrieve background knowledge from arXiv or Wikipedia using arxiv and wikipedia tools.  

                **Rules**  
                1. **Always** begin with **Action_1** to analyze the image. 
                2. **Always** follow with **Action_2** to get additional knowledge.  

                **Input**  
                - **Context:** `{context}`  
                - **Question:** `{question}`  

                **Output**  
                Response format: [Action_1, Action_2]  
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

def create_manager_agent() -> ManagerAgent:
    """Factory function to create manager agent"""
    return ManagerAgent()