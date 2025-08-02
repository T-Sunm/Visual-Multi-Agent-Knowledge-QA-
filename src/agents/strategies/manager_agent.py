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

                When all three required actions have been executed, finish with: Finish

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
                4. Respond **in Vietnamese** on one line in the format: `Answer: <Vietnamese_candidate_name>`

                ### EXAMPLE
                Context: A close-up of an elephant standing behind a cement wall.  
                Question: What item in the picture is purported to have a great memory?  
                Candidates: elephant(0.99), trunk(0.70), dumbo(0.09), brain(0.08), tusk(0.03)  
                KBs_knowledge: Elephants are renowned for their excellent memory and are often housed in zoos and sanctuaries.
                LLM_knowledge: A cement wall is a wall made of cement. Cement is a mixture of sand, gravel, and other xxxxxx. Great memory is a memory that is very good at remembering things xxxxxx.  
                Answer: Con voi
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