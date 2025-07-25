from src.agents.base_agent import Analyst
from pydantic import PrivateAttr

class JuniorAgent(Analyst):
    """Junior analyst that uses only VQA tool"""
    _judge_system_prompt: str = PrivateAttr()
    def __init__(self):
        super().__init__(
            name="Junior",
            description="A junior analyst who uses only the vanilla VQA model to generate candidate answers.",
            tools=["vqa_tool"],
            system_prompt="""
                You are **Junior Planner**, a lightweight agent that decides which actions to take for basic image-based Q&A tasks.

                **Available Actions**  
                - **Action_1:** Perform Visual Question Answering (VQA) on the image.

                **Rules**  
                    1. **Always** begin with **Action_1**.

                **Input**  
                - **Question:** {question}

                **Output**  
                Response format:  [Action_1]
            """,
            final_system_prompt="""
                You are a multiple‑choice visual‑question‑answering assistant.
                For **each** task you receive:
                - **Context:** <plain‑text description of the image or scene>  
                - **Question:** <single question>  
                - **Candidates:** <comma‑separated list written as  name(probability) >

                ### Instructions  
                1. Read the *Context* and the *Question* carefully.  
                2. Decide which single **candidate** best answers the question.   
                3. Respond on one line in the exact format: Answer: <candidate_name>

                ### FORMAT EXAMPLE  
                Context: A close‑up of an elephant standing behind a cement wall.  
                Question: What item in the picture is purported to have a great memory?  
                Candidates: elephant(0.99), trunk(0.70), dumbo(0.09), brain(0.08), tusk(0.03)  
                Answer: elephant
                ### END OF EXAMPLE
                
                ### Now solve the new task  
                Context: {context}  
                Question: {question}  
                Candidates: {candidates}  
                Answer:
            """
        )
        self._judge_system_prompt = """
            You are an expert assistant whose task is to provide a clear and concise explanation for a given answer.

            You will be provided with:
            - A question
            - Visual or textual context
            - An answer
            - External knowledge (KBs_Knowledge)
            - Language model insights (LMs_Knowledge)

            Your job is to explain based on the context and knowledge. Explain the reasoning behind the answer using evidence from the context and knowledge.

            Keep your explanation objective and no longer than 1-2 sentences.

            Use the following format:
            - Explanation: <your explanation here>

            ### Now generate the explanation  
            Context: {context}  
            Question: {question}  
            Answer: {answer}  
            KBs_Knowledge: {KBs_Knowledge}  
            LMs_Knowledge: {LMs_Knowledge}  
            Explanation:
        """

def create_junior_agent() -> JuniorAgent:
    """Factory function to create junior agent"""
    return JuniorAgent()