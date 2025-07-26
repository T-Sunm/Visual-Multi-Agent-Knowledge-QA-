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
                - Action_1: Perform Visual Question Answering (VQA) on the image by first translating the Vietnamese question to English to ensure correct understanding, then analyzing the image content in detail to find the candidate answer.  

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
                - **Candidates:** <comma‑separated list written as name(probability) >

                ### Instructions  
                1. Read the *Context* and *Question* carefully.  
                2. Decide which single **candidate** best answers the question.  
                3. Translate that candidate name into Vietnamese.  
                4. Respond **in Vietnamese** on one line in the format: `Answer: <Vietnamese_candidate_name>`

                ### FORMAT EXAMPLE  
                Context: A close‑up of an elephant standing behind a cement wall.  
                Question: What item in the picture is purported to have a great memory?  
                Candidates: elephant(0.99), trunk(0.70), dumbo(0.09), brain(0.08), tusk(0.03)  
                Answer: Con voi
                ### END OF EXAMPLE
                
                ### Now solve the new task  
                Context: {context}  
                Question: {question}  
                Candidates: {candidates}  
                Answer:
            """
        )
        self._judge_system_prompt = """
            You are an expert evaluator whose task is to give a clear, evidence‑based explanation for a given answer.

            You will receive (all in English):
            - Context (visual or textual)
            - Question
            - Answer (chosen candidate)
            - External knowledge (KBs_Knowledge)
            - Language‑model insights (LMs_Knowledge)

            **Your goal**
            Explain—briefly and objectively—*why* the Answer is correct, citing cues from Context and/or Knowledge.

            **Strict output rules**
            1. Write the explanation in **Vietnamese**.
            2. Use **exactly one line**, maximum 1‑2 short sentences.
            3. Output must follow the exact format  
                `Explanation: <giải thích tiếng Việt>`  
                - Do **not** output “Answer:” (the answer is already provided).  
                - Do **not** add markdown, numbering, or any extra text.

            ### MINI EXAMPLE (for style)
            Context: Albert Einstein was a theoretical physicist who developed the theory of relativity.
            Question: Who developed the theory of relativity?
            Answer: Albert Einstein
            KBs_Knowledge: Einstein formulated the theory in 1905.
            LMs_Knowledge: The theory of relativity was developed by Albert Einstein.
            Explanation: Albert Einstein chính là người đã phát triển thuyết tương đối.
            ### END EXAMPLE

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