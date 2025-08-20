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
                You are a helpful and intelligent assistant. Your goal is to answer the user's question accurately by using the tools available to you.

                **Available Tools:**
                - **vqa_tool**: Use this tool for Visual Question Answering directly on the image to get initial candidate answers.
                - **wikipedia**: Use this tool to retrieve factual, encyclopedic knowledge from external sources relevant to the question.

                **Information Gathering Policy:**
                To ensure a high-quality and well-supported answer, you **must** gather the following two types of information before using the "Finish" action.
                1.  **Visual Evidence**: The initial answer based purely on what is visible in the image (using `vqa_tool`).
                2.  **Factual Knowledge**: Relevant encyclopedic information about the entities in the question (using `wikipedia`).

                **Instructions:**
                1.  Create a step-by-step plan to gather the information.
                2.  After you have gathered all information, review everything you have collected.
                3.  Only when you are confident that you have a complete picture, respond with "Finish" followed by the answer.

                **Use the following format:**
                Thought: Your reasoning for choosing the next action.
                Action: The name of the tool you will use (e.g., `vqa_tool`, `wikipedia`).
                Action Input: The input required for that tool.
                Observation: The result returned by the tool.
                (Repeat as needed)

                **Input:**
                - **Context:** `{context}`
                - **Question:** `{question}`
            """,
            rationale_system_prompt="""
                Your task is to generate a logical explanation in Vietnamese, around 10-15 words. Do not include a final concluding sentence. Synthesize the visual details from 'Candidates', 'Context', with the facts from 'KBs_Knowledge'.
                Important: The 'Candidates' list is a suggestion and may be misleading or entirely incorrect.

                ### EXAMPLE 1
                Context: A wooden dining table is shown with a glossy finish.
                Question: Bàn được làm bằng gì?
                Candidates: Gỗ (0.92), Kim loại (0.05), Nhựa (0.02), Đá (0.01), Kính (0.00)
                KBs_Knowledge: Materials that are brown, smooth, and shiny are often polished or varnished wood.
                Rationale: Ngữ cảnh cho thấy một chiếc bàn có bề mặt màu nâu, mịn và sáng bóng, khớp với đặc điểm của gỗ.

                ### EXAMPLE 2
                Context: A photo of a single banana that has been partially peeled.
                Question: Quả chuối có đóng không?
                Candidates: không (0.95), có (0.05), bị thối (0.00), còn xanh (0.00), bằng nhựa (0.00)
                KBs_Knowledge: A banana is considered 'not closed' or 'open' when its peel is removed to expose the edible fruit inside.
                Rationale: Ngữ cảnh mô tả quả chuối đã được bóc vỏ một phần làm lộ ruột, trạng thái này được coi là 'không đóng'.

                ### EXAMPLE 3
                Context: A herd of zebras are gathered on a grassy field.
                Question: Các con vật đang làm gì?
                Candidates: Gặm cỏ (0.88), Đứng im (0.09), Chạy (0.02), Uống nước (0.01), Nằm nghỉ (0.00)
                KBs_Knowledge: Herbivores like zebras eat grass by lowering their mouths to the ground. This action is called grazing.
                Rationale: Ngữ cảnh cho thấy đàn ngựa vằn đang cúi đầu xuống đất, đây là hành vi gặm cỏ.

                ### EXAMPLE 4
                Context: Two police officers on horseback patrolling a city street.
                Question: Người đàn ông đang làm gì?
                Candidates: đi bộ (0.65), đứng im (0.20), nói chuyện (0.10), chạy (0.05)
                KBs_Knowledge: The action of sitting on and controlling a horse is called 'riding a horse' (cưỡi ngựa).
                Rationale: Ngữ cảnh mô tả người đàn ông đang ở 'trên lưng ngựa', hành động này được gọi là 'cưỡi ngựa'.

                ### Now solve the new task
                Context: {context}
                Question: {question}
                Candidates: {candidates}
                KBs_Knowledge: {KBs_Knowledge}
                Rationale:
            """,
            final_system_prompt="""
                You are an visual-question-answering assistant that generates the most accurate answer based on evidence.
                For each task you receive:
                - **Context:** <plain-text description of the image or scene>
                - **Question:** <single question>
                - **Candidates:** <A list of possible answers with probabilities, generated by a vision model>
                - **Rationale:** <The core reasoning that justifies the final answer.>

                ### Instructions
                1. Read **Context**, **Question**, **Candidates**, and **Rationale** carefully.
                2. The 'Candidates' are only suggestions and may be incorrect. Your final answer must be based on the evidence in the `Rationale`. If the `Rationale` points to an answer not in the `Candidates` list, you must ignore the list.
                3. Answer each question concisely in a single word or short phrase.
                4. The final answer MUST be in Vietnamese, matching the language of the Question.
                
                ### EXAMPLE 1
                Context: A wooden dining table is shown with a glossy finish.
                Question: Bàn được làm bằng gì?
                Candidates: Gỗ (0.92), Kim loại (0.05), Nhựa (0.02), Đá (0.01), Kính (0.00)
                Rationale: Ngữ cảnh cho thấy một chiếc bàn có bề mặt màu nâu, mịn và sáng bóng, khớp với đặc điểm của gỗ.
                Answer: Gỗ

                ### EXAMPLE 2
                Context: A photo of a single banana that has been partially peeled.
                Question: Quả chuối có đóng không?
                Candidates: không (0.95), có (0.05), bị thối (0.00), còn xanh (0.00), bằng nhựa (0.00)
                Rationale: Ngữ cảnh mô tả quả chuối đã được bóc vỏ một phần làm lộ ruột, trạng thái này được coi là 'không đóng'.
                Answer: không

                ### EXAMPLE 3
                Context: A herd of zebras are gathered on a grassy field.
                Question: Các con vật đang làm gì?
                Candidates: Gặm cỏ (0.88), Đứng im (0.09), Chạy (0.02), Uống nước (0.01), Nằm nghỉ (0.00)
                Rationale: Ngữ cảnh cho thấy đàn ngựa vằn đang cúi đầu xuống đất, đây là hành vi gặm cỏ.
                Answer: Gặm cỏ

                ### EXAMPLE 4 (BIAS-BREAKING EXAMPLE)
                Context: Two police officers on horseback patrolling a city street.
                Question: Người đàn ông đang làm gì?
                Candidates: đi bộ (0.65), đứng im (0.20), nói chuyện (0.10), chạy (0.05)
                Rationale: Ngữ cảnh mô tả người đàn ông đang ở 'trên lưng ngựa', hành động này được gọi là 'cưỡi ngựa'.
                Answer: cưỡi ngựa
                ### END OF EXAMPLE

                ### Now solve the new task
                Context: {context}
                Question: {question}
                Candidates: {candidates}
                Rationale: {rationale}
                Answer:
            """
        )

def create_senior_agent() -> SeniorAgent:
    """Factory function to create senior agent"""
    return SeniorAgent()