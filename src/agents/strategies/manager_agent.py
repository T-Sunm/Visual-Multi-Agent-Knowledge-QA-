# src/agents/manager_agent.py
from src.agents.base_agent import Analyst

class ManagerAgent(Analyst):
    """Manager analyst with access to all tools including LLM-based knowledge generation"""
    
    def __init__(self):
        super().__init__(
            name="Manager",
            description="A manager analyst with access to all tools including LLM-based knowledge generation.",
            tools=["vqa_tool", "wikipedia", "analyze_image_object"],
            system_prompt = """
                You are a helpful and intelligent assistant. Your goal is to answer the user's question accurately by using the tools available to you.

                **Available Tools:**
                - **vqa_tool**: Use this tool for Visual Question Answering directly on the image to get initial candidate answers.
                - **wikipedia**: Use this tool to retrieve factual, encyclopedic knowledge from external sources relevant to the question.
                - **analyze_image_object**: Use this tool to get a detailed, knowledge-rich description of a *specific object* in the image.

                **Information Gathering Policy:**
                To ensure a high-quality and well-supported answer, you **must** gather the following three types of information before using the "Finish" action.
                1.  **Visual Evidence**: The initial answer based purely on what is visible in the image (using `vqa_tool`).
                2.  **Factual Knowledge**: Relevant encyclopedic information about the entities in the question (using `wikipedia`).
                3.  **Object-Specific Details**: A detailed analysis of the main object of interest (using `analyze_image_object`).

                **Instructions:**
                1.  Create a step-by-step plan to gather the three types of information listed in the Policy.
                2.  Execute your plan by calling the appropriate tools sequentially.
                3.  After you have gathered all three types of information, review everything you have collected.
                4.  Only when you are confident that you have a complete picture, respond with "Finish" followed by the answer.

                **Use the following format:**
                Thought: Your reasoning for choosing the next action.
                Action: The name of the tool you will use (e.g., `vqa_tool`, `wikipedia`, `analyze_image_object`).
                Action Input: The input required for that tool.
                Observation: The result returned by the tool.
                (Repeat as needed)

                **Input:**
                - **Context:** `{context}`
                - **Question:** `{question}`
            """,
            rationale_system_prompt="""
                Your task is to generate a concise rationale in Vietnamese that only explains the reasoning process. Do not include a final concluding sentence. Synthesize the visual details from 'Candidates', 'Context', 'Object_Analysis', with the facts from 'KBs_Knowledge' to form a step-by-step reasoning process.

                ### EXAMPLE 1
                Context: A wooden dining table is shown with a glossy finish.
                Question: Bàn được làm bằng gì?
                Candidates: Gỗ (0.92), Kim loại (0.05), Nhựa (0.02), Đá (0.01), Kính (0.00)
                KBs_Knowledge: Materials that are brown, smooth, and shiny are often polished or varnished wood.
                Object_Analysis: The object is a table. Its surface is observed to be brown, smooth, and shiny.
                Rationale: Phân tích đối tượng cho thấy bề mặt của chiếc bàn có đặc điểm là màu nâu, mịn và sáng bóng. Dựa trên kiến thức chung, đây là những đặc tính điển hình của vật liệu gỗ đã qua xử lý.
                
                ### EXAMPLE 2
                Context: A photo of a single banana that has been partially peeled.
                Question: Quả chuối có đóng không?
                Candidates: không (0.95), có (0.05), bị thối (0.00), còn xanh (0.00), bằng nhựa (0.00)
                KBs_Knowledge: A banana is considered 'not closed' or 'open' when its peel is removed to expose the edible fruit inside.
                Object_Analysis: The object is a banana with its yellow peel partially pulled back. The inner, edible part of the fruit is visible.
                Rationale: Quan sát hình ảnh cho thấy phần ruột của chuối đã lộ ra bên ngoài lớp vỏ vàng đã được bóc. Theo định nghĩa, khi phần ruột lộ ra thì quả chuối không còn đóng.
                
                ### EXAMPLE 3
                Context: A herd of zebras are gathered on a grassy field.
                Question: Các con vật đang làm gì?
                Candidates: Gặm cỏ (0.88), Đứng im (0.09), Chạy (0.02), Uống nước (0.01), Nằm nghỉ (0.00)
                KBs_Knowledge: Herbivores like zebras eat grass by lowering their mouths to the ground. This action is called grazing.
                Object_Analysis: A group of zebras is visible. Their heads are lowered, and their mouths are positioned close to the ground where the grass is.
                Rationale: Phân tích hình ảnh cho thấy miệng của những con ngựa vằn đang ở rất gần mặt đất, nơi có cỏ. Hành động này khớp với kiến thức về tập tính ăn uống của chúng.
                ### END OF EXAMPLES
                
                ### Now solve the new task
                Context: {context}
                Question: {question}
                Candidates: {candidates}
                KBs_Knowledge: {KBs_Knowledge}
                Object_Analysis: {Object_Analysis}
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
                2. The 'Candidates' are suggestions, but the 'Rationale' is the definitive evidence for your final answer.
                3. Answer in the exact format : 
                    - Answer:

                ### EXAMPLE 1
                Context: A wooden dining table is shown with a glossy finish.
                Question: Bàn được làm bằng gì?
                Candidates: Gỗ (0.92), Kim loại (0.05), Nhựa (0.02), Đá (0.01), Kính (0.00)
                Rationale: Phân tích đối tượng cho thấy bề mặt của chiếc bàn có đặc điểm là màu nâu, mịn và sáng bóng. Dựa trên kiến thức chung, đây là những đặc tính điển hình của vật liệu gỗ đã qua xử lý.
                Answer: Gỗ


                ### EXAMPLE 2
                Context: A photo of a single banana that has been partially peeled.
                Question: Quả chuối có đóng không?
                Candidates: không (0.95), có (0.05), bị thối (0.00), còn xanh (0.00), bằng nhựa (0.00)
                Rationale: Quan sát hình ảnh cho thấy phần ruột của chuối đã lộ ra bên ngoài lớp vỏ vàng đã được bóc. Theo định nghĩa, khi phần ruột lộ ra thì quả chuối không còn đóng.
                Answer: không

                ### EXAMPLE 3
                Context: A herd of zebras are gathered on a grassy field.
                Question: Các con vật đang làm gì?
                Candidates: Gặm cỏ (0.88), Đứng im (0.09), Chạy (0.02), Uống nước (0.01), Nằm nghỉ (0.00)
                Rationale: Phân tích hình ảnh cho thấy miệng của những con ngựa vằn đang ở rất gần mặt đất, nơi có cỏ. Hành động này khớp với kiến thức về tập tính ăn uống của chúng.
                Answer: Gặm cỏ
                ### END OF EXAMPLE

                ### Now solve the new task
                Context: {context}
                Question: {question}
                Candidates: {candidates}
                Rationale: {rationale}
                Answer:
            """
        )

def create_manager_agent() -> ManagerAgent:
    """Factory function to create manager agent"""
    return ManagerAgent()