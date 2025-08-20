from bert_score import score as bert_score
from typing import List, Dict
from src.models.llm_provider import get_llm
from src.utils.text_processing import extract_explanation, remove_think_block


class ConsensusJudgeAgent():
    """Consensus judge agent"""
    def __init__(self, sim_threshold: float = 0.5, min_pairs: int = 2):
        self.system_prompt = """
            **Goal**: From the evidence, write a logical explanation in Vietnamese.

            **Rules**:
            1.  Your final output must use this exact format:
                Explanation: <lời giải thích bằng tiếng Việt>
            2.  The explanation must be around **7-10 words**. Only state the main visual fact that justifies the answer. DO NOT add any extra words or concluding thoughts.

            ### EXAMPLE 1
            Question: Bàn được làm bằng gì?
            Answer: Gỗ
            Evidence1: Bề mặt bàn màu nâu, mịn và bóng, là đặc điểm của gỗ.
            Evidence2: Mô tả về 'bàn ăn bằng gỗ' khớp với lựa chọn 'Gỗ'.
            Evidence3: Phân tích cho thấy bàn có bề mặt nâu, mịn, sáng bóng, khớp với kiến thức về gỗ đã qua xử lý.
            Explanation: Chiếc bàn trong ảnh có bề mặt màu nâu và bóng.

            ### EXAMPLE 2
            Question: Các con vật đang làm gì?
            Answer: Gặm cỏ
            Evidence1: Những con ngựa vằn đang cúi đầu gần bãi cỏ.
            Evidence2: Bối cảnh ngựa vằn trên đồng cỏ gợi ý hành động gặm cỏ.
            Evidence3: Hình ảnh cho thấy miệng ngựa vằn gần mặt đất, khớp với tập tính ăn uống của chúng.
            Explanation: Đàn ngựa vằn đang cúi đầu xuống ăn trên đồng cỏ.

            ### EXAMPLE 3
            Question: Người đàn ông đang làm gì?
            Answer: cưỡi ngựa
            Evidence1: Phân tích cho thấy hai người đàn ông ngồi trên ngựa.
            Evidence2: Ngữ cảnh mô tả cảnh sát 'trên lưng ngựa', tức là cưỡi ngựa.
            Evidence3: Rationale chỉ ra hành động là 'cưỡi ngựa', dù các ứng viên đều sai.
            Explanation: Bức ảnh cho thấy hai người đàn ông đang ngồi trên ngựa.
            ### END EXAMPLE

            ### Now, using the same format, generate the final explanation for the new task:
            Question: {question}
            Answer: {answer}
            Evidence 1: {evidence_1}
            Evidence 2: {evidence_2}
            Evidence 3: {evidence_3}
            Explanation:
        """

        self.sim_threshold = sim_threshold
        self.min_pairs = min_pairs
        self.lang = "vi"

    def __call__(self, question: str, answer: str, rationales: List[Dict[str, str]]) -> tuple[str, str]:

        # convert list of dict to dict
        agent_results = {k: v for d in rationales for k, v in d.items()}
        # 1) Tính mức độ tương đồng giữa 3 thinking
        junior_result = agent_results.get("Junior", "")
        senior_result = agent_results.get("Senior", "")
        manager_result = agent_results.get("Manager", "")
        thinkings = [junior_result, senior_result, manager_result]
        sim_ok = self._is_consistent(thinkings)

        if sim_ok:
            explanation = self._aggregate_explanation(question, answer, thinkings)
            return (answer, explanation)
        else:
            return ("uncertain", "uncertain")

    def _is_consistent(self, thinkings: List[str]) -> bool:
        """
        Trả về True nếu có ít nhất min_pairs cặp thinking có BERTScore F1 >= threshold
        """
        refs, cands = [], []
        for i in range(len(thinkings)):
            for j in range(i + 1, len(thinkings)):
                refs.append(thinkings[i])
                cands.append(thinkings[j])

        P, R, F1 = bert_score(cands, refs, lang=self.lang, verbose=False)
        ok_flags = [f.item() >= self.sim_threshold for f in F1]
        ok_count  = sum(ok_flags)

        return ok_count >= self.min_pairs

    def _aggregate_explanation(self, question: str, answer: str, thinkings: List[str]) -> str:
        llm = get_llm(temperature=0.7)
        format_dict = {
            "question": question,
            "answer": answer,
            "evidence_1": thinkings[0],
            "evidence_2": thinkings[1],
            "evidence_3": thinkings[2]
        }
        system_prompt = self.system_prompt.format(**format_dict)
        response = llm.invoke(system_prompt)
        cleaned_content = remove_think_block(response.content)
        explanation = extract_explanation(cleaned_content)
        return explanation

