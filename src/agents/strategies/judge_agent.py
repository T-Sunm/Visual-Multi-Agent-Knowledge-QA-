from bert_score import score as bert_score
from typing import List, Dict
from src.models.llm_provider import get_llm
from src.utils.text_processing import extract_explanation, remove_think_block


class ConsensusJudgeAgent():
    """Consensus judge agent"""
    def __init__(self, sim_threshold: float = 0.5, min_pairs: int = 2):
        self.system_prompt = """
            **Goal**: From the evidence, write a very short, logical explanation in Vietnamese.

            **Rules**:
            1.  Your final output must use this exact format:
                Explanation: <lời giải thích bằng tiếng Việt>
            2.  The explanation must be **EXTREMELY SHORT**, around **7-10 words**. Only state the main visual fact. DO NOT add any extra words.

            ### EXAMPLE 
            Question: Loại quả nào trong hình thường có màu vàng khi chín?
            Answer: Quả chuối
            Evidence1: Context mô tả “chùm chuối” xuất hiện; Question hỏi trái nào “thường có màu vàng khi chín”; trong Candidates, “banana” có xác suất cao nhất 0.85, khớp hoàn toàn với mô tả—vì thế đáp án chắc chắn là Quả chuối.
            Evidence2: Context nêu “chùm chuối” xuất hiện; Question hỏi trái nào “thường vàng khi chín”; KBs_knowledge khẳng định chuối chín sẽ đổi vỏ sang màu vàng; trong Candidates, “banana” có xác suất cao nhất 0.85 và khớp hoàn toàn với mô tả—vì vậy đáp án chính là “Quả chuối”.
            Evidence3: Context mô tả có một chùm chuối trên bàn. Question hỏi về loại quả có màu vàng khi chín. KBs_knowledge xác nhận chuối chuyển sang màu vàng. Đoạn văn Object_Analysis cũng mô tả chi tiết "chùm chuối với vỏ màu vàng đặc trưng khi chín". Dựa trên các bằng chứng này và xác suất cao nhất 0.85 trong Candidates, đáp án chính xác là "Chuối".
            Explanation: Bức ảnh cho thấy một nải chuối, đây là loại quả có vỏ màu vàng khi chín.
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

