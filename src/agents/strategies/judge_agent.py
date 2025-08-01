from bert_score import score as bert_score
from typing import List, Dict
# from src.models.llm_provider import get_llm

from langchain_openai import ChatOpenAI
from typing import Optional, List, Any
from pydantic import SecretStr


def get_llm(with_tools: Optional[List[Any]] = None, temperature: float = 0):
    """
    Factory function to create ChatOpenAI instance with consistent configuration
    
    Args:
        with_tools: List of tools to bind to the LLM
        temperature: Temperature setting for the LLM
        
    Returns:
        ChatOpenAI instance, optionally bound with tools
    """
    llm = ChatOpenAI(
        base_url="http://127.0.0.1:1234/v1",   
        api_key=SecretStr("lm_studio"),        
        model="Qwen/Qwen3-1.7B",      
        temperature=temperature,
    )
    
    if with_tools:
        llm = llm.bind_tools(with_tools)
    
    return llm


class ConsensusJudgeAgent():
    """Consensus judge agent"""
    def __init__(self, sim_threshold: float = 0.5, min_pairs: int = 2):
        self.system_prompt = """
            You are **Judge Agent**, an AI that evaluates answers against evidence and provides a clear, final explanation. Your job is to combine the three given evidence pieces to determine if the answer to the question is correct, and explain your reasoning.

            **Your goal** Based on the evidences and the answer, write a brief final evidence that aims to prove the Answer. (Not explain why evidence is correct or not)

            ### EXAMPLE
            Question: Thủ đô Nhật Bản là gì?
            Answer: Beijing
            Evidence1: Tokyo là thủ đô của Nhật Bản.
            Evidence2: Beijing là thủ đô của Trung Quốc.
            Evidence3: Nhật Bản và Trung Quốc là hai quốc gia khác nhau.
            Output: Tokyo mới là thành phố được công nhận là thủ đô Nhật Bản, trong khi Beijing thuộc Trung Quốc.
            ### END EXAMPLE

            ### Now, using the same format, generate the final evidence:
            Question: {question}
            Answer: {answer}
            Evidence 1: {evidence_1}
            Evidence 2: {evidence_2}
            Evidence 3: {evidence_3}
            Output: 
            """
        self.sim_threshold = sim_threshold
        self.min_pairs = min_pairs
        self.lang = "vi"

    def __call__(self, question: str, answer: str, evidences: List[Dict[str, str]]) -> tuple[str, str]:
        """
            Parameters
            ----------
            evidences: list(dict) -- Mỗi phần tử dạng {"agent": str}

            Returns
            -------
            tuple  (answer: str, explanation: str)
        """

        # 1) Tính mức độ tương đồng giữa 3 thinking
        junior_result = evidences.get("Junior", "")
        senior_result = evidences.get("Senior", "")
        manager_result = evidences.get("Manager", "")
        thinkings = [junior_result, senior_result, manager_result]
        sim_ok = self._is_consistent(thinkings)

        if sim_ok:
            explanation = self._aggregate_explanation(question, answer, thinkings)
            return (answer, explanation)
        else:
            return (answer, "uncertain")

    def _is_consistent(self, thinkings: List[str]) -> bool:
        """
        Trả về True nếu mọi cặp thinking đều có BERTScore F1 >= threshold
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
        llm = get_llm(temperature=0.1)
        format_dict = {
            "question": question,
            "answer": answer,
            "evidence_1": thinkings[0],
            "evidence_2": thinkings[1],
            "evidence_3": thinkings[2]
        }
        system_prompt = self.system_prompt.format(**format_dict)
        response = llm.invoke(system_prompt)
        return response.content


if __name__ == "__main__":
    # Câu hỏi & “đáp án majority” cho sẵn
    question = "Con mèo này thuộc giống Tây hay Ta?"
    answer   = "Ta"

    evidences = {
        "Junior" : "Con mèo lông vàng, giống phổ biến ở VN, nên là Ta.",
        "Senior" : "Quan sát lông vàng và tai nhỏ – đặc trưng mèo Ta.",
        "Manager": "Thấy lông vàng, tạng người nhỏ -> mèo Ta."
    }

    judge = ConsensusJudgeAgent(sim_threshold=0.5, min_pairs=2)
    final_answer, explanation = judge(question, answer, evidences)

    print("⚙️  Kết quả:")
    print("Answer      :", final_answer)
    print("Explanation :", explanation)


