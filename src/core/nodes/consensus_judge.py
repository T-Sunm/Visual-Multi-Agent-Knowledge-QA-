from src.agents.strategies.judge_agent import ConsensusJudgeAgent
from typing import Dict

def consensus_judge_node(state) -> Dict[str, str]:
    judge = ConsensusJudgeAgent()
    final_answer, explanation = judge(state["question"], state["final_answer"], state["rationales"])

    updates = {
        "final_answer": final_answer,
        "explanation": explanation
    }
    return updates
