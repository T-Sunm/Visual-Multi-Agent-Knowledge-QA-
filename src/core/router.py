from typing import Dict, Any, List
from langgraph.types import Send
from src.core.state import ViReAgentState

def route_to_analysts(state: ViReAgentState) -> Dict[str, Any]:
    """Router using Send API for parallel distribution"""
    sends = []
    
    for analyst in state["analysts"]:
        # Create separate state for each analyst
        analyst_state = {
            "question": state["question"],
            "image": state["image"],
            "analyst": analyst,
            "number_of_steps": 0,
            "messages": []
        }
        
        # Send to appropriate subgraph
        if "junior" in analyst.name.lower():
            sends.append(Send("junior_subgraph", analyst_state))
        elif "senior" in analyst.name.lower():
            sends.append(Send("senior_subgraph", analyst_state))
        elif "manager" in analyst.name.lower():
            sends.append(Send("manager_subgraph", analyst_state))

    return {"send": sends}
