from typing import Dict, Any
from langgraph.graph import StateGraph, END, START

from src.core.nodes.caption_node import caption_node
from src.core.state import ViReAgentState
from src.core.graph_builder.sub_graph import SubGraphBuilder
from src.core.nodes.voting_node import voting_node

def set_pre_phase(state: ViReAgentState) -> ViReAgentState:
    return state.update(phase="prevote")

def set_post_phase(state: ViReAgentState) -> ViReAgentState:
    return state.update(phase="postvote")
class MainGraphBuilder:
    """Builder for the main multi-agent workflow"""
    
    def __init__(self, tools_registry: Dict[str, Any]):
        self.tools_registry = tools_registry
        self.subgraph_builder = SubGraphBuilder(tools_registry)
        
    def create_main_workflow(self):
        main = StateGraph(ViReAgentState)

        # 1) Khai báo node
        main.add_node("caption", caption_node)

        main.add_node("set_pre_phase", set_pre_phase)
        main.add_node("set_post_phase", set_post_phase)

        main.add_node("junior_analyst", self.subgraph_builder.create_junior_subgraph())
        main.add_node("senior_analyst", self.subgraph_builder.create_senior_subgraph())
        main.add_node("manager_analyst", self.subgraph_builder.create_manager_subgraph())

        main.add_node("voting", voting_node)

        # 2) Luồng PRE-vote
        main.add_edge(START,          "caption")
        main.add_edge("caption",      "junior_analyst")
        main.add_edge("caption",      "senior_analyst")
        main.add_edge("caption",      "manager_analyst")


        main.add_conditional_edges(
            "junior_analyst",
            lambda s: "prevote" if s["phase"] == "prevote" else "postvote",
            {
                "prevote": "voting",   # lần 1 → voting
                "postvote": END        # lần 2 → dừng
            }
        )

        main.add_edge("senior_analyst",  "voting")
        main.add_edge("manager_analyst", "voting")
        main.add_edge("voting",          "junior_analyst")

        return main.compile()


