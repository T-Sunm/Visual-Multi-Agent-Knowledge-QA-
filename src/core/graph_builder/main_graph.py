from typing import Dict, Any
from langgraph.graph import StateGraph, END, START

from src.core.nodes.caption_node import caption_node
from src.core.state import ViReAgentState
from src.core.graph_builder.sub_graph import SubGraphBuilder
from src.core.nodes.voting_node import voting_node

class MainGraphBuilder:
    """Builder for the main multi-agent workflow"""
    
    def __init__(self, tools_registry: Dict[str, Any]):
        self.tools_registry = tools_registry
        self.subgraph_builder = SubGraphBuilder(tools_registry)
        
    def create_main_workflow(self):
        """Create the main multi-agent workflow"""
        
        
        main_workflow = StateGraph(ViReAgentState)

        junior_node = self.subgraph_builder.create_junior_subgraph()
        senior_node = self.subgraph_builder.create_senior_subgraph()
        manager_node = self.subgraph_builder.create_manager_subgraph()
    
        # Add nodes
        main_workflow.add_node("caption", caption_node)
        main_workflow.add_node("junior_analyst", junior_node)
        main_workflow.add_node("senior_analyst", senior_node)
        main_workflow.add_node("manager_analyst", manager_node)
        main_workflow.add_node("voting", voting_node)

        main_workflow.add_edge(START, "caption")
        main_workflow.add_edge("caption", "junior_analyst")
        main_workflow.add_edge("caption", "senior_analyst")
        main_workflow.add_edge("caption", "manager_analyst")

        main_workflow.add_edge("junior_analyst", "voting")
        main_workflow.add_edge("senior_analyst", "voting")
        main_workflow.add_edge("manager_analyst", "voting")

        main_workflow.add_edge("voting", END)
        
        return main_workflow.compile()


