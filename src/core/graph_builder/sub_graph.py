from typing import Dict, Any, Type
from langgraph.graph import StateGraph, END, START

from src.core.nodes.subgraph_node import tool_node, call_agent_node, final_reasoning_node, should_continue, rationale_node
from src.core.state import (    
    ViReJuniorState, 
    ViReSeniorState, 
    ViReManagerState,
    JuniorOutputState,
    SeniorOutputState,
    ManagerOutputState
)
from src.agents.strategies.junior_agent import JuniorAgent
from src.agents.strategies.senior_agent import SeniorAgent
from src.agents.strategies.manager_agent import ManagerAgent

class SubGraphBuilder:
    """Builder for individual agent subgraphs"""
    
    def __init__(self, tools_registry: Dict[str, Any]):
        self.tools_registry = tools_registry
    
    def create_agent_subgraph(self, state_class: Type, analyst_instance, output_state) -> StateGraph:
        """Create a subgraph for a specific agent type with analyst instance"""
        workflow = StateGraph(state_class, output=output_state)
        
        def agent_node(state, config):
            state["analyst"] = analyst_instance 
            return call_agent_node(state, config, self.tools_registry)
        
        def tools_node(state):
            return tool_node(state, self.tools_registry)
        
        def final_reasoning_with_analyst(state):
            return final_reasoning_node(state)
        
        # Add nodes
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tools_node)
        workflow.add_node("rationale", rationale_node)
        workflow.add_node("final_reasoning", final_reasoning_with_analyst)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add conditional edges
        workflow.add_conditional_edges("agent", should_continue, {
            "continue": "tools",
            "rationale": "rationale"
        })
        
        # Add edges
        workflow.add_edge("tools", "agent")
        workflow.add_edge("rationale", "final_reasoning")
        workflow.add_edge("final_reasoning", END)
        
        return workflow
    
    def create_junior_subgraph(self):
        """Create subgraph for Junior Analyst"""
        junior_analyst = JuniorAgent()
        workflow = self.create_agent_subgraph(ViReJuniorState, junior_analyst, JuniorOutputState)
        return workflow.compile()
    
    def create_senior_subgraph(self):
        """Create subgraph for Senior Analyst"""
        senior_analyst = SeniorAgent()
        workflow = self.create_agent_subgraph(ViReSeniorState, senior_analyst, SeniorOutputState)
        return workflow.compile()
    
    def create_manager_subgraph(self):
        """Create subgraph for Manager Analyst"""
        manager_analyst = ManagerAgent()
        workflow = self.create_agent_subgraph(ViReManagerState, manager_analyst, ManagerOutputState)
        return workflow.compile()

