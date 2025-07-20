from typing import Dict, Any, Optional
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver

class SessionMemory:
    
    def __init__(self, checkpointer: Optional[BaseCheckpointSaver] = None):
        self.checkpointer = checkpointer or MemorySaver()
    
    def create_thread_config(self, thread_id: str = "default") -> Dict[str, Any]:
        """Create thread configuration for session"""
        return {"configurable": {"thread_id": thread_id}}
    
    def get_checkpointer(self) -> BaseCheckpointSaver:
        """Get the checkpointer instance"""
        return self.checkpointer

session_memory = SessionMemory()