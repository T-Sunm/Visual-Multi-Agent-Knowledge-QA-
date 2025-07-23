from typing import Annotated, List, Dict, Any
from langgraph.graph import MessagesState
from src.agents.strategies.junior_agent import JuniorAgent
from src.agents.strategies.senior_agent import SeniorAgent
from src.agents.strategies.manager_agent import ManagerAgent
import operator
from PIL import Image
from typing import Union
from typing import Optional
class ViReAgentState(MessagesState):
    question: str
    image: Union[str, Image.Image]
    image_caption: str
    results: Annotated[List[Dict[str, str]], operator.add]
    final_answer: str
    voting_details: Dict[str, Any]
    phase: str
    explanation: str
    final_kbs_knowledge: str
    final_lms_knowledge: str

class ViReJuniorState(MessagesState):
    question: str
    image: Union[str, Image.Image]
    analyst: JuniorAgent
    phase: str
    number_of_steps: int
    image_caption: str
    answer_candidate: str
    
    # Explanation
    explanation: str
    final_kbs_knowledge: str
    final_lms_knowledge: str

    # Results
    results: Dict[str, str]
class ViReSeniorState(MessagesState):
    question: str
    image: Union[str, Image.Image]
    analyst: SeniorAgent
    image_caption: str
    number_of_steps: int
    answer_candidate: str

    # Knowledge
    KBs_Knowledge: Annotated[List[str], operator.add]
    final_kbs_knowledge: str
    
    # Results
    results: Dict[str, str]
class ViReManagerState(MessagesState):
    question: str
    image: Union[str, Image.Image]
    analyst: ManagerAgent
    image_caption: str
    number_of_steps: int
    answer_candidate: str

    # Knowledge
    KBs_Knowledge: Annotated[List[str], operator.add]
    LMs_Knowledge: Annotated[List[str], operator.add]
    final_lms_knowledge: str
    
    # Results
    results: Dict[str, str]


class JuniorOutputState(MessagesState):
    phase: str
    results: Optional[Dict[str, str]] = None
    explanation: Optional[str] = None

class SeniorOutputState(MessagesState):
    results: Dict[str, str]
    final_kbs_knowledge: str

class ManagerOutputState(MessagesState):
    results: Dict[str, str]
    final_lms_knowledge: str