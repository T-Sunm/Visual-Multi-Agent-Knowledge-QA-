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
    
    #---- Results ----#
    results: Annotated[List[Dict[str, str]], operator.add]
    final_answer: str
    voting_details: Dict[str, Any]

    #---- Explanation ----#
    explanation: str
    rationales: Annotated[List[Dict[str, str]], operator.add]

class ViReJuniorState(MessagesState):
    question: str
    image: Union[str, Image.Image]
    analyst: JuniorAgent
    count_of_tool_calls: int
    image_caption: str
    answer_candidate: str
    
    # Results
    results: Dict[str, str]
    final_answer: str

    #---- Rationale ----#
    rationales: Dict[str, str]

class ViReSeniorState(MessagesState):
    question: str
    image: Union[str, Image.Image]
    analyst: SeniorAgent
    image_caption: str
    count_of_tool_calls: int
    answer_candidate: str
    kbs_knowledge: Annotated[List[str], operator.add]
    # Results
    results: Dict[str, str]

    #---- Rationale ----#
    rationales: Dict[str, str]

class ViReManagerState(MessagesState):
    question: str
    image: Union[str, Image.Image]
    analyst: ManagerAgent
    image_caption: str
    count_of_tool_calls: int
    answer_candidate: str
    kbs_knowledge: Annotated[List[str], operator.add]
    object_analysis: Annotated[List[str], operator.add]
    lms_knowledge: Annotated[List[str], operator.add]
    # Results
    results: Dict[str, str]

    #---- Rationale ----#
    rationales: Dict[str, str]

class JuniorOutputState():
    results: Optional[Dict[str, str]] = None
    rationales: Optional[Dict[str, str]] = None

class SeniorOutputState():
    results: Optional[Dict[str, str]] = None
    rationales: Optional[Dict[str, str]] = None

class ManagerOutputState():
    results: Optional[Dict[str, str]] = None
    rationales: Optional[Dict[str, str]] = None
