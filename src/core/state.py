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
    evidences: Annotated[List[Dict[str, str]], operator.add]

class ViReJuniorState(MessagesState):
    question: str
    image: Union[str, Image.Image]
    analyst: JuniorAgent
    count_of_tool_calls: int
    image_caption: str
    answer_candidate: str
    
    rationale: str
    # Results
    results: Dict[str, str]
    final_answer: str

    #---- Evidence ----#
    evidences: Dict[str, str]

class ViReSeniorState(MessagesState):
    question: str
    image: Union[str, Image.Image]
    analyst: SeniorAgent
    image_caption: str
    count_of_tool_calls: int
    answer_candidate: str
    kbs_knowledge: Annotated[List[str], operator.add]
    rationale: str
    # Results
    results: Dict[str, str]

    #---- Evidence ----#
    evidences: Dict[str, str]

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
    rationale: str
    # Results
    results: Dict[str, str]

    #---- Evidence ----#
    evidences: Dict[str, str]

class JuniorOutputState(MessagesState):
    results: Optional[Dict[str, str]] = None
    evidences: Optional[Dict[str, str]] = None

class SeniorOutputState(MessagesState):
    results: Optional[Dict[str, str]] = None
    evidences: Optional[Dict[str, str]] = None

class ManagerOutputState(MessagesState):
    results: Optional[Dict[str, str]] = None
    evidences: Optional[Dict[str, str]] = None
