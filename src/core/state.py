from typing import Annotated, List, Dict, Any
from langgraph.graph import MessagesState
from src.agents.strategies.junior_agent import JuniorAgent
from src.agents.strategies.senior_agent import SeniorAgent
from src.agents.strategies.manager_agent import ManagerAgent
import operator
from PIL import Image
from typing import Union

class ViReAgentState(MessagesState):
    question: str
    image: Union[str, Image.Image]
    image_caption: str
    results: Annotated[List[Dict[str, str]], operator.add]
    final_answer: str
    voting_details: Dict[str, Any]


class ViReJuniorState(MessagesState):
    question: str
    image: Union[str, Image.Image]
    analyst: JuniorAgent
    image_caption: str
    number_of_steps: int
    answer_candidate: str
    results: Dict[str, str]

class ViReSeniorState(MessagesState):
    question: str
    image: Union[str, Image.Image]
    analyst: SeniorAgent
    image_caption: str
    number_of_steps: int
    answer_candidate: str
    KBs_Knowledge: Annotated[List[str], operator.add]
    results: Dict[str, str]
class ViReManagerState(MessagesState):
    question: str
    image: Union[str, Image.Image]
    analyst: ManagerAgent
    image_caption: str
    number_of_steps: int
    answer_candidate: str
    KBs_Knowledge: Annotated[List[str], operator.add]
    LLM_Knowledge: str
    results: Dict[str, str]


class SubgraphOutputState(MessagesState):
    results: Dict[str, str]