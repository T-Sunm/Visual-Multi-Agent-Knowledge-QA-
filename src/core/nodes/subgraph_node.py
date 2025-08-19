from typing import Union, Dict, Any
import json
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import ToolMessage
from src.core.state import ViReJuniorState, ViReSeniorState, ViReManagerState
from src.models.llm_provider import get_llm
from src.utils.tools_utils import _process_knowledge_result
import re
from src.utils.image_processing import pil_to_base64
from src.utils.text_processing import extract_answer_from_result, remove_think_block

def tool_node(state: Union[ViReJuniorState, ViReSeniorState, ViReManagerState], 
              tools_registry: Dict[str, Any]) -> Dict[str, Any]:
    """Process tool calls and update state"""
    outputs = []
    tool_calls = getattr(state["messages"][-1], "tool_calls", [])
    count_of_tool_calls = state.get("count_of_tool_calls", 0)
    updates = {"messages": state["messages"] + outputs, "count_of_tool_calls": count_of_tool_calls + len(tool_calls)} # Số tool gọi trong 1 lần có thể nhiều hơn 1 nên không thể tăng 1 lần mà phải tăng số lần gọi tool

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        print("Agent: ", state["analyst"].name, "tool_name: ", tool_name, "args: ", tool_call["args"])
        try:
            if tool_name == "vqa_tool" or tool_name == "lm_knowledge" or tool_name == "analyze_image_object":
                tool_call['args']['image'] = state.get("image")
                tool_call["args"]["image"] = pil_to_base64(tool_call['args']['image']) 
                result = tools_registry[tool_name].invoke(tool_call["args"])
                if tool_name == "vqa_tool":
                    updates["answer_candidate"] = result
                elif tool_name == "lm_knowledge":
                    updates["lms_knowledge"] = [result]
                elif tool_name == "analyze_image_object":
                    updates["object_analysis"] = [result]
                
            elif tool_name in ["arxiv", "wikipedia"]:
                raw_result = tools_registry[tool_name].invoke(tool_call["args"])
                print(f"Agent: {state['analyst'].name} - Tool: {tool_name}")
                # Process and format the result
                processed_result = _process_knowledge_result(raw_result, tool_name)
                updates["kbs_knowledge"] = [processed_result]
            else:
                result = f"Unknown tool: {tool_name}"

            outputs.append(
                ToolMessage(
                    content=json.dumps(processed_result if 'processed_result' in locals() else result),
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                )
            )
        except Exception as e:
            print(f"Error processing tool {tool_name}: {e}")
            outputs.append(
                ToolMessage(
                    content=f"Error: {str(e)}",
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                )
            )
    return updates


def call_agent_node(state: Union[ViReJuniorState, ViReSeniorState, ViReManagerState],
                   config: RunnableConfig,
                   tools_registry: Dict[str, Any]) -> Dict[str, Any]:
    """Call the agent with appropriate tools"""
    tools = state["analyst"].tools
    tools = [tools_registry[tool] for tool in tools if tool in tools_registry]
    
    llm = get_llm(tools)
    # Auto-detect placeholders từ system prompt
    base_prompt = state["analyst"].system_prompt
    placeholders = re.findall(r'\{(\w+)\}', base_prompt)
    
    # Prepare available values
    format_values = {
        'question': state.get('question', ''),
        'context': state.get('image_caption', ''),
    }
    
    # Chỉ format với placeholders có trong prompt
    format_dict = {key: format_values[key] for key in placeholders if key in format_values}
    
    formatted_prompt = base_prompt.format(**format_dict)
    
    response = llm.invoke(formatted_prompt, config)
    
    return {
        "messages": [response],
        "analyst": state["analyst"]
    }

def rationale_node(state: Union[ViReJuniorState, ViReSeniorState, ViReManagerState]) -> Dict[str, Any]:
    """Rationale node to generate rationale"""
    llm = get_llm(temperature=0.1)
    format_values = {
            'context': state.get("image_caption", ""),
            'question': state.get("question", ""),
            'candidates': state.get("answer_candidate", ""),
            'KBs_Knowledge': "\n".join(state.get("kbs_knowledge", [])),
            'LMs_Knowledge': "\n".join(state.get("lms_knowledge", [])),
            'Object_Analysis': "\n".join(state.get("object_analysis", []))
    }
    return {
        "messages": [state["messages"][-1]],
        "analyst": state["analyst"]
    }

def final_reasoning_node(state: Union[ViReJuniorState, ViReSeniorState, ViReManagerState]) -> Dict[str, Any]:
    """Final reasoning node to synthesize results"""
    # Auto-detect placeholders từ final_system_prompt
    base_prompt = state["analyst"].final_system_prompt
    placeholders = re.findall(r'\{(\w+)\}', base_prompt)
        
    # Prepare available values
    format_values = {
            'context': state.get("image_caption", ""),
            'question': state.get("question", ""),
            'candidates': state.get("answer_candidate", ""),
            'KBs_Knowledge': "\n".join(state.get("kbs_knowledge", [])),
            'LMs_Knowledge': "\n".join(state.get("lms_knowledge", [])),
            'Object_Analysis': "\n".join(state.get("object_analysis", []))
    }
    print("agent: ", state["analyst"].name, "state: ", format_values)
    # Chỉ format với placeholders có trong prompt
    format_dict = {key: format_values[key] for key in placeholders if key in format_values}
    
    final_system_prompt = base_prompt.format(**format_dict)
    
    llm = get_llm(temperature=0.1)
    
    final_response = llm.invoke(final_system_prompt)
    cleaned_content = remove_think_block(final_response.content)
    answer, evidence = extract_answer_from_result(cleaned_content)

    return {
        "results": [{state["analyst"].name: answer}],
        "evidences": [{state["analyst"].name: evidence}]
    }


def should_continue(state: Union[ViReJuniorState, ViReSeniorState, ViReManagerState]) -> str:
    """Decide whether to continue with tools or move to final reasoning"""
    messages = state["messages"]
    last_message = messages[-1]
    
    count_of_tool_calls = state.get("count_of_tool_calls", 0)
    
    max_steps = {
        "Junior": 1,  
        "Senior": 2,     
        "Manager": 3    
    }.get(state.get("analyst", {}).name)
    
    if count_of_tool_calls >= max_steps:
        return "final_reasoning"
    # If no tool calls, go to final reasoning  
    if not getattr(last_message, "tool_calls", None):
        return "final_reasoning"
    # If has tool calls, continue with tools
    else:
        return "continue"
