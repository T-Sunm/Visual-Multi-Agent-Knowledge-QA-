from typing import Union, Dict, Any
import json
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import ToolMessage
from src.core.state import ViReJuniorState, ViReSeniorState, ViReManagerState
from src.models.llm_provider import get_llm
from src.utils.tools_utils import _process_knowledge_result
import re
from src.utils.image_processing import pil_to_base64
from src.utils.text_processing import extract_answer_from_result

def tool_node(state: Union[ViReJuniorState, ViReSeniorState, ViReManagerState], 
              tools_registry: Dict[str, Any]) -> Dict[str, Any]:
    """Process tool calls and update state"""
    outputs = []
    tool_calls = getattr(state["messages"][-1], "tool_calls", [])
    
    updates = {"messages": outputs}

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        try:
            if tool_name == "vqa_tool" or tool_name == "lm_knowledge":
                tool_call['args']['image'] = state.get("image")
                tool_call["args"]["image"] = pil_to_base64(tool_call['args']['image']) 
                result = tools_registry[tool_name].invoke(tool_call["args"])
                print(f"Agent: {state['analyst'].name} - Tool: {tool_name}")
                if tool_name == "vqa_tool":
                    updates["answer_candidate"] = result
                elif tool_name == "lm_knowledge":
                    updates["LMs_Knowledge"] = [result]
                
            elif tool_name in ["arxiv", "wikipedia"]:
                raw_result = tools_registry[tool_name].invoke(tool_call["args"])
                print(f"Agent: {state['analyst'].name} - Tool: {tool_name}")
                # Process and format the result
                processed_result = _process_knowledge_result(raw_result, tool_name)
                updates["KBs_Knowledge"] = [processed_result]
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
        "messages": state["messages"] + [response],
        "analyst": state["analyst"],
        "number_of_steps": state.get("number_of_steps", 0) + 1
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
            'KBs_Knowledge': "\n".join(state.get("KBs_Knowledge", [])),
            'LMs_Knowledge': "\n".join(state.get("LMs_Knowledge", []))
    }
    
    # Chỉ format với placeholders có trong prompt
    format_dict = {key: format_values[key] for key in placeholders if key in format_values}
    
    final_system_prompt = base_prompt.format(**format_dict)
    
    llm = get_llm(temperature=0.1)
    
    final_response = llm.invoke(final_system_prompt)
    print(f"Final response: {final_response.content}")
    answer, evidence = extract_answer_from_result(final_response.content)
    # Return logic for each agent
    # Return logic for each agent
    return {
        "results": [{state["analyst"].name: answer}],
        "evidences": [{state["analyst"].name: evidence}]
    }


def should_continue(state: Union[ViReJuniorState, ViReSeniorState, ViReManagerState]) -> str:
    """Decide whether to continue with tools or move to final reasoning"""
    messages = state["messages"]
    last_message = messages[-1]
    
    number_of_steps = state.get("number_of_steps", 0)
    
    max_steps = {
        "Junior": 1,  
        "Senior": 2,     
        "Manager": 3    
    }.get(state.get("analyst", {}).name)
    
    if number_of_steps >= max_steps:
        return "final_reasoning"
    # If no tool calls, go to final reasoning  
    if not getattr(last_message, "tool_calls", None):
        return "final_reasoning"
    # If has tool calls, continue with tools
    else:
        return "continue"
