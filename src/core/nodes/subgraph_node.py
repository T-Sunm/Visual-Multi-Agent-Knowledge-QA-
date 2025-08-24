from typing import Union, Dict, Any
import json
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import ToolMessage, AIMessage
from src.core.state import ViReJuniorState, ViReSeniorState, ViReManagerState
from src.models.llm_provider import get_llm
from src.utils.tools_utils import _process_knowledge_result
import re
from src.utils.image_processing import pil_to_base64
from src.utils.text_processing import extract_answer, remove_think_block, extract_rationale

def tool_node(state: Union[ViReJuniorState, ViReSeniorState, ViReManagerState], 
              tools_registry: Dict[str, Any]) -> Dict[str, Any]:
    """Process tool calls and update state"""
    tool_calls = getattr(state["messages"][-1], "tool_calls", [])
    count_of_tool_calls = state.get("count_of_tool_calls", 0)
    updates = {"messages": [], "count_of_tool_calls": count_of_tool_calls + len(tool_calls)}

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
                processed_result = _process_knowledge_result(raw_result, tool_name)
                updates["kbs_knowledge"] = [processed_result]
            else:
                result = f"Unknown tool: {tool_name}"

            # Remove image data from result because it's too large
            result_content = processed_result if 'processed_result' in locals() else result
            if isinstance(result_content, dict) and 'image' in result_content:
                result_content = {k: v for k, v in result_content.items() if k != 'image'}
            
            updates["messages"].append(
                ToolMessage(
                    content=json.dumps(result_content),
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                )
            )
        except Exception as e:
            print(f"Error processing tool {tool_name}: {e}")
            tool_call_id = tool_call.get("id", f"call_{tool_name}_error_{len(updates['messages'])}")
            updates["messages"].append(
                ToolMessage(
                    content=f"Error: {str(e)}", 
                    name=tool_name,
                    tool_call_id=tool_call_id
                )
            )
    return updates


def call_agent_node(state: Union[ViReJuniorState, ViReSeniorState, ViReManagerState],
                   config: RunnableConfig,
                   tools_registry: Dict[str, Any]) -> Dict[str, Any]:
    """Call the agent with appropriate tools"""
    tools = state["analyst"].tools
    tools = [tools_registry[tool] for tool in tools if tool in tools_registry]
    
    llm = get_llm(with_tools=tools, temperature=0.2)
    base_prompt = state["analyst"].system_prompt
    placeholders = re.findall(r'\{(\w+)\}', base_prompt)
    
    # Expanded format_values to include current state information
    format_values = {
        'question': state.get('question', ''),
        'context': state.get('image_caption', ''),
        'count_of_tool_calls': state.get('count_of_tool_calls', 0),
        'answer_candidate': state.get('answer_candidate', ''),
        'kbs_knowledge': "\n".join(state.get('kbs_knowledge', [])),
        'object_analysis': "\n".join(state.get('object_analysis', [])),
        'lms_knowledge': "\n".join(state.get('lms_knowledge', [])),
    }
    
    format_dict = {key: format_values[key] for key in placeholders if key in format_values}
    formatted_prompt = base_prompt.format(**format_dict)

    messages = [{"role": "system", "content": formatted_prompt}]
    response = llm.invoke(messages, config)
    cleaned_content = remove_think_block(response.content)
    
    # Create clean AIMessage preserving tool_calls
    cleaned_response = AIMessage(
        content=cleaned_content,
        tool_calls=getattr(response, 'tool_calls', [])
    )
    
    return {
        "messages": [cleaned_response],
        "analyst": state["analyst"]
    }

def rationale_node(state: Union[ViReJuniorState, ViReSeniorState, ViReManagerState]) -> Dict[str, Any]:
    """Rationale node to generate rationale"""
    llm = get_llm(temperature=0.7)
    format_values = {
            'context': state.get("image_caption", ""),
            'question': state.get("question", ""),
            'candidates': state.get("answer_candidate", ""),
            'KBs_Knowledge': "\n".join(state.get("kbs_knowledge", [])),
            'LMs_Knowledge': "\n".join(state.get("lms_knowledge", [])),
            'Object_Analysis': "\n".join(state.get("object_analysis", []))
    }

    rationale_system_prompt = state["analyst"].rationale_system_prompt.format(**format_values)
    rationale_response = llm.invoke(rationale_system_prompt)
    
    cleaned_content = remove_think_block(rationale_response.content)
    rationale = extract_rationale(cleaned_content)

    # Create clean AIMessage with only content
    rationale_message = AIMessage(content=rationale)

    return {
        "messages": [rationale_message],
        "rationales": [{state["analyst"].name: rationale}]
    }

def final_reasoning_node(state: Union[ViReJuniorState, ViReSeniorState, ViReManagerState]) -> Dict[str, Any]:
    """Final reasoning node to synthesize results"""
    # Auto-detect placeholders từ final_system_prompt
    base_prompt = state["analyst"].final_system_prompt
    placeholders = re.findall(r'\{(\w+)\}', base_prompt)

    format_values = {
            'context': state.get("image_caption", ""),
            'question': state.get("question", ""),
            'candidates': state.get("answer_candidate", ""),
            'rationale': state.get("rationales", [])[0].get(state["analyst"].name, "")
    }

    print("agent: ", state["analyst"].name, "state: ", format_values)
    # Chỉ format với placeholders có trong prompt
    format_dict = {key: format_values[key] for key in placeholders if key in format_values}
    
    final_system_prompt = base_prompt.format(**format_dict)
    
    llm = get_llm(temperature=0.7)
    
    final_response = llm.invoke(final_system_prompt)
    cleaned_content = remove_think_block(final_response.content)
    answer = extract_answer(cleaned_content)
    print("agent: ", state["analyst"].name, "answer: ", answer)
    return {
        "results": [{state["analyst"].name: answer}]
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
        return "rationale"
    # If no tool calls, go to final reasoning  
    if not getattr(last_message, "tool_calls", None):
        return "rationale"
    # If has tool calls, continue with tools
    else:
        return "continue"
