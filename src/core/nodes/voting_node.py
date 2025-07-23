from typing import Dict, Any, List, Tuple
from collections import Counter
import re
from src.utils.text_processing import extract_answer_from_result

def normalize_answer_for_voting(answer: str) -> str:
    """
    Normalize answer for voting by extracting the core answer from various formats.
    
    Examples:
    - "10(0.98)" -> "10"
    - "Candidates: 10(0.99), 9(0.92)" -> "10"
    - "The answer is 10" -> "10"
    - "10" -> "10"
    """
    if not answer:
        return ""
    
    answer = str(answer).strip()
    
    # Extract from VQA format: "10(0.98)" -> "10"
    match = re.match(r'^(\w+)\s*\(\d+\.\d+\)', answer)
    if match:
        return match.group(1).strip()
    
    # Extract from candidates format: "Candidates: 10(0.99), 9(0.92)" -> "10"
    candidates_match = re.search(r'Candidates:\s*(\w+)\s*\(\d+\.\d+\)', answer)
    if candidates_match:
        return candidates_match.group(1).strip()
    
    # Extract from "The answer is X" format
    answer_match = re.search(r'the\s+answer\s+is\s+(\w+)', answer.lower())
    if answer_match:
        return answer_match.group(1).strip()
    
    # If it's just a simple answer, return as is
    return answer

def voting_function(junior_answer: str, senior_answer: str, manager_answer: str) -> Tuple[str, Dict[str, int]]:
    """
    Weighted voting function implementing AF = Voting(AJ[w1], AS[w2], AM[w3])
    
    Args:
        junior_answer: Answer from Junior agent (weight = 2)
        senior_answer: Answer from Senior agent (weight = 3)  
        manager_answer: Answer from Manager agent (weight = 4)
        
    Returns:
        Tuple of (final_answer, vote_breakdown)
    """
    # Define weights according to paper
    weights = {
        'junior': 2,
        'senior': 3,
        'manager': 4
    }
    
    # Count votes for each unique answer
    vote_counts = Counter()
    
    # Add weighted votes
    if junior_answer:
        vote_counts[junior_answer] += weights['junior']
        
    if senior_answer:
        vote_counts[senior_answer] += weights['senior']
        
    if manager_answer:
        vote_counts[manager_answer] += weights['manager']
    
    if vote_counts:
        final_answer = vote_counts.most_common(1)[0][0]
        return final_answer, dict(vote_counts)
    
    # Fallback if no valid answers
    return "", {}

def voting_node(state) -> Dict[str, Any]:
    """
    Voting node that implements weighted voting mechanism from paper.
    
    Process:
    1. Extract answers from each agent's results
    2. Apply weighted voting: Junior(2), Senior(3), Manager(4)
    3. Select answer with highest vote count
    4. Return final answer and voting details
    """
    
    # Extract results from agents
    # Note: results are accumulated in order [junior, senior, manager]
    results = state.get("results", [])
    
    if len(results) < 3:
        print(f"Warning: Expected 3 agent results, got {len(results)}")
        return {
            "final_answer": "",
            "voting_details": {
                "error": f"Insufficient results: expected 3, got {len(results)}"
            }
        }
    
    # convert list of dict to dict
    agent_results = {k: v for d in results for k, v in d.items()}
        
    # Extract answers from each agent by name
    junior_result = agent_results.get("Junior", "")
    senior_result = agent_results.get("Senior", "")
    manager_result = agent_results.get("Manager", "")
    junior_answer = normalize_answer_for_voting(extract_answer_from_result(junior_result))
    senior_answer = normalize_answer_for_voting(extract_answer_from_result(senior_result))
    manager_answer = normalize_answer_for_voting(extract_answer_from_result(manager_result))
    
    # Apply weighted voting
    final_answer, vote_breakdown = voting_function(
        junior_answer, senior_answer, manager_answer
    )
    
    # Create detailed voting information
    voting_details = {
        "agent_answers": {
            "junior": {"answer": junior_answer, "weight": 2},
            "senior": {"answer": senior_answer, "weight": 3},
            "manager": {"answer": manager_answer, "weight": 4}
        },
        "vote_breakdown": vote_breakdown,
        "final_answer": final_answer,
        "total_votes": sum(vote_breakdown.values()) if vote_breakdown else 0
    }
    
    print("\nVOTING BREAKDOWN:")
    for answer, votes in vote_breakdown.items():
        print(f"  Answer '{answer}': {votes} votes")
    
    print(f"\nFINAL SELECTED ANSWER: '{final_answer}'")
    print("=" * 50)
    
    return {
        "final_answer": final_answer,
        "voting_details": voting_details,
        "phase": "postvote"
    }

# def weighted_voting_example():
#     """
#     Example demonstrating the voting mechanism with realistic data structure.
    
#     Simulates the actual data structure after all agents complete their analysis.
#     Example scenario:
#     - Junior says "cat": 2 votes
#     - Senior says "dog": 3 votes  
#     - Manager says "dog": 4 votes
#     → "dog" wins with 7 votes vs "cat" with 2 votes
#     """
    
#     # Simulate realistic state after graph completion
#     # This mimics what voting_node actually receives
#     dummy_state = {
#         "results": [
#             {"Junior": "cat"},
#             {"Senior": "dog"}, 
#             {"Manager": "dog"}
#         ]
#     }
    
#     print("Simulated results from agents:")
#     for i, result_dict in enumerate(dummy_state["results"]):
#         for agent_name, response in result_dict.items():
#             print(f"  {agent_name}: '{response}'")
    
#     # Process using actual voting_node logic
#     print("\nProcessing through voting_node logic:")
    
#     # Convert list of dict to dict (same as voting_node)
#     agent_results = {k: v for d in dummy_state["results"] for k, v in d.items()}
    
#     # Extract answers (same logic as voting_node)
#     junior_result = agent_results.get("Junior", "")
#     senior_result = agent_results.get("Senior", "")
#     manager_result = agent_results.get("Manager", "")
    
#     junior_answer = extract_answer_from_result(junior_result)
#     senior_answer = extract_answer_from_result(senior_result)
#     manager_answer = extract_answer_from_result(manager_result)
    
#     print(f"  Extracted Junior answer: '{junior_answer}' (Weight: 2)")
#     print(f"  Extracted Senior answer: '{senior_answer}' (Weight: 3)")
#     print(f"  Extracted Manager answer: '{manager_answer}' (Weight: 4)")
    
#     # Apply weighted voting
#     final_answer, vote_breakdown = voting_function(
#         junior_answer, senior_answer, manager_answer
#     )
    
#     print(f"\nVote breakdown: {vote_breakdown}")
#     print(f"Winner: '{final_answer}' with {vote_breakdown.get(final_answer, 0)} total votes")
#     print("-" * 40)
    
#     return final_answer, vote_breakdown

# if __name__ == "__main__":
#     # Run example
#     weighted_voting_example()
