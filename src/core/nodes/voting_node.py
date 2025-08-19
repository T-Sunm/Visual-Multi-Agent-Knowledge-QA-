from typing import Dict, Any, Tuple
from collections import Counter
import re

def normalize_answer_for_voting(answer: str) -> str:
    """
    Normalize answer for voting by extracting the core answer from various formats.
    
    Examples:
    - "10(0.98)" -> "10"
    - "Candidates: 10(0.99), 9(0.92)" -> "10"
    - "The answer is 10" -> "10"
    - "10" -> "10"
    - "Có\\n\\nRationale: ..." -> "có" 
    """
    if not answer:
        return ""
    
    answer_str = str(answer).strip()
    
    # Take only the first line in case rationale is appended
    core_answer = answer_str.split('\n')[0].strip()
    
    # Extract from VQA format: "10(0.98)" -> "10"
    match = re.match(r'^(\w+)\s*\(\d+\.\d+\)', core_answer)
    if match:
        return match.group(1).strip().lower()
    
    # Extract from candidates format: "Candidates: 10(0.99), 9(0.92)" -> "10"
    candidates_match = re.search(r'Candidates:\s*(\w+)\s*\(\d+\.\d+\)', core_answer)
    if candidates_match:
        return candidates_match.group(1).strip().lower()
    
    # Extract from "The answer is X" format
    answer_match = re.search(r'the\s+answer\s+is\s+(\w+)', core_answer.lower())
    if answer_match:
        return answer_match.group(1).strip() # Already lowercase
    
    # If it's just a simple answer, return it in lowercase
    return core_answer.lower()

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
    
    # convert list of dict to dict
    agent_results = {k: v for d in results for k, v in d.items()}
        
    # Extract answers from each agent by name
    junior_result = agent_results.get("Junior", "")
    senior_result = agent_results.get("Senior", "")
    manager_result = agent_results.get("Manager", "")

    junior_answer = normalize_answer_for_voting(junior_result)
    senior_answer = normalize_answer_for_voting(senior_result)
    manager_answer = normalize_answer_for_voting(manager_result)
    
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
    
    updates = {
        "final_answer": final_answer,
        "voting_details": voting_details
    }
    return updates
