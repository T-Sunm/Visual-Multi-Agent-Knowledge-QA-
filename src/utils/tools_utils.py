def _process_knowledge_result(raw_result: str, tool_name: str) -> str:
    """Process knowledge base results for clean formatting"""
    if tool_name == "arxiv":
        return _process_arxiv_result(raw_result)
    elif tool_name == "wikipedia":
        return _process_wikipedia_result(raw_result)
    else:
        return raw_result

def _process_arxiv_result(raw_result: str) -> str:
    """Extract key info from arxiv results"""
    lines = raw_result.strip().split('\n')
    papers = []
    current_paper = {}
    
    for line in lines:
        if line.startswith("Published:"):
            if current_paper:
                papers.append(current_paper)
            current_paper = {"published": line.replace("Published: ", "")}
        elif line.startswith("Title:"):
            current_paper["title"] = line.replace("Title: ", "")
        elif line.startswith("Summary:"):
            current_paper["summary"] = line.replace("Summary: ", "")
    
    if current_paper:
        papers.append(current_paper)
    
    # Format for final prompt
    formatted = []
    for paper in papers[:2]:  # Limit to 2 papers
        formatted.append(f"Paper: {paper.get('title', 'Unknown')}\nSummary: {paper.get('summary', 'No summary')[:300]}...")
    
    return "\n\n".join(formatted)

def _process_wikipedia_result(raw_result: str) -> str:
    """Extract key info from wikipedia results"""
    lines = raw_result.strip().split('\n')
    pages = []
    current_page = {}
    
    for line in lines:
        if line.startswith("Page:"):
            if current_page:
                pages.append(current_page)
            current_page = {"page": line.replace("Page: ", "")}
        elif line.startswith("Summary:"):
            current_page["summary"] = line.replace("Summary: ", "")
    
    if current_page:
        pages.append(current_page)
    
    # Format for final prompt
    formatted = []
    for page in pages[:2]:  # Limit to 2 pages
        formatted.append(f"Wikipedia - {page.get('page', 'Unknown')}: {page.get('summary', 'No summary')[:300]}...")
    
    return "\n\n".join(formatted)