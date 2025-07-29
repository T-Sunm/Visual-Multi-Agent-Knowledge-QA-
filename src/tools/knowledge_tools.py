from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
)
from langchain_community.utilities import (
    ArxivAPIWrapper,
    WikipediaAPIWrapper,
)

arxiv_wrapper = ArxivAPIWrapper(
    top_k_results=1, 
    arxiv_search=None,
    arxiv_exceptions=None
)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=1, wiki_client=None)
wikipedia = WikipediaQueryRun(
    api_wrapper=wikipedia_wrapper,
    description="Search for information on a given topic using Wikipedia"
)


@rate_limiter.rate_limit("arxiv", ARXIV_DELAY)
def search_arxiv(query: str) -> str:
    """Search for information on a given topic using Arxiv"""
    return arxiv.run(query)

@rate_limiter.rate_limit("wikipedia", WIKIPEDIA_DELAY)
def search_wikipedia(query: str) -> str:
    """Search for information on a given topic using Wikipedia"""
    return wikipedia.invoke(query)



