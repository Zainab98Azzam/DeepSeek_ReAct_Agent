import os
from tavily import TavilyClient
from .base_tool import BaseTool

class SearchAgent(BaseTool):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

    def use(self, query: str) -> str:
        """
        Performs a targeted web search for the given query on trusted medical sites.
        """
        try:
            # We'll configure this to search specific sites
            response = self.tavily_client.search(
                query=query,
                # Add the specific domains from your prompt
                domains=[
                    "pubmed.ncbi.nlm.nih.gov",
                    "clinicaltrials.gov",
                    "open.fda.gov",
                    "accessdata.fda.gov",
                    "drugs.com",
                    "nimh.nih.gov",
                    "aap.org",
                    "cdc.gov",
                    "nice.org.uk",
                    "uptodate.com"
                ]
            )
            if response and response.get('results'):
                return response['results'][0]['content']
            return "No search results found from specified domains."
        except Exception as e:
            return f"An error occurred during search: {e}"