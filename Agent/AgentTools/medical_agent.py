from typing import List, Optional
import json
import os
from pytrials.client import ClinicalTrials
from pymed import PubMed
from .base_tool import BaseTool # Assuming BaseTool is in the same parent directory structure
from dotenv import load_dotenv

load_dotenv()

# Install these libraries first: pip install pytrials pymed
class MedicalAgent(BaseTool):
    """
    A specialized tool for medical research, accessing PubMed for scientific literature 
    and ClinicalTrials.gov for clinical studies.
    """

    def __init__(self):
        # BaseTool currently has an empty __init__, so we only set up clients here.
        super().__init__() 
        
        # Initialize clients for both APIs
        self.ct_client = ClinicalTrials()
        # PubMed requires a tool name and email for API compliance
        self.pubmed_client = PubMed(tool="MedicalAgent", email=os.environ.get("AGENT_EMAIL", "agent@example.com")) 

    def search_pubmed(self, query: str, max_results: int = 2) -> List[str]:
        """Fetches the title, date, and abstract for recent PubMed articles."""
        articles = []
        try:
            results = self.pubmed_client.query(query, max_results=max_results)
            for article in results:
                title = getattr(article, 'title', 'No Title')
                abstract = getattr(article, 'abstract', 'No Abstract')
                date = getattr(article, 'publication_date', 'No Date')
                
                snippet = abstract[:200] + '...' if len(abstract) > 200 else abstract
                
                articles.append(f"ARTICLE: {title}\nDATE: {date}\nSNIPPET: {snippet}\n---")
        except Exception as e:
            articles.append(f"Error fetching PubMed articles: {e}")

        return articles

    def search_clinical_trials(self, query: str, max_studies: int = 2) -> List[str]:
        """Fetches key fields from recent ClinicalTrials.gov studies."""
        trials = []
        try:
            fields = ["NCTId", "BriefTitle", "OverallStatus", "Phase", "StartDate"]
            data = self.ct_client.get_study_fields(
                search_expr=query, 
                fields=fields, 
                max_studies=max_studies, 
                fmt='json'
            )

            if data and isinstance(data, dict) and 'studies' in data:
                for study in data['studies']:
                    study_info = study.get('protocolSection', {}).get('identificationModule', {})
                    status_info = study.get('protocolSection', {}).get('statusModule', {})
                    design_info = study.get('protocolSection', {}).get('designModule', {})
                    
                    nct_id = study_info.get('nctId', 'N/A')
                    title = study_info.get('briefTitle', 'No Title')
                    status = status_info.get('overallStatus', 'N/A')
                    phase = design_info.get('phase', 'N/A')
                    
                    trials.append(f"TRIAL: {title} (ID: {nct_id})\nSTATUS: {status}\nPHASE: {phase}\n---")
        
        except Exception as e:
            trials.append(f"Error fetching clinical trials: {e}")

        return trials
    
    def use(self, query: str) -> str:
        """
        Takes a medical query and returns combined results from PubMed and ClinicalTrials.gov.
        """
        pubmed_results = self.search_pubmed(query)
        trial_results = self.search_clinical_trials(query)

        output = "--- PubMed Scientific Literature ---\n"
        output += "\n".join(pubmed_results) if pubmed_results else "No PubMed results found."
        output += "\n\n--- Clinical Trial Studies ---\n"
        output += "\n".join(trial_results) if trial_results else "No Clinical Trial results found."
        
        return output