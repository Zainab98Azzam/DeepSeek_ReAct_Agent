# In AgentTools/writing_agent.py

import os
from openai import OpenAI
from .base_tool import BaseTool
from dotenv import load_dotenv

load_dotenv()

class WritingAgent(BaseTool):
    """
    A tool that writes a final, structured report based on research notes.
    """
    def __init__(self):
        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        self.api_model_name = "deepseek-chat"
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com/v1")

        # Correctly load the prompt using a robust method
        self.writing_prompt = self.load_prompt("writing_prompt.txt")

    def load_prompt(self, file_name: str) -> str:
        # Use os.path.join and os.path.dirname for cross-platform compatibility
        # This gets the directory of the current file and then navigates to the Prompts folder
        prompt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "Prompts",
            file_name
        )
        
        try:
            with open(prompt_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: The file {prompt_path} was not found.")
            return ""

    def use(self, notes: str) -> str:
        # The rest of this method remains unchanged
        messages = [
            {"role": "system", "content": self.writing_prompt},
            {"role": "user", "content": f"Based on these notes, write a professional report:\n{notes}"}
        ]
        
        completion = self.client.chat.completions.create(
            model=self.api_model_name,
            messages=messages,
            temperature=0.7,
        )
        return completion.choices[0].message.content