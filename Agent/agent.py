import os
import json
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv

from .AgentTools.base_tool import BaseTool

load_dotenv()

class Agent:
    """
    A ReAct (Reasoning and Acting) Agent that uses DeepSeek API to reason about problems
    and act by using tools to solve them.
    """
    
    MAX_ITERATIONS = 5
    MAX_SEARCHES = 2
    
    def __init__(self, tools: List[BaseTool], api_model_name: str = "deepseek-chat"):
        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        self.api_model_name = api_model_name
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com/v1")
        
        self.system_prompt = self.load_prompt("Agent/Prompts/system_prompt.txt")
        self.summary_prompt = self.load_prompt("Agent/Prompts/summary_prompt.txt")
        
        self.messages = []
        self.search_count = 0
        
        self.tools = {tool.__class__.__name__: tool for tool in tools}
        self.deepseek_tools = self.format_tools_for_deepseek()
        self.tools_string = self.build_tools_string()
        
    def load_prompt(self, file_path: str) -> str:
        """Load prompt content from a file."""
        try:
            with open(file_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
            return ""

    def register_tool(self, tool_instance: BaseTool):
        """Register a new tool with the agent."""
        self.tools[tool_instance.__class__.__name__] = tool_instance
        self.deepseek_tools = self.format_tools_for_deepseek()
        self.tools_string = self.build_tools_string()
    
    def build_tools_string(self) -> str:
        """Create a string representation of available tools for the LLM prompt."""
        tools_list = []
        for name, tool in self.tools.items():
            docstring = tool.use.__doc__.strip() if tool.use.__doc__ else "No documentation provided."
            tools_list.append(f'tool_name("{name}") -> {docstring}')
        return "\n".join(tools_list)


    def format_tools_for_deepseek(self):
            """Format tools for DeepSeek API function calling."""
            tools_list = []
            for name, tool in self.tools.items():
                docstring = tool.use.__doc__.strip() if tool.use.__doc__ else "No documentation provided."
                
                # Define parameters based on the tool's purpose
                # ADDED: MedicalAgent now shares the 'query' parameter structure
                if name in ["SearchAgent", "MedicalAgent"]:
                    parameters = {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query."}
                        },
                        "required": ["query"] 
                    }
                elif name == "WritingAgent":
                    parameters = {
                        "type": "object",
                        "properties": {
                            "notes": {"type": "string", "description": "The research notes to use for writing."}
                        },
                        "required": ["notes"]
                    }
                # Note: If PDFGeneratorAgent is registered in app.py, you would need to add its parameter logic here as well.
                else:
                    parameters = {"type": "object", "properties": {}, "required": []}
                
                tools_list.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": docstring,
                        "parameters": parameters
                    }
                })
            return tools_list

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.messages.append({"role": role, "content": content})
        
    def think(self) -> str:
        """Generate a response from the DeepSeek API."""
        completion = self.client.chat.completions.create(
            model=self.api_model_name,
            messages=self.messages,
            tools=self.deepseek_tools,
            tool_choice="auto",
            max_tokens=500,
            temperature=0.1,
        )
        return completion.choices[0].message

    def decide(self, response) -> Optional[str]:
        """Decide what to do based on the API response."""
        tool_calls = response.tool_calls
        if tool_calls:
            return self._handle_tool_calls(response, tool_calls)
        else:
            return self._handle_final_answer(response)
    
    # In Agent/agent.py (around line 140)

    def _handle_tool_calls(self, response, tool_calls) -> str:
        """Handle tool calls from the API response."""
        # Add the assistant's response (with tool calls) to the chat history
        self.messages.append({"role": response.role, "tool_calls": [
            {
                "id": call.id,
                "type": call.type,
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments
                }
            } for call in tool_calls
        ]})

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            tool_id = tool_call.id

            if tool_name in self.tools:
                tool_instance = self.tools[tool_name]
                observation = tool_instance.use(**tool_args)

                # FIX: Route both research agents to the same handler
                if tool_name in ["SearchAgent", "MedicalAgent"]:
                    return self._handle_search_agent(observation, tool_id)
                
                elif tool_name == "WritingAgent":
                    return self._handle_writing_agent(observation, tool_id)
                
                # Fallback for any other tool: Log the observation
                self.messages.append({
                    "role": "tool",
                    "content": observation,
                    "tool_call_id": tool_id
                })
        
        return "loop"
    
    def _handle_search_agent(self, observation: str, tool_id: str) -> str:
        """Handle SearchAgent tool call."""
        self.messages.append({
            "role": "tool",
            "content": observation,
            "tool_call_id": tool_id
        })
        self.search_count += 1

        # Force transition to WritingAgent after max searches
        if self.search_count >= self.MAX_SEARCHES:
            self.messages.append({
                "role": "user",
                "content": "The search is complete. Please use the WritingAgent to cre m   hhhhate a professional report based on the provided results. Do not perform any further searches."
            })
        
        return "loop"
    
    def _handle_writing_agent(self, observation: str, tool_id: str) -> str:
        """Handle WritingAgent tool call."""
        self.messages.append({
            "role": "tool",
            "content": observation,
            "tool_call_id": tool_id
        })
        return f"Final Answer: {observation}"
    
    def _handle_final_answer(self, response) -> Optional[str]:
        """Handle final answer from the LLM."""
        final_answer = response.content
        if final_answer:
            self.add_message("assistant", final_answer)
            return f"Final Answer: {final_answer}"
        return None




   # In Agent/agent.py

    def run(self, query: str) -> str:
        """Run the agent with a given query."""
        self.add_message("user", query)
        # ADDED: Print a header to organize the output
        print("\n--- AGENT EXECUTION TRACE ---") 
        
        for i in range(self.MAX_ITERATIONS): # Use 'i' for iteration count
            
            # Print the current step number
            print(f"\n--- STEP {i+1} / {self.MAX_ITERATIONS} ---") 
            
            response = self.think()
            
            # ADDED: Print the LLM's raw thought/response content
            print(f"Thought/Content: {response.content}") 
            
            # ADDED: Check if the LLM chose a tool
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    print(f"Action/Tool Call: {tool_call.function.name}")
                    print(f"Arguments: {tool_call.function.arguments}")
            
            result = self.decide(response)
            
            if result == "loop":
                # ADDED: Print the result of the action (the observation)
                # Note: We won't print the full observation here as it's too long, 
                # but we confirm the tool ran successfully.
                print("Observation: Tool executed successfully. Looping to next step.") 
                continue
            
            if result and result.startswith("Final Answer:"):
                # ADDED: Print a clean footer before returning
                print("--- TRACE COMPLETE ---\n") 
                return result.replace("Final Answer:", "").strip()

        print("--- TRACE FAILED ---\n")
        return "The agent failed to find a final answer after multiple attempts."
    
    def clear_history(self):
        """Clear the agent's chat history."""
        self.messages = []
        self.search_count = 0
        print("Agent's chat history has been cleared.")