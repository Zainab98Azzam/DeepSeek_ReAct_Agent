
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from .agent import Agent
from .AgentTools.search_agent import SearchAgent
from .AgentTools.writing_agent import WritingAgent

# Add the WritingAgent back to the tools list
tools_list = [
    SearchAgent(),
    WritingAgent()
]
agent = Agent(tools=tools_list)
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat(request: QueryRequest) -> Dict:
    try:
        result = agent.run(request.query)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear_history")
async def clear_history():
    agent.clear_history()
    return {"message": "Chat history cleared successfully."}