import os
import asyncio
from langchain_groq import ChatGroq
from langchain_core.tools import StructuredTool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel

class MockInput(BaseModel):
    query: str

async def mock_coroutine(query: str):
    return "Mock Result"

def run_test():
    try:
        print("Initializing LLM...")
        llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key="gsk_mock_key")
        
        print("Creating Tools...")
        # Replicating the exact pattern from chat_routes.py
        tools = []
        lc_tool = StructuredTool.from_function(
            func=None,
            coroutine=mock_coroutine,
            name="mock_tool",
            description="A mock tool",
            args_schema=MockInput
        )
        # tools.append(lc_tool)
        tools = [] # Force empty tools
        
        print("Creating Prompt...")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helper."),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        print("Creating Agent...")
        agent = create_tool_calling_agent(llm, tools, prompt)
        print("✅ Agent created successfully.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
