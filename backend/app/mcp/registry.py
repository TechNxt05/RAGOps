from typing import Dict, Type, Callable, Any
from app.mcp.schemas import ToolMetadata
from pydantic import BaseModel

class MCPRegistry:
    _tools: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(cls, name: str, description: str, input_model: Type[BaseModel], fn: Callable):
        """
        Register a new tool.
        input_model: Pydantic model defining the arguments.
        fn: The async function to execute.
        """
        cls._tools[name] = {
            "metadata": ToolMetadata(
                name=name,
                description=description,
                input_schema=input_model.schema()
            ),
            "fn": fn,
            "model": input_model
        }
        
    @classmethod
    def get_tool(cls, name: str):
        return cls._tools.get(name)
        
    @classmethod
    def get_all_tools(cls):
        return [t["metadata"] for t in cls._tools.values()]
