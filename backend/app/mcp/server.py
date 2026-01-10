from app.mcp.registry import MCPRegistry
from app.mcp.permissions import MCPPermissionEngine
from app.mcp.schemas import MCPContext, MCPToolOutput
import logging

logger = logging.getLogger(__name__)

class MCPServer:
    def __init__(self, context: MCPContext):
        self.context = context
        
    def get_available_tools(self):
        """Get tools allowed for this user"""
        all_tools = MCPRegistry.get_all_tools()
        return MCPPermissionEngine.filter_tools(all_tools, self.context)
        
    async def call_tool(self, tool_name: str, arguments: dict) -> MCPToolOutput:
        """
        Execute a tool safely.
        """
        # 1. Check Permission
        if not MCPPermissionEngine.check_permission(tool_name, self.context):
            raise PermissionError(f"User {self.context.user_id} ({self.context.user_role}) denied access to {tool_name}")
            
        # 2. Get Tool
        tool_entry = MCPRegistry.get_tool(tool_name)
        if not tool_entry:
            raise ValueError(f"Tool {tool_name} not found")
            
        # 3. Validate Input
        try:
            input_model = tool_entry["model"](**arguments)
        except Exception as e:
            return MCPToolOutput(content=f"Error validating arguments: {str(e)}")
            
        # 4. Execute
        fn = tool_entry["fn"]
        try:
            logger.info(f"MCP: Executing {tool_name} for Project {self.context.project_id}")
            result = await fn(input_model, self.context)
            return MCPToolOutput(content=result)
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"MCP Tool Error: {error_trace}")
            return MCPToolOutput(content=f"Tool execution failed: {str(e)}")
