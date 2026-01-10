from typing import List, Dict
from app.mcp.schemas import MCPContext

# Hardcoded permissions for now, can be moved to DB later
TOOL_PERMISSIONS = {
    "search_documents": ["client", "admin"],
    "get_active_rag_config": ["admin"], # Only admin can see detailed config
    "get_document_sources": ["client", "admin"]
}

class MCPPermissionEngine:
    @staticmethod
    def check_permission(tool_name: str, context: MCPContext) -> bool:
        allowed_roles = TOOL_PERMISSIONS.get(tool_name, [])
        if context.user_role in allowed_roles:
            return True
        return False
        
    @staticmethod
    def filter_tools(tools_metadata: List, context: MCPContext) -> List:
        """Return only tools the user is allowed to see/use"""
        allowed = []
        for tool in tools_metadata:
            if MCPPermissionEngine.check_permission(tool.name, context):
                allowed.append(tool)
        return allowed
