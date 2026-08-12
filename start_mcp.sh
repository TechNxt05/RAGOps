#!/bin/bash
# RAGOps MCP Server Launcher
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="$SCRIPT_DIR/backend:$PYTHONPATH"
python "$SCRIPT_DIR/backend/mcp_server.py"
