@echo off
REM RAGOps MCP Server Batch Launcher
set SCRIPT_DIR=%~dp0
set PYTHONPATH=%SCRIPT_DIR%backend;%PYTHONPATH%
python "%SCRIPT_DIR%backend\mcp_server.py"
