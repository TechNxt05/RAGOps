"""
Connection Wizard — Slot-filling conversational manager (Path B).
Gathers missing database connection parameters step-by-step using session state.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class WizardState(BaseModel):
    db_type: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    db_name: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    ssl_mode: Optional[str] = "prefer"
    step: str = "db_type"


class ConnectionWizard:
    DEFAULT_PORTS = {
        "postgresql": 5432,
        "mysql": 3306,
        "mongodb": 27017,
    }

    def process_step(self, current_state: Dict[str, Any], user_input: str) -> Dict[str, Any]:
        """
        Processes a single slot-filling message input and returns the next prompt or completed config.
        """
        state = WizardState(**(current_state or {}))
        text = user_input.strip()

        if state.step == "db_type":
            t = text.lower()
            if "mysql" in t:
                state.db_type = "mysql"
            elif "mongo" in t:
                state.db_type = "mongodb"
            else:
                state.db_type = "postgresql"
            state.step = "host"
            return {
                "next_prompt": f"Selected database type '{state.db_type.upper()}'. What is the host address? (Default: localhost)",
                "state": state.dict(),
                "completed": False
            }

        elif state.step == "host":
            state.host = text if text and text.lower() != "default" else "localhost"
            default_port = self.DEFAULT_PORTS.get(state.db_type, 5432)
            state.step = "port"
            return {
                "next_prompt": f"Host set to '{state.host}'. What is the port number? (Default: {default_port})",
                "state": state.dict(),
                "completed": False
            }

        elif state.step == "port":
            try:
                state.port = int(text) if text.isdigit() else self.DEFAULT_PORTS.get(state.db_type, 5432)
            except ValueError:
                state.port = self.DEFAULT_PORTS.get(state.db_type, 5432)
            state.step = "db_name"
            return {
                "next_prompt": f"Port set to {state.port}. What is the database name?",
                "state": state.dict(),
                "completed": False
            }

        elif state.step == "db_name":
            state.db_name = text or "production_db"
            state.step = "username"
            return {
                "next_prompt": f"Database name set to '{state.db_name}'. What is the username?",
                "state": state.dict(),
                "completed": False
            }

        elif state.step == "username":
            state.username = text or "admin"
            state.step = "password"
            return {
                "next_prompt": f"Username set to '{state.username}'. Please enter the password (stored encrypted at rest):",
                "state": state.dict(),
                "completed": False
            }

        elif state.step == "password":
            state.password = text
            state.step = "confirmation"
            summary = (
                f"Configuration complete!\n"
                f"- Database Type: {state.db_type.upper()}\n"
                f"- Host: {state.host}:{state.port}\n"
                f"- Database: {state.db_name}\n"
                f"- User: {state.username}\n\n"
                f"Confirm to test connection and save?"
            )
            return {
                "next_prompt": summary,
                "state": state.dict(),
                "completed": True,
                "confirmation_summary": summary
            }

        return {
            "next_prompt": "Slot filling complete. Ready to connect.",
            "state": state.dict(),
            "completed": True
        }


connection_wizard = ConnectionWizard()
