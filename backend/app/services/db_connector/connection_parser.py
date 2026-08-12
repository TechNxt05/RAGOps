"""
Connection Parser — LLM & Regex-assisted extraction of database parameters from connection URIs.
Parses pasted connection strings into a structured ParsedConnection object and human confirmation summary.
Does NOT execute connection attempts using unconfirmed parsed values.
"""

import re
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class ParsedConnection(BaseModel):
    db_type: str  # postgresql, mysql, mongodb
    host: str
    port: int
    db_name: str
    username: str
    password: str
    ssl_mode: Optional[str] = "prefer"
    raw_uri: str


class ConnectionParser:
    DEFAULT_PORTS = {
        "postgresql": 5432,
        "postgres": 5432,
        "mysql": 3306,
        "mongodb": 27017,
        "mongodb+srv": 27017,
    }

    def parse_connection_string(self, connection_string: str) -> Dict[str, Any]:
        """
        Parse connection URI or freeform input into ParsedConnection + confirmation summary.
        """
        raw_str = connection_string.strip()
        
        # Match standard URI formats (postgresql://user:pass@host:port/dbname)
        try:
            if "://" in raw_str:
                parsed = urlparse(raw_str)
                scheme = parsed.scheme.lower()
                db_type = "postgresql" if scheme in ("postgresql", "postgres") else "mysql" if scheme == "mysql" else "mongodb" if "mongodb" in scheme else "postgresql"
                
                host = parsed.hostname or "localhost"
                default_port = self.DEFAULT_PORTS.get(scheme, self.DEFAULT_PORTS.get(db_type, 5432))
                port = parsed.port or default_port
                db_name = (parsed.path or "").strip("/") or "postgres"
                username = parsed.username or "postgres"
                password = parsed.password or ""
                
                # Check for sslmode query parameter
                query_params = parse_qs(parsed.query)
                ssl_mode = query_params.get("sslmode", ["prefer"])[0]

                parsed_obj = ParsedConnection(
                    db_type=db_type,
                    host=host,
                    port=port,
                    db_name=db_name,
                    username=username,
                    password=password,
                    ssl_mode=ssl_mode,
                    raw_uri=raw_str
                )

                summary = (
                    f"I found a {db_type.upper()} connection to database '{db_name}' "
                    f"on host '{host}:{port}' with user '{username}'. Confirm to save?"
                )

                return {
                    "success": True,
                    "parsed": parsed_obj.dict(),
                    "confirmation_summary": summary
                }

        except Exception as exc:
            pass

        # Fallback regex extraction for non-URI strings
        db_type = "postgresql"
        if "mysql" in raw_str.lower():
            db_type = "mysql"
        elif "mongo" in raw_str.lower():
            db_type = "mongodb"

        host_match = re.search(r'host\s*[:=]\s*([^\s,;]+)', raw_str, re.IGNORECASE)
        port_match = re.search(r'port\s*[:=]\s*(\d+)', raw_str, re.IGNORECASE)
        db_match = re.search(r'(db|database|name)\s*[:=]\s*([^\s,;]+)', raw_str, re.IGNORECASE)
        user_match = re.search(r'(user|username)\s*[:=]\s*([^\s,;]+)', raw_str, re.IGNORECASE)
        pass_match = re.search(r'(pass|password|pwd)\s*[:=]\s*([^\s,;]+)', raw_str, re.IGNORECASE)

        host = host_match.group(1) if host_match else "localhost"
        port = int(port_match.group(1)) if port_match else self.DEFAULT_PORTS.get(db_type, 5432)
        db_name = db_match.group(2) if db_match else "default_db"
        username = user_match.group(2) if user_match else "admin"
        password = pass_match.group(2) if pass_match else ""

        parsed_obj = ParsedConnection(
            db_type=db_type,
            host=host,
            port=port,
            db_name=db_name,
            username=username,
            password=password,
            ssl_mode="prefer",
            raw_uri=raw_str
        )

        summary = (
            f"I extracted a {db_type.upper()} connection to database '{db_name}' "
            f"on host '{host}:{port}' with user '{username}'. Confirm to save?"
        )

        return {
            "success": True,
            "parsed": parsed_obj.dict(),
            "confirmation_summary": summary
        }


connection_parser = ConnectionParser()
