import re
from typing import List, Dict, Any
from enum import Enum

class SecretSeverity(Enum):
    CRITICAL = "critical"    # API keys, passwords, private keys
    HIGH = "high"            # PII - SSN, credit cards, Aadhaar
    MEDIUM = "medium"        # PII - phone, email
    LOW = "low"              # Potential secrets, low confidence

class IngestionScanner:
    SECRET_PATTERNS = {
        # API Keys and tokens
        "openai_api_key": (
            r"sk-[a-zA-Z0-9]{20,}",
            SecretSeverity.CRITICAL
        ),
        "anthropic_api_key": (
            r"sk-ant-[a-zA-Z0-9\-_]{20,}",
            SecretSeverity.CRITICAL
        ),
        "generic_api_key": (
            r"(?:api[_\-]?key|apikey|api[_\-]?secret)\s*[:=]\s*['\"]?([a-zA-Z0-9\-_]{16,})",
            SecretSeverity.CRITICAL
        ),
        "private_key_header": (
            r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----",
            SecretSeverity.CRITICAL
        ),
        "password_field": (
            r"(?:password|passwd|pwd)\s*[:=]\s*['\"]?(\S{6,})",
            SecretSeverity.CRITICAL
        ),
        "connection_string": (
            r"(?:mongodb|postgresql|mysql|redis):\/\/[^\s'\"]+:[^\s'\"]+@",
            SecretSeverity.CRITICAL
        ),
        "aws_access_key": (
            r"AKIA[0-9A-Z]{16}",
            SecretSeverity.CRITICAL
        ),
        "aws_secret_key": (
            r"(?:aws[_\-]?secret|aws[_\-]?access)\s*[:=]\s*['\"]?([a-zA-Z0-9/+]{40})",
            SecretSeverity.CRITICAL
        ),
        # PII
        "ssn": (
            r"\b\d{3}-\d{2}-\d{4}\b",
            SecretSeverity.HIGH
        ),
        "aadhaar": (
            r"\b\d{4}\s\d{4}\s\d{4}\b",
            SecretSeverity.HIGH
        ),
        "credit_card": (
            r"\b(?:\d{4}[- ]){3}\d{4}\b",
            SecretSeverity.HIGH
        ),
        "phone_number": (
            r"\b(?:\+91|0)?[6-9]\d{9}\b",
            SecretSeverity.MEDIUM
        ),
        "email_address": (
            r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
            SecretSeverity.MEDIUM
        ),
    }
    
    def scan_document(self, text: str) -> dict:
        """
        Scan full document text before chunking.
        Returns scan results with all detected secrets/PII.
        """
        findings = []
        
        for secret_type, (pattern, severity) in self.SECRET_PATTERNS.items():
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                findings.append({
                    "type": secret_type,
                    "severity": severity.value,
                    "position_start": match.start(),
                    "position_end": match.end(),
                    "preview": self._mask_value(match.group(0)),
                    "line_number": text[:match.start()].count("\n") + 1
                })
        
        critical_findings = [
            f for f in findings 
            if f["severity"] == SecretSeverity.CRITICAL.value
        ]
        
        return {
            "has_secrets": len(findings) > 0,
            "has_critical_secrets": len(critical_findings) > 0,
            "total_findings": len(findings),
            "findings_by_severity": {
                "critical": len([f for f in findings if f["severity"] == "critical"]),
                "high": len([f for f in findings if f["severity"] == "high"]),
                "medium": len([f for f in findings if f["severity"] == "medium"]),
                "low": len([f for f in findings if f["severity"] == "low"]),
            },
            "findings": findings,
            "critical_findings": critical_findings
        }
    
    def redact_document(self, text: str) -> dict:
        """
        Redact all detected secrets/PII in place.
        Returns redacted text + redaction log.
        """
        redacted_text = text
        redaction_log = []
        
        # Process in reverse order to preserve positions
        all_matches = []
        for secret_type, (pattern, severity) in self.SECRET_PATTERNS.items():
            for match in re.finditer(pattern, redacted_text, re.IGNORECASE):
                all_matches.append((match.start(), match.end(), secret_type, severity))
        
        # Sort by position descending (process from end to preserve offsets)
        all_matches.sort(key=lambda x: x[0], reverse=True)
        
        for start, end, secret_type, severity in all_matches:
            original_preview = self._mask_value(redacted_text[start:end])
            replacement = f"[REDACTED:{secret_type.upper()}]"
            redacted_text = redacted_text[:start] + replacement + redacted_text[end:]
            redaction_log.append({
                "type": secret_type,
                "severity": severity.value,
                "original_preview": original_preview,
                "replacement": replacement,
                "line_number": text[:start].count("\n") + 1
            })
        
        return {
            "redacted_text": redacted_text,
            "redaction_count": len(redaction_log),
            "redaction_log": redaction_log
        }
    
    def _mask_value(self, value: str) -> str:
        """Show first 4 chars then mask rest."""
        if len(value) <= 4:
            return "****"
        return value[:4] + "*" * (len(value) - 4)
    
    def should_quarantine(self, scan_result: dict) -> bool:
        """
        Return True if document should be quarantined.
        """
        return scan_result["has_critical_secrets"]
