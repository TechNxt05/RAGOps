from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

router = APIRouter(prefix="/playground", tags=["playground"])

class PlaygroundMessage(BaseModel):
    role: str
    content: str

class PlaygroundRequest(BaseModel):
    messages: List[PlaygroundMessage]
    model_provider: str = "groq"
    model_name: str = "llama-3.3-70b-versatile"
    temperature: float = 0.7
    max_tokens: Optional[int] = 1000
    system_prompt: Optional[str] = None

@router.post("/generate")
async def playground_generate(request: PlaygroundRequest):
    try:
        # 1. Initialize LLM
        if request.model_provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")
            llm = ChatGroq(
                model_name=request.model_name, 
                api_key=api_key, 
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
        elif request.model_provider == "google":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")
            llm = ChatGoogleGenerativeAI(
                model=request.model_name, 
                google_api_key=api_key, 
                temperature=request.temperature,
                max_output_tokens=request.max_tokens
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported provider: {request.model_provider}")

        # 2. Build Messages
        langchain_msgs = []
        
        # System Message
        if request.system_prompt and request.system_prompt.strip():
            langchain_msgs.append(SystemMessage(content=request.system_prompt))
            
        # Chat History
        for msg in request.messages:
            if msg.role == "user":
                langchain_msgs.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                langchain_msgs.append(AIMessage(content=msg.content))
            elif msg.role == "system":
                langchain_msgs.append(SystemMessage(content=msg.content))
                
        # 3. Generate
        response = await llm.ainvoke(langchain_msgs)
        
        return {
            "content": response.content,
            "usage": response.response_metadata.get("token_usage", {})
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
