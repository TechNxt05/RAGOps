from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select
from app.db import get_session
from app.models.chat import ChatSession, Message
from app.models.user import User
from app.auth.deps import get_current_user
from app.rag.engine import RAGEngine
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import StructuredTool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.mcp.server import MCPServer
from app.mcp.schemas import MCPContext
from app.mcp.registry import MCPRegistry
import os
import json
from pydantic import BaseModel

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/message")
async def send_message(
    content: str,
    project_id: int = None,
    session_id: int = None,
    temperature: float = 0.1,
    model_provider: str = "google",
    model_name: str = "gemini-1.5-flash", 
    history_limit: int = 5,
    project_context_limit: int = 2, # Number of *other* conversations to consider
    title: str = None,
    session_db: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    # 1. Get or Create Session
    if session_id:
        chat_session = session_db.get(ChatSession, session_id)
        if not chat_session or chat_session.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Session not found")
        project_id = chat_session.project_id
        is_new_session = False
    else:
        if not project_id:
            raise HTTPException(status_code=400, detail="project_id is required for new session")
        chat_session = ChatSession(user_id=current_user.id, title=title or content[:30], project_id=project_id)
        session_db.add(chat_session)
        session_db.commit()
        session_db.refresh(chat_session)
        session_id = chat_session.id
        is_new_session = True
    
    # Update Session Settings
    chat_session.settings = {
        "model_provider": model_provider,
        "model_name": model_name,
        "temperature": temperature,
        "history_limit": history_limit,
        "project_context_limit": project_context_limit
    }
    session_db.add(chat_session)
    session_db.commit()

    # 2. Store User Message
    user_msg = Message(session_id=session_id, role="user", content=content)
    session_db.add(user_msg)
    session_db.commit()

    # 3. Retrieve Current Session History (Context Window)
    past_messages = session_db.exec(
        select(Message)
        .where(Message.session_id == session_id)
        .where(Message.id != user_msg.id) # Exclude current input
        .order_by(Message.created_at.desc())
        .limit(history_limit)
    ).all()
    past_messages = sorted(past_messages, key=lambda m: m.created_at)
    
    chat_history = []
    for msg in past_messages:
        if msg.role == 'user':
            chat_history.append(HumanMessage(content=msg.content))
        elif msg.role == 'assistant':
            chat_history.append(AIMessage(content=msg.content))


    # 4. Retrieve Cross-Session Context (Other Conversations)
    other_context_str = ""
    if project_context_limit > 0:
        other_sessions = session_db.exec(
            select(ChatSession)
            .where(ChatSession.project_id == project_id)
            .where(ChatSession.id != session_id) # Exclude current session
            .order_by(ChatSession.created_at.desc())
            .limit(project_context_limit)
        ).all()
        
        if other_sessions:
            other_context_str = "\n\n### RELATED PROJECT CHATS (CONTEXT):\n"
            for osess in other_sessions:
                # Fetch last 3 messages from this other session to give context
                osess_msgs = session_db.exec(
                    select(Message)
                    .where(Message.session_id == osess.id)
                    .order_by(Message.created_at.desc())
                    .limit(3)
                ).all()
                osess_msgs = sorted(osess_msgs, key=lambda m: m.created_at)
                
                if osess_msgs:
                    other_context_str += f"- Chat '{osess.title}':\n"
                    for m in osess_msgs:
                        other_context_str += f"  {m.role.upper()}: {m.content[:200]}...\n" # Truncate for token efficiency

    # 5. Initialize MCP Server & Context
    mcp_context = MCPContext(
        user_id=current_user.id,
        user_role=current_user.role,
        project_id=project_id,
        session_id=session_id
    )
    mcp_server = MCPServer(mcp_context)
    
    # 6. Prepare Tools for LangChain
    available_tools_meta = mcp_server.get_available_tools()
    langchain_tools = []
    
    
    for meta in available_tools_meta:
        # Create a closure that calls MCPServer.call_tool
        tool_name = meta.name
        
        async def _tool_wrapper(tool_input: BaseModel = None, tool_name=tool_name, **kwargs):
            if tool_input:
                kwargs = tool_input.dict()
            output = await mcp_server.call_tool(tool_name, kwargs)
            return output.content
            
        tool_entry = MCPRegistry.get_tool(tool_name)
        input_model = tool_entry["model"]
        
        lc_tool = StructuredTool.from_function(
            func=None,
            coroutine=_tool_wrapper,
            name=tool_name,
            description=meta.description,
            args_schema=input_model
        )
        langchain_tools.append(lc_tool)

    # 7. Initialize LLM
    if model_provider == "groq":
        llm = ChatGroq(model_name=model_name, api_key=os.getenv("GROQ_API_KEY"), temperature=temperature)
    else:
        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=os.getenv("GEMINI_API_KEY"), temperature=temperature)

    print(f"DEBUG: Chat Temperature: {temperature} | Model: {model_name}")

    # 8. Create Agent
    rag_engine = RAGEngine(session_db)
    try:
        rag_config = rag_engine.get_active_config(project_id)
        style = rag_config.response_style
    except:
        style = "Concise"

    system_prompt = f"""You are a helpful AI assistant with access to tools.
    Current Project ID: {project_id}.
    Response Style: {style}.
    
    {other_context_str}
    
    IMPORTANT: You are an augmented AI. 
    1. If the user asks a question, use 'search_documents' to find the answer.
    2. Answer the question DIRECTLY based on the retrieved content. 
    3. Do NOT start your answer with "The project contains..." or "The document mentions...". Just state the fact.
    4. Do NOT mention "The project ID is..." unless explicitly asked.
    5. If asked for a summary, synthesize information from ALL retrieved chunks to provide a comprehensive overview.
    6. If the retrieved chunks contain a section explicitly labeled "SUMMARY" or "ABSTRACT", prioritize that information.
    """

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="chat_history"), # Inject History
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Fallback if agent creation fails
    try:
        agent = create_tool_calling_agent(llm, langchain_tools, prompt)
    except Exception as e:
         return {
            "session_id": session_id,
            "role": "assistant", 
            "content": f"System Error: Failed to initialize AI Agent. Details: {str(e)}",
            "sources": []
         }
    agent_executor = AgentExecutor(agent=agent, tools=langchain_tools, verbose=True)

    # 9. Execute Agent
    try:
        # Pass chat_history to the agent
        result = await agent_executor.ainvoke({
            "input": content,
            "chat_history": chat_history
        })
        answer = result["output"]
    except Exception as e:
        answer = f"Error during processing: {str(e)}"
        
    # 10. Store Assistant Message
    assistant_msg = Message(
        session_id=session_id, 
        role="assistant", 
        content=answer, 
        sources="[]" 
    )
    session_db.add(assistant_msg)
    session_db.commit()

    # 11. Auto-Title for New Sessions
    if is_new_session and not title:
        try:
            title_prompt = f"Summarize this conversation into a very short 3-5 word title. Do not use quotes. User: {content}\nAI: {answer}"
            title_response = llm.invoke(title_prompt)
            new_title = title_response.content.strip()
            
            # Clean up title
            new_title = new_title.replace('"', '').replace("'", "").replace("**", "")[:50]
            
            chat_session.title = new_title
            session_db.add(chat_session)
            session_db.commit()
            print(f"DEBUG: Auto-titled session {session_id} to '{new_title}'")
        except Exception as e:
            print(f"DEBUG: Failed to auto-title session: {e}")

    return {
        "session_id": session_id,
        "role": "assistant", 
        "content": answer,
        "sources": []
    }

@router.get("/sessions")
def get_sessions(project_id: int = None, session_db: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    query = select(ChatSession).where(ChatSession.user_id == current_user.id)
    if project_id:
        query = query.where(ChatSession.project_id == project_id)
    
    print(f"DEBUG: get_sessions project_id={project_id} user_id={current_user.id}")
    sessions = session_db.exec(query.order_by(ChatSession.created_at.desc())).all()
    print(f"DEBUG: found {len(sessions)} sessions")
    return sessions

@router.delete("/sessions/{session_id}")
def delete_session(session_id: int, session_db: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    chat_session = session_db.get(ChatSession, session_id)
    if not chat_session or chat_session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_db.delete(chat_session)
    session_db.commit()
    return {"ok": True}

@router.get("/history/{session_id}")
def get_history(session_id: int, session_db: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    chat_session = session_db.get(ChatSession, session_id)
    if not chat_session or chat_session.user_id != current_user.id:
         raise HTTPException(status_code=404, detail="Session not found")
    messages = session_db.exec(select(Message).where(Message.session_id == session_id).order_by(Message.created_at)).all()
    
    # Also return session settings if needed, but this endpoint returns List[Message]
    # Maybe we should return {messages: [], settings: {}}?
    # For now, let's keep it simplest: Frontend can get settings from `get_sessions` list or we add a new endpoint `get_session_details`.
    # Actually, let's update this to return a wrapped object or just stick to messages and let frontend find settings in the session list.
    return messages
