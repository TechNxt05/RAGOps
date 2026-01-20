from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="RAGOps API", version="1.0.0")

# CORS Setup
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:3007",
    "https://ragops-production.up.railway.app", # Placeholder for deployment
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_origin_regex="https://.*\.vercel\.app",
)

from app.db import init_db
from app.auth import routes as auth_routes
from app.rag import config_routes, ingest_routes, chat_routes, project_routes, inspector_routes, play_routes, analytics_routes

@app.on_event("startup")
def on_startup():
    init_db()
    from app.mcp.registry_init import register_tools
    register_tools()

app.include_router(auth_routes.router)
app.include_router(project_routes.router)
app.include_router(config_routes.router)
app.include_router(ingest_routes.router)
app.include_router(chat_routes.router)
app.include_router(inspector_routes.router, prefix="/rag")
app.include_router(play_routes.router)
app.include_router(analytics_routes.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to RAGOps API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
