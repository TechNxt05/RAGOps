# RAGOps: Enterprise-Grade RAG Platform

> **Admin-Controlled, Project-Based Retrieval Augmented Generation System**
>
> *Built with Next.js, FastAPI, LangChain, and PostgreSQL.*



## 🚀 Problem Statement

In the rapidly evolving landscape of Generative AI, organizations face three critical challenges when deploying RAG (Retrieval Augmented Generation) solutions:

1.  **Data Isolation & Security**: Generic chatbots lack boundaries. Sensitive HR documents shouldn't mix with casual internal wiki searches.
2.  **Hallucination Control**: "Black box" AI systems often answer questions without grounding, leading to misinformation. Admins need control over *what* the AI knows.
3.  **Lack of Transparency**: Users rarely know *why* an AI gave a specific answer. Was it from the Policy Document v1 or v2?

**RAGOps** solves this by introducing a strict **Project-Based Architecture** with **Admin-Controlled Context**.

---

## ✨ Key Features

### 1. Project-Based Knowledge Scoping
*   **Isolation**: Create distinct "Projects" (e.g., *Legal*, *Engineering*, *Marketing*).
*   **Context Fencing**: Documents uploaded to *Legal* are **never** accessible to the *Engineering* chat bot.
*   **Granular Config**: Set different RAG parameters (Chunk Size, Overlap, Temperature) per project.

### 2. Role-Based Access Control (RBAC)
*   **Admin Role**:
    *   Full control over Knowledge Base ingestion (PDF/TXT uploads).
    *   Manage Projects and RAG configurations.
    *   Inspect specific document chunks and vector scores.
*   **Client Role**:
    *   Consumer-facing Chat Interface.
    *   Can only query active projects.
    *   Cannot tamper with the underlying knowledge base.

### 3. Advanced Chat Experience
*   **Multi-Model Support**: Switch seamlessly between **Google Gemini 1.5** (Cost-effective) and **Groq Llama 3** (High speed).
*   **Auto-Organized History**: Chat sessions are automatically named and categorized under their respective projects.
*   **Smart Context**: The AI remembers previous turns and can even pull context from related sessions in the same project.
*   **Citations**: Every answer includes clickable source badges, showing exactly which document (and file) the information came from.

### 4. Admin Inspector & Debugging
*   **Retrieval Playground**: Test how your documents are being chunked and retrieved before users see them.
*   **Visual Scoring**: See "Similarity Scores" to tune the vector search precision.

---

## 🛠️ Tech Stack

### Frontend
*   **Framework**: Next.js 14 (App Router)
*   **Styling**: Tailwind CSS + Shadcn UI
*   **State**: React Hooks + Context API
*   **Animations**: Framer Motion

### Backend
*   **API**: FastAPI (Python)
*   **Database**: PostgreSQL (via SQLModel/Pydantic)
*   **AI Orchestration**: LangChain
*   **Embeddings**: Local HuggingFace Models (No API cost for embeddings)
*   **LLM Providers**: Google Gemini API, Groq API

---

## ⚡ Deployment

### Prerequisites
*   Node.js 18+
*   Python 3.10+
*   PostgreSQL Database (Remote or Local)

### 1. Backend Setup
```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate | Mac/Linux: source venv/bin/activate
pip install -r requirements.txt

# Create .env file with your keys (DATABASE_URL, GEMINI_API_KEY, etc.)
python -m app.main
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### 3. Production Build
See `deployment_guide.md` for detailed instructions on deploying to **Render** or **Railway**.

---

## 🔮 Future Roadmap
- [ ] Multi-Modal Retrieval (Images/Tables)
- [ ] Slack/Discord Integration
- [ ] Team Collaboration (Shared Chat Sessions)

---

**Author**: Amritanshu Yadav
**License**: MIT
