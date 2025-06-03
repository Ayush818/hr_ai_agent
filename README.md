# 🤖 HR AI Agent

<div align="center">

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-00a393.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

**An Intelligent HR Assistant powered by AI for Policy Q&A and Leave Management**

[Features](#-features) • [Quick Start](#-quick-start) • [Installation](#-installation) • [API Documentation](#-api-documentation) • [Contributing](#-contributing)

</div>

---

## 🌟 Overview

HR AI Agent is a cutting-edge FastAPI application that revolutionizes human resources management through artificial intelligence. It combines the power of LangChain, Ollama, and ChromaDB to provide intelligent policy question-answering and streamlined leave management capabilities.

### ✨ Key Capabilities

- 🧠 **Intelligent Policy Q&A**: Get instant answers about company policies using RAG (Retrieval-Augmented Generation)
- 📝 **Smart Leave Management**: Natural language leave applications with automatic form filling
- 🔐 **Secure Authentication**: JWT-based authentication with role-based access control
- 🗄️ **Robust Database**: SQLAlchemy-powered data persistence with SQLite
- 🚀 **Modern API**: RESTful endpoints with automatic OpenAPI documentation

---

## 🛠️ Tech Stack

<div align="center">

| Category | Technology | Purpose |
|----------|------------|---------|
| **Web Framework** | ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white) | High-performance async API |
| **Server** | ![Uvicorn](https://img.shields.io/badge/Uvicorn-2E8B57?style=flat) | ASGI server for FastAPI |
| **Database** | ![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-D71F00?style=flat&logo=sqlalchemy&logoColor=white) | ORM and database toolkit |
| **Validation** | ![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=flat) | Data validation and serialization |
| **Migrations** | ![Alembic](https://img.shields.io/badge/Alembic-FF6B6B?style=flat) | Database schema migrations |
| **Authentication** | ![JWT](https://img.shields.io/badge/JWT-000000?style=flat&logo=jsonwebtokens&logoColor=white) | Secure token-based auth |
| **Password Security** | ![Passlib](https://img.shields.io/badge/Passlib-4CAF50?style=flat) | Password hashing with bcrypt |
| **AI Framework** | ![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat) | AI application development |
| **LLM** | ![Ollama](https://img.shields.io/badge/Ollama-FF6B35?style=flat) | Local language model inference |
| **Vector Store** | ![ChromaDB](https://img.shields.io/badge/ChromaDB-FF4081?style=flat) | Embeddings and similarity search |
| **Embeddings** | ![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-FFD21E?style=flat) | Sentence transformers |

</div>

---

## 🚀 Features

### 🎯 Core Functionality

- **🤔 Policy Question Answering**
  - Upload company policy documents (PDF, TXT, MD)
  - Intelligent document chunking and embedding
  - Context-aware responses with source citations

- **📋 Leave Request Management**
  - Natural language leave applications
  - Automatic form field extraction
  - Manager approval workflow
  - Status tracking and notifications

- **👥 User Management**
  - Role-based access (Employee, Manager, Admin)
  - Hierarchical organization structure
  - Secure password management

### 🔧 Technical Features

- **⚡ High Performance**: Async/await support for concurrent request handling
- **📚 Auto Documentation**: Interactive Swagger UI and ReDoc
- **🛡️ Security First**: JWT authentication, password hashing, SQL injection protection
- **🔄 Real-time Processing**: Streaming responses for AI interactions
- **📊 Comprehensive Logging**: Detailed request/response logging for debugging

---

## 🏁 Quick Start

### Prerequisites

- Python 3.9+
- Ollama installed and running
- Git

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/hr-ai-agent.git
cd hr-ai-agent
```

### 2️⃣ Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3️⃣ Environment Setup

```bash
# Create environment file
cp .env.example .env

# Edit .env with your configuration
nano .env
```

### 4️⃣ Initialize Database

```bash
# Create database and seed data
python database.py
```

### 5️⃣ Prepare Policy Documents

```bash
# Create policies directory and add your documents
mkdir policies
# Add your PDF, TXT, or MD policy files to the policies/ directory
```

### 6️⃣ Generate Embeddings

```bash
# Process policy documents and create vector store
python ingest_policies.py
```

### 7️⃣ Start the Server

```bash
# Development server with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

🎉 **Your HR AI Agent is now running at `http://localhost:8000`**

---

## 📦 Installation

### Requirements

Create a `requirements.txt` file with the following dependencies:

```txt
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
sqlalchemy>=2.0.0
pydantic[email]>=2.4.0
alembic>=1.12.0
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
langchain>=0.1.0
langchain-core>=0.1.0
langchain-ollama>=0.1.0
langchain-chroma>=0.1.0
langchain-text-splitters>=0.1.0
langchain-huggingface>=0.1.0
chromadb>=0.4.0
requests>=2.31.0
python-dotenv>=1.0.0
```

### Docker Installation (Optional)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 📋 Environment Variables

Create a `.env` file in your project root:

```env
# Security
SECRET_KEY=your_super_secret_key_here_make_it_long_and_random
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Database
DATABASE_URL=sqlite:///./hr_agent.db

# AI Models
OLLAMA_MODEL_NAME=mistral:7b-instruct-q4_K_M
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# Paths
CHROMA_DB_PATH=chroma_db_hr
POLICY_DOCS_PATH=policies
```

---

## 🔗 API Documentation

### Authentication Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/auth/token` | Obtain JWT access token |
| `POST` | `/auth/register` | Register new user |

### Chat Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Interact with AI agent |

### Leave Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/leave_requests/` | Create leave request |
| `GET` | `/leave_requests/` | List leave requests |
| `PUT` | `/leave_requests/{id}/approve` | Approve leave request |
| `PUT` | `/leave_requests/{id}/reject` | Reject leave request |

### 📖 Interactive Documentation

Once the server is running, visit:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## 🏗️ Project Structure

```
hr-ai-agent/
├── 📄 main.py              # FastAPI application and routes
├── 🗄️ database.py          # SQLAlchemy models and database setup
├── 📚 ingest_policies.py   # Document processing and embedding
├── 🔧 requirements.txt     # Python dependencies
├── ⚙️ .env                 # Environment variables
├── 📁 policies/            # Policy documents directory
├── 🗃️ chroma_db_hr/        # ChromaDB vector store
├── 💾 hr_agent.db          # SQLite database
└── 📖 README.md            # This file
```

---

## 🎯 Usage Examples

### Policy Question

```bash
curl -X POST "http://localhost:8000/chat" \
-H "Authorization: Bearer YOUR_JWT_TOKEN" \
-H "Content-Type: application/json" \
-d '{
  "query": "What is our company policy on sick leave?",
  "session_id": "user123"
}'
```

### Leave Application

```bash
curl -X POST "http://localhost:8000/chat" \
-H "Authorization: Bearer YOUR_JWT_TOKEN" \
-H "Content-Type: application/json" \
-d '{
  "query": "I want to apply for vacation leave from December 20th to 25th for a family trip",
  "session_id": "user123"
}'
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Commit your changes: `git commit -m 'Add amazing feature'`
5. Push to the branch: `git push origin feature/amazing-feature`
6. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


---

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) for the amazing web framework
- [LangChain](https://langchain.com/) for AI application development tools
- [Ollama](https://ollama.ai/) for local LLM inference
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Hugging Face](https://huggingface.co/) for pre-trained embeddings

---

<div align="center">

**Built with ❤️ for modern HR teams**

⭐ Star this repo if you find it helpful!

</div>
