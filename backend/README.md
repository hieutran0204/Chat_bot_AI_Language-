# Language AI — English Learning AI Agent Platform (Backend)

Hệ thống Backend phục vụ ứng dụng **Học tiếng Anh thông minh (AI Agent Platform)** tích hợp kỹ thuật RAG (Retrieval-Augmented Generation), mô hình Agent tự chủ (Reasoning & Tool Execution), Ký ức dài hạn (Long-Term Memory & SRS), và kiến trúc chuẩn **Hexagonal Architecture (Ports & Adapters)**.

---

## 📌 1. Mô tả Dự án (Project Overview)

Dự án được thiết kế để giải quyết bài toán luyện tập tiếng Anh toàn diện cho người học Việt Nam với 3 trụ cột AI Agent chính:

1. **🗣️ Speaking Partner Agent (Luyện nói theo chủ đề)**:
   - Người học được chọn hoặc đề xuất bất kỳ chủ đề giao tiếp nào (*Job Interview, Travel, Daily Talk, IELTS...*).
   - Agent tự điều chỉnh vai diễn và phản hồi tự nhiên.
   - **Real-time Feedback**: Tự động nhận xét nhẹ nhàng ở cuối phản hồi nếu phát hiện người học mắc lỗi ngữ pháp/từ vựng nghiêm trọng.

2. **💾 Long-Term Memory & Student Profile (Ghi nhớ & Đánh giá người học)**:
   - Theo dõi và lưu giữ điểm yếu ngữ pháp/từ vựng thường gặp qua từng phiên học (`user_memories`).
   - Quản lý kho từ vựng theo phương pháp Lặp lại ngắt quãng Spaced Repetition (`user_vocabulary`).
   - Đánh giá và cập nhật trình độ tiếng Anh linh hoạt (A1 ➔ C2).

3. **🛠️ Agent Toolset (Bộ công cụ hỗ trợ thông minh)**:
   - **Tra từ điển (`DictionaryTool`)**: Trả về IPA, từ loại, định nghĩa, câu ví dụ, collocations.
   - **Sửa lỗi ngữ pháp chuyên sâu (`GrammarAnalyzerTool`)**: Soi lỗi sai + Gợi ý 3 cách diễn đạt chuẩn bản xứ.
   - **Tự động tạo bài tập (`QuizGeneratorAgent`)**: Tự động thiết kế bài tập trắc nghiệm/điền từ dựa trên chính điểm yếu trong quá khứ của người học.
   - **Hỏi đáp tài liệu (`RagRetrievalTool`)**: Tìm kiếm vector similarity search trên tài liệu PDF/DOCX đã upload bằng pgvector.

---

## 🛠️ 2. Tech Stack

| Layer | Công nghệ |
| :--- | :--- |
| **Backend Framework** | FastAPI + Uvicorn (Python 3.11) |
| **Architecture** | Hexagonal Architecture (Ports & Adapters) |
| **Database** | PostgreSQL 16 + `pgvector` extension (Port 5435) |
| **LLM Engine** | Ollama (`llama3.1:8b` local) / HuggingFace Inference API |
| **Embedding Model** | HuggingFace (`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`) |
| **Auth & Security** | JWT (Access & Refresh Token) + Bcrypt |
| **Migration** | Alembic |
| **Caching (Phase 3)** | Redis 7 |

---

## 🚀 3. Hướng dẫn Khởi chạy Dự án (How to Run)

### Bước 1: Chuẩn bị Môi trường
- **Python 3.11** (khuyên dùng để tương thích tốt nhất với PyMuPDF & Torch).
- **Docker Desktop** (đã bật sẵn).
- **Ollama** (đã cài đặt và tải model):
  ```powershell
  ollama pull llama3.1:8b
  ollama pull nomic-embed-text
  ```

### Bước 2: Tạo Virtual Environment & Cài đặt Library
Mở terminal tại thư mục `e:\Language_AI\backend`:

```powershell
# 1. Tạo virtual environment với Python 3.11
py -3.11 -m venv .venv

# 2. Kích hoạt môi trường ảo (Windows PowerShell)
.\.venv\Scripts\activate

# 3. Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### Bước 3: Cấu hình File Môi trường (`.env`)
```powershell
copy .env.example .env
```
Kiểm tra cấu hình cổng PostgreSQL trong `.env`:
```env
POSTGRES_PORT=5435
DATABASE_URL=postgresql+asyncpg://language_ai:language_ai_pass@localhost:5435/language_ai_db
```

### Bước 4: Khởi chạy Database (PostgreSQL + pgvector & Redis)
```powershell
docker compose up -d postgres redis
```

### Bước 5: Chạy Database Migration (Alembic)
```powershell
alembic upgrade head
```

### Bước 6: Khởi chạy Server Backend
```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
👉 Truy cập API Documentation (Swagger UI): **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## 🛑 4. Hướng dẫn Dừng & Tắt Dự án (How to Stop)

Khi muốn tạm dừng làm việc hoặc ngắt hoàn toàn hệ thống:

### Cách 1: Tắt Server Backend (Uvicorn)
Tại cửa sổ Terminal đang chạy `uvicorn`, nhấn tổ hợp phím:
```powershell
Ctrl + C
```
*(Server Uvicorn sẽ ngắt kết nối an toàn và dừng lại)*.

### Cách 2: Dừng các Docker Container (PostgreSQL & Redis)
Mở terminal tại thư mục project:

```powershell
# Dừng các container nhưng GIỮ LẠI dữ liệu Database:
docker compose stop

# Hoặc dừng và XÓA container (Dữ liệu vẫn giữ trong Docker Volume):
docker compose down

# (Tùy chọn) Xóa sạch hoàn toàn cả Dữ liệu Database cũ nếu muốn reset mới:
docker compose down -v
```

### Cách 3: Hủy kích hoạt Môi trường ảo (Deactivate VirtualEnv)
```powershell
deactivate
```

---

## 📂 5. Cấu trúc Dự án (Project Structure - Hexagonal Architecture)

```
backend/
├── app/
│   ├── api/
│   │   ├── deps.py                  # JWT Auth dependencies
│   │   └── v1/
│   │       ├── auth.py              # /auth endpoints (Register, Login, Refresh)
│   │       ├── chat.py              # /chat endpoints (SSE stream & Full message)
│   │       ├── documents.py         # /documents endpoints (Upload PDF/DOCX)
│   │       └── users.py             # /users endpoints (Profile & Progress)
│   ├── core/
│   │   ├── interfaces/              # PORTS: Abstract Interfaces
│   │   │   ├── embedder.py          # IEmbedder
│   │   │   ├── llm_provider.py      # ILLMProvider
│   │   │   ├── vector_store.py      # IVectorStore
│   │   │   ├── chunker.py           # IChunker
│   │   │   └── pipeline.py          # IRagPipeline
│   │   ├── config.py                # App settings (.env parser)
│   │   ├── database.py              # Async SQLAlchemy engine & session
│   │   └── security.py              # JWT & Bcrypt password hashing
│   ├── adapters/                    # ADAPTERS: Concrete Implementations
│   │   ├── embedders/               # HuggingFace & Ollama embedders
│   │   ├── llm/                     # Ollama & HuggingFace LLM providers
│   │   ├── vector_stores/           # PgVectorStore (pgvector implementation)
│   │   └── chunkers/                # RecursiveChunker (document splitter)
│   ├── pipelines/                   # RAG & AGENT PIPELINES
│   │   ├── base_pipeline.py         # Base pipeline & system prompts
│   │   └── naive_rag/               # Naive RAG pipeline implementation
│   ├── factory/
│   │   └── pipeline_factory.py      # Dependency Injection Factory
│   ├── models/                      # SQLAlchemy ORM Models
│   │   ├── user.py
│   │   ├── document.py              # Document & DocumentChunk models
│   │   ├── conversation.py          # Conversation & Message models
│   │   └── progress.py              # UserProgress analytics model
│   ├── schemas/                     # Pydantic Request/Response DTOs
│   ├── services/
│   │   ├── chat_service.py          # Conversation & message logic
│   │   └── ingest_service.py        # Document parsing & chunking service
│   └── main.py                      # FastAPI App entry point
├── alembic/                         # Database Migration Scripts
│   └── versions/                    # Revision script versions
├── docker-compose.yml               # Docker setup for Postgres (5435) & Redis
├── requirements.txt                 # Python dependencies
└── .env.example                     # Environment template
```

---

## 📡 6. Summary API Endpoints

### 🔐 Authentication
- `POST /api/v1/auth/register`: Đăng ký tài khoản mới.
- `POST /api/v1/auth/login`: Đăng nhập lấy cặp Access Token & Refresh Token.
- `POST /api/v1/auth/refresh`: Cấp lại Access Token mới.

### 💬 Chat & AI Agent
- `POST /api/v1/chat/conversations`: Tạo phiên trò chuyện mới (chọn mode: `conversation`, `grammar`, `vocabulary`, `document_qa`).
- `GET /api/v1/chat/conversations`: Lấy danh sách cuộc trò chuyện.
- `GET /api/v1/chat/conversations/{id}/messages`: Lấy toàn bộ lịch sử tin nhắn.
- `POST /api/v1/chat/stream`: Gửi tin nhắn và nhận phản hồi dòng chảy token realtime (SSE Stream).
- `POST /api/v1/chat/message`: Gửi tin nhắn và nhận phản hồi trọn gói.

### 📄 Documents (RAG)
- `POST /api/v1/documents/upload`: Upload tài liệu (PDF, DOCX, TXT).
- `GET /api/v1/documents/`: Lấy danh sách tài liệu đã upload.
- `DELETE /api/v1/documents/{id}`: Xóa tài liệu và các vector chunks tương ứng.

---

## 📜 7. License & Credits
Phát triển bởi đội ngũ **Language AI Team**. Sử dụng FastAPI, Ollama, pgvector và SQLAlchemy.
