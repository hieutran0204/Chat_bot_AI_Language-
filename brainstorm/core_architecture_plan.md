# 🏛️ Core Architecture Plan: LangChain-Standard & Self-Hosted Redis Core

> **Mode**: BRAINSTORM (Thinking & System Analysis)  
> **Ngày lập**: 2026-09-12  
> **Mục tiêu**: Thiết kế kiến trúc tầng Core vững chắc theo chuẩn triết lý LangChain, tự code toàn bộ lõi tích hợp Redis & pgvector & Ollama nội bộ (Self-hosted), tuyệt đối không phụ thuộc vào dịch vụ Cloud/SaaS bên thứ 3.

---

## 1. Phân tích Hiện trạng & Vấn đề (Problem Analysis)

### 1.1. Hiện trạng Codebase
1. **Sự phân mảnh kiến trúc**:
   - Tồn tại song song 2 luồng: `app/rag/` (code monolithic cũ) và `app/adapters/` + `app/core/interfaces/` (Hexagonal mới).
   - `DocumentService` đang ôm đồm từ File I/O, Database CRUD đến Ingestion (chunking, embedding).
2. **Thiếu vắng chuẩn LangChain Core**:
   - Format trao đổi giữa các tầng hiện tại là `dict[str, str]` thô sơ (`{"role": "user", "content": "..."}`).
   - Chưa có định nghĩa chuẩn về **Messages** (`SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage`).
   - Chưa có **Prompt Template Engine** chuẩn hóa để format động role, context, và user profile.
   - Chưa có chuẩn **Tool Abstraction** (`BaseTool` với Pydantic schema validation).
3. **Redis đã có hạ tầng nhưng chưa có Core Code**:
   - `docker-compose.yml` đã chạy `redis:7-alpine` (Port 6379).
   - `requirements.txt` đã cài `redis==5.1.0`.
   - `config.py` đã có `redis_url = "redis://localhost:6379/0"`.
   - **Tuy nhiên, trong code chưa có bất kỳ dòng nào sử dụng Redis!** Chưa có Redis connection pool, Cache manager, Chat sliding-window memory, hay Response caching.

### 1.2. Mục tiêu Thiết kế Core mới
* **Chuẩn LangChain Design Patterns**: Áp dụng các nguyên lý thiết kế kinh điển của LangChain (Message Polymorphism, Prompt Templates, Runnable Pipeline, BaseTool, Memory Interface) nhưng **tự code trực tiếp trong Core**, không biến dự án thành "black-box" phụ thuộc vào thư viện bên ngoài.
* **100% Self-Hosted / Local-First**:
  * **Database**: PostgreSQL 16 + `pgvector` (Vector store & Persistent relational data).
  * **Fast In-Memory / Caching**: Redis 7 (Session, Chat Memory Buffer, Query Cache).
  * **LLM & Embeddings**: Ollama (`llama3.1:8b`) + HuggingFace `SentenceTransformers` local.
  * Tuyệt đối không dùng dịch vụ Cloud SaaS trả phí (OpenAI API, LangSmith, Pinecone, Redis Cloud).

---

## 2. Thiết kế Kiến trúc Tầng Core (Core Architecture Blueprint)

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   APPLICATION CORE LAYER                                    │
│                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                                 LANGCHAIN-STANDARD CORE                               │  │
│  │                                                                                       │  │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐  │  │
│  │  │     MESSAGES     │  │ PROMPT TEMPLATES │  │    BASE TOOLS    │  │   RUNNABLE    │  │  │
│  │  │ - SystemMessage  │  │ - ChatPrompt     │  │ - BaseTool       │  │ - IRunnable   │  │  │
│  │  │ - HumanMessage   │  │   Template       │  │ - Pydantic Schema│  │ - Sequential  │  │  │
│  │  │ - AIMessage      │  │ - FewShotPrompt  │  │ - ToolRegistry   │  │   Pipeline    │  │  │
│  │  │ - ToolMessage    │  │                  │  │                  │  │               │  │  │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘  └───────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                                SELF-HOSTED INFRASTRUCTURE                             │  │
│  │                                                                                       │  │
│  │  ┌───────────────────────────────────────────────┐   ┌─────────────────────────────┐  │  │
│  │  │               REDIS CORE                      │   │       PGVECTOR STORE        │  │  │
│  │  │ - Async Redis Connection Pool                 │   │ - Vector Similarity Search  │  │  │
│  │  │ - RedisChatMessageHistory (Sliding Buffer)    │   │ - HNSW Cosine Index (<=>)   │  │  │
│  │  │ - Response & Translation Cache                │   │ - Document Chunk Repository │  │  │
│  │  │ - Rate Limiting & User State                  │   │                             │  │  │
│  │  └───────────────────────────────────────────────┘   └─────────────────────────────┘  │  │
│  │                                                                                       │  │
│  │  ┌───────────────────────────────────────────────┐   ┌─────────────────────────────┐  │  │
│  │  │               LOCAL LLM                       │   │       LOCAL EMBEDDER        │  │  │
│  │  │ - Ollama Chat Client (llama3.1:8b)            │   │ - SentenceTransformers     │  │  │
│  │  │ - Unified Async Stream Generator              │   │   (multilingual-mpnet)      │  │  │
│  │  └───────────────────────────────────────────────┘   └─────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Chi tiết các Module Core cần xây dựng

### Module 1: LangChain-Standard Primitives (`app/core/primitives/`)
1. **Message Abstraction (`messages.py`)**:
   - `BaseMessage`: Lớp cơ sở chứa `content: str`, `additional_kwargs: dict`.
   - `SystemMessage`: Chỉ dẫn cho AI (nhân vật, vai trò, chủ đề).
   - `HumanMessage`: Câu nói của người học.
   - `AIMessage`: Phản hồi của trợ lý AI (hỗ trợ lưu trữ tool calls).
   - `ToolMessage`: Kết quả trả về sau khi thực thi 1 công cụ (`tool_call_id`, `name`, `content`).
   - Helper chuyển đổi qua lại giữa `BaseMessage` ➔ format Ollama / OpenAI dict.

2. **Prompt Template Engine (`prompts.py`)**:
   - `ChatPromptTemplate`: Định nghĩa danh sách các message template, hỗ trợ inject biến linh hoạt:
     ```python
     template = ChatPromptTemplate.from_messages([
         ("system", "You are an English Speaking Partner for level {user_level}."),
         ("system", "User weaknesses to watch out: {user_weaknesses}"),
         ("history", "{chat_history}"),
         ("human", "{user_input}")
     ])
     ```

3. **Tool Base Interface (`tools.py`)**:
   - `BaseTool`: Kế thừa `ABC`, bao gồm:
     - `name: str`: Định danh duy nhất (e.g. `dictionary_lookup`, `grammar_checker`).
     - `description: str`: Hướng dẫn cho LLM biết khi nào cần dùng tool.
     - `args_schema: type[BaseModel]`: Pydantic schema mô tả rõ ràng tham số đầu vào.
     - `async def arun(self, **kwargs) -> str`: Hàm thực thi bất đồng bộ.
   - `ToolRegistry`: Quản lý danh sách các tool khả dụng và tự động sinh system prompt hướng dẫn tool calling cho LLM.

---

### Module 2: Self-Hosted Redis Core (`app/core/redis/`)
Tự code toàn bộ tầng Redis, không dùng service đám mây của bên ngoài.

1. **Redis Connection Manager (`client.py`)**:
   - Sử dụng `redis.asyncio.ConnectionPool`.
   - Quản lý lifecycle qua FastAPI lifespan (`connect()` khi startup, `disconnect()` khi shutdown an toàn).
   - Xử lý auto-reconnect và health check (`PING`).

2. **Redis Chat Message History (`memory.py`)**:
   - Lưu trữ N lượt tin nhắn gần nhất của phiên trò chuyện vào Redis List (`LPUSH` / `LRANGE`).
   - Key format: `langai:conversation:{conversation_id}:messages`.
   - TTL tự động (e.g. 24h) để tiết kiệm RAM.
   - **Tác dụng**: Giảm tải 90% truy vấn `SELECT` vào PostgreSQL mỗi khi user chat, giúp phản hồi streaming tức thì. Khi kết thúc phiên hoặc theo định kỳ, tin nhắn vẫn được lưu vĩnh viễn vào PostgreSQL.

3. **Redis Cache Manager (`cache.py`)**:
   - `get(key)`, `set(key, value, ttl)`, `delete(key)`.
   - Hỗ trợ lưu cache kết quả tra từ điển (`langai:dict:{word}`) và giải thích ngữ pháp để tránh gọi LLM lặp lại cho các từ phổ biến.

---

### Module 3: Chuẩn hóa Ports & Adapters (`app/core/interfaces/` & `app/adapters/`)
1. **Giao diện Port đồng bộ với Primitives**:
   - `ILLMProvider`: Nhận `list[BaseMessage]`, trả về `AIMessage` hoặc `AsyncGenerator[str, None]`.
   - `IVectorStore`: Tìm kiếm vector, trả về `list[RetrievedChunk]`.
   - `IEmbedder`: Tạo vector embedding (HuggingFace / Ollama).
2. **Dọn dẹp triệt để thư mục cũ**:
   - Xóa bỏ folder `app/rag/` cũ (vốn là code prototype gộp chung).
   - Di chuyển các logic cần thiết (retriever, chunker) vào `app/adapters/` và `app/services/ingest_service.py`.

---

## 4. Kế hoạch Triển khai Từng bước (Step-by-Step Execution Plan)

| Bước | Nội dung công việc | Output cụ thể |
| :--- | :--- | :--- |
| **Bước 1** | **Xây dựng LangChain Primitives** | File `app/core/primitives/messages.py`, `prompts.py`, `tools.py` |
| **Bước 2** | **Xây dựng Redis Core Module** | File `app/core/redis/client.py`, `app/core/redis/memory.py`, `app/core/redis/cache.py` |
| **Bước 3** | **Tích hợp Redis vào FastAPI Lifespan & Config** | Cập nhật `app/main.py` (lifespan connect/disconnect Redis) |
| **Bước 4** | **Cập nhật Ports & Adapters theo Primitives mới** | Cập nhật `ILLMProvider`, `OllamaLLM`, `HuggingFaceLLM` dùng `BaseMessage` |
| **Bước 5** | **Refactor Ingest & Document Services** | Chuẩn hóa `document_service.py` (chỉ lo DB & File) và `ingest_service.py` (lo chunk/embed), loại bỏ `app/rag/` cũ |
| **Bước 6** | **Nối Redis Memory vào Chat Pipeline** | Cập nhật luồng chat: Đọc context nhanh từ Redis, lưu song song Postgres |

---

## 5. Rủi ro & Giải pháp Phòng ngừa (Risks & Mitigation)

1. **Rủi ro Redis Container bị tắt / mất kết nối**:
   - *Nguy cơ*: Hệ thống crash nếu Redis không khả dụng.
   - *Giải pháp*: Xây dựng cơ chế **Graceful Fallback**. Nếu Redis gặp lỗi, hệ thống tự động fallback đọc/ghi trực tiếp vào PostgreSQL mà không ngắt phiên chat của user.
2. **Rủi ro Local LLM (Ollama Llama 3.1) gọi Tool không chuẩn JSON**:
   - *Nguy cơ*: Khác với GPT-4, các model local đôi khi sinh text lẫn với JSON arguments của tool.
   - *Giải pháp*: Xây dựng Robust Output Parser với cơ chế Regex trích xuất JSON fallback trong `BaseTool` / `AgentEngine`.
3. **Rủi ro Đồng bộ Dữ liệu (Redis vs PostgreSQL)**:
   - *Nguy cơ*: Dữ liệu trong Redis mất khi container restart đột ngột (dù đã bật AOF).
   - *Giải pháp*: Quy ước PostgreSQL luôn là **Single Source of Truth** (lưu vĩnh viễn), Redis chỉ đóng vai trò **Fast Cache & Buffer**.
