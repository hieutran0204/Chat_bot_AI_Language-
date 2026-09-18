# 🧠 Brainstorm: RAG Chatbot Học Tiếng Anh

> **Mode**: BRAINSTORM — Không có source code nào được tạo ở bước này.
> **Ngày**: 2026-09-04

---

## 1. Tổng quan mục tiêu

Xây dựng một AI Chatbot phục vụ học tiếng Anh với 3 tính năng chính:

| Tính năng | Mô tả |
|-----------|-------|
| 🗣️ Conversation Practice | Chatbot đóng vai partner, luyện hội thoại theo chủ đề |
| 📚 Grammar & Vocabulary | Giải thích ngữ pháp, từ vựng theo ngữ cảnh của user |
| 🔍 Document Q&A (RAG) | Hỏi đáp dựa trên tài liệu học (sách, giáo trình, PDF) |

**Đặc điểm kỹ thuật:**
- Backend: **Python FastAPI**
- LLM Engine: **Ollama** (local) + **HuggingFace Inference API** (fallback/embedding)
- Vector Store: **PostgreSQL + pgvector**
- Frontend: **Flutter Mobile App**
- Kiến trúc: **RAG (Retrieval-Augmented Generation)**

---

## 2. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        FLUTTER MOBILE APP                        │
│   [Chat Screen] [Document Upload] [Progress Tracker] [Settings]  │
└─────────────────────────────┬────────────────────────────────────┘
                              │ HTTPS / WebSocket
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                     FASTAPI BACKEND                               │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────────┐ │
│  │ /chat       │  │ /documents  │  │ /auth / /users / /progress│ │
│  │ (RAG chain) │  │ (ingestion) │  │ (user management)        │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────────────────────┘ │
│         │                │                                        │
│  ┌──────▼──────────────────────────────────────────────────────┐ │
│  │                  RAG PIPELINE ENGINE                         │ │
│  │  [Query Preprocessing] → [Retrieval] → [Context Building]   │ │
│  │         → [LLM Generation] → [Post-processing]              │ │
│  └──────┬─────────────────────────────────┬────────────────────┘ │
└─────────┼─────────────────────────────────┼──────────────────────┘
          │                                 │
          ▼                                 ▼
┌─────────────────────┐        ┌────────────────────────────┐
│   LLM LAYER         │        │    VECTOR STORE             │
│  ┌───────────────┐  │        │  PostgreSQL + pgvector       │
│  │ Ollama (local)│  │        │  - documents table           │
│  │ - Llama 3.1   │  │        │  - document_chunks table     │
│  │ - Mistral     │  │        │  - embeddings (vector)       │
│  │ - Gemma 2     │  │        │  - users table               │
│  └───────────────┘  │        │  - conversations table       │
│  ┌───────────────┐  │        │  - messages table            │
│  │ HuggingFace   │  │        └────────────────────────────┘
│  │ (embedding +  │  │
│  │  fallback LLM)│  │
│  └───────────────┘  │
└─────────────────────┘
```

---

## 3. Chi tiết từng Component

### 3.1 FastAPI Backend — Cấu trúc thư mục

```
language_ai_backend/
├── app/
│   ├── api/
│   │   ├── v1/
│   │   │   ├── chat.py          # Chat endpoints (RAG query)
│   │   │   ├── documents.py     # Upload, list, delete documents
│   │   │   ├── auth.py          # JWT auth
│   │   │   ├── users.py         # User profile & progress
│   │   │   └── health.py        # Health check
│   ├── core/
│   │   ├── config.py            # Settings (env vars)
│   │   ├── security.py          # JWT, password hashing
│   │   └── database.py          # SQLAlchemy + asyncpg setup
│   ├── models/
│   │   ├── user.py
│   │   ├── document.py
│   │   ├── conversation.py
│   │   └── message.py
│   ├── schemas/                 # Pydantic schemas
│   ├── rag/
│   │   ├── pipeline.py          # Main RAG orchestration
│   │   ├── embedder.py          # Embedding (HuggingFace / Ollama)
│   │   ├── retriever.py         # pgvector similarity search
│   │   ├── chunker.py           # Document chunking strategy
│   │   ├── reranker.py          # Optional: cross-encoder reranking
│   │   └── llm_client.py        # Ollama + HuggingFace client
│   ├── services/
│   │   ├── document_service.py  # Document ingestion service
│   │   ├── chat_service.py      # Conversation management
│   │   └── user_service.py      # User & progress logic
│   └── main.py
├── alembic/                     # DB migrations
├── tests/
├── .env
├── docker-compose.yml
└── requirements.txt
```

### 3.2 RAG Pipeline — 2 luồng chính

#### Luồng 1: Document Ingestion (Nạp tài liệu)

```
[PDF/DOCX/TXT Upload]
        │
        ▼
[Document Parser]
  - PyMuPDF (PDF)
  - python-docx (DOCX)
  - Unstructured.io (mixed formats)
        │
        ▼
[Text Chunker]
  Strategy: RecursiveCharacterTextSplitter
  - chunk_size: 512 tokens
  - chunk_overlap: 64 tokens
  - Giữ nguyên câu, không cắt giữa chừng
        │
        ▼
[Embedding Model]
  Primary: HuggingFace — sentence-transformers/paraphrase-multilingual-mpnet-base-v2
  (hỗ trợ cả Tiếng Anh + Tiếng Việt trong metadata)
  Fallback: Ollama nomic-embed-text
        │
        ▼
[Store vào PostgreSQL/pgvector]
  - document_chunks (content, embedding vector, metadata)
```

#### Luồng 2: RAG Query (Chat thực tế)

```
[User gửi câu hỏi]
        │
        ▼
[Query Preprocessing]
  - Detect intent: grammar_explain | vocab | conversation | document_qa
  - Nếu là hội thoại thuần: bypass retrieval, dùng conversation history
        │
        ▼ (nếu cần retrieval)
[Query Embedding]
  - Embed câu hỏi bằng cùng model với lúc ingestion
        │
        ▼
[Vector Similarity Search — pgvector]
  SELECT content, metadata, 1 - (embedding <=> query_vec) AS similarity
  FROM document_chunks
  ORDER BY similarity DESC
  LIMIT 5 (top-k)
        │
        ▼
[Context Building]
  - Ghép top-k chunks thành context string
  - Thêm conversation history (last 5 turns)
  - Inject vào System Prompt
        │
        ▼
[LLM Generation — Ollama]
  - Model mặc định: llama3.1:8b hoặc mistral:7b
  - Streaming response (Server-Sent Events)
        │
        ▼
[Post-processing]
  - Lưu message vào DB
  - Cập nhật progress tracking
  - Return response cho Flutter
```

---

## 4. Database Schema (PostgreSQL + pgvector)

```sql
-- Extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Users
CREATE TABLE users (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email       VARCHAR(255) UNIQUE NOT NULL,
    username    VARCHAR(100) NOT NULL,
    hashed_pw   TEXT NOT NULL,
    level       VARCHAR(20) DEFAULT 'beginner', -- A1/A2/B1/B2/C1/C2
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Documents (tài liệu người dùng upload)
CREATE TABLE documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID REFERENCES users(id) ON DELETE CASCADE,
    filename    TEXT NOT NULL,
    file_type   VARCHAR(20),   -- pdf, docx, txt
    status      VARCHAR(20) DEFAULT 'processing', -- processing | ready | failed
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Document Chunks (chunks đã được embed)
CREATE TABLE document_chunks (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    content     TEXT NOT NULL,
    embedding   vector(768),   -- 768 dims cho mpnet-base
    chunk_index INTEGER,
    metadata    JSONB          -- page_number, section, etc.
);

-- Index cho similarity search
CREATE INDEX ON document_chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Conversations
CREATE TABLE conversations (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID REFERENCES users(id) ON DELETE CASCADE,
    mode        VARCHAR(30),   -- conversation | grammar | vocabulary | doc_qa
    title       TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Messages
CREATE TABLE messages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role            VARCHAR(10),  -- user | assistant
    content         TEXT NOT NULL,
    sources         JSONB,        -- retrieved chunks used (for doc_qa)
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- User Progress
CREATE TABLE user_progress (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID REFERENCES users(id) ON DELETE CASCADE,
    date        DATE DEFAULT CURRENT_DATE,
    messages_sent INTEGER DEFAULT 0,
    vocab_learned INTEGER DEFAULT 0,
    study_minutes INTEGER DEFAULT 0
);
```

---

## 5. LLM Strategy: Ollama + HuggingFace

### Ollama (Local — Primary LLM)

| Model | Use case | VRAM |
|-------|----------|------|
| `llama3.1:8b` | Conversation, Q&A tổng quát | ~6 GB |
| `mistral:7b` | Grammar explanation | ~5 GB |
| `gemma2:9b` | Vocab & detailed explanation | ~7 GB |
| `nomic-embed-text` | Embedding (nếu không dùng HuggingFace) | ~0.3 GB |

### HuggingFace (Embedding — Primary + LLM Fallback)

| Model | Role |
|-------|------|
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | Embedding chính (có hỗ trợ Tiếng Việt) |
| `BAAI/bge-large-en-v1.5` | Embedding tiếng Anh chất lượng cao hơn |
| HuggingFace Inference API | Fallback nếu Ollama không khả dụng |

### Intent Routing Strategy

```
user_input
    │
    ├─ "Explain this grammar..."  → Grammar Mode (RAG + grammar prompt)
    ├─ "What does X mean..."      → Vocabulary Mode (dict lookup + LLM)
    ├─ "Let's practice talking..."→ Conversation Mode (no RAG, roleplay)
    └─ "In chapter 3, what is..." → Document QA Mode (full RAG)
```

---

## 6. System Prompts (Thiết kế prompt)

### Conversation Practice Prompt
```
You are an English conversation partner for a {level} level Vietnamese learner.
- Speak naturally but clearly
- Correct major grammar mistakes gently
- Keep responses under 3 sentences to encourage dialogue
- Ask follow-up questions to keep conversation going
```

### Grammar Explanation Prompt
```
You are an English grammar expert.
Context from study materials: {retrieved_context}
- Explain in simple terms, provide examples
- Compare with Vietnamese grammar if helpful
- Give 2-3 practice exercises
```

### Document Q&A Prompt
```
You are a helpful study assistant. Answer ONLY based on the provided context.
Context: {retrieved_context}
If the answer is not in the context, say "I couldn't find this in your materials."
```

---

## 7. Flutter App — Màn hình chính

```
App Structure:
├── AuthScreen (Login / Register)
├── HomeScreen (Dashboard, progress stats)
├── ChatScreen
│   ├── ModeSelector (Conversation / Grammar / Vocab / Doc QA)
│   ├── MessageList (streaming support)
│   └── InputBar + Attach Document
├── LibraryScreen (Quản lý tài liệu đã upload)
├── ProgressScreen (Charts: messages/day, vocab learned)
└── SettingsScreen (LLM model selection, level)
```

**Packages Flutter cần:**
- `dio` — HTTP client
- `flutter_chat_ui` — Chat UI component
- `file_picker` — Document upload
- `fl_chart` — Progress charts
- `flutter_secure_storage` — JWT storage

---

## 8. Rủi ro & Edge Cases

| Rủi ro | Mức độ | Giải pháp |
|--------|--------|-----------|
| Ollama chậm trên CPU | 🔴 High | Dùng quantized models (Q4), fallback HuggingFace API |
| Embedding dimension mismatch | 🟡 Medium | Lock embedding model, migration script khi đổi model |
| Context window overflow | 🟡 Medium | Giới hạn top-k=5, mỗi chunk max 512 tokens |
| User upload tài liệu rác | 🟡 Medium | Validate file type, size limit (20MB), async processing |
| HuggingFace rate limit (free tier) | 🟡 Medium | Cache embeddings trong DB, chỉ embed khi có tài liệu mới |
| Câu trả lời hallucination | 🔴 High | Strict prompt engineering, hiển thị sources cho doc_qa mode |
| Bảo mật JWT | 🟡 Medium | Refresh token rotation, short expiry (15 min access token) |

---

## 9. Lộ trình phát triển (3 Phase)

### Phase 1 — MVP (2-3 tuần)
- [ ] FastAPI boilerplate + PostgreSQL + pgvector setup
- [ ] Document ingestion pipeline (PDF, TXT)
- [ ] Basic RAG Q&A với Ollama
- [ ] Flutter chat screen (text only)
- [ ] JWT Auth

### Phase 2 — Core Features (3-4 tuần)
- [ ] Intent routing (4 modes)
- [ ] Conversation history & context management
- [ ] Document library management
- [ ] Flutter: Library screen, Progress screen
- [ ] HuggingFace embedding integration
- [ ] Streaming response (SSE)

### Phase 3 — Polish & Scale (2-3 tuần)
- [ ] Reranking (cross-encoder để cải thiện retrieval quality)
- [ ] User progress analytics
- [ ] Multi-document support (chat với nhiều tài liệu cùng lúc)
- [ ] Caching layer (Redis) cho embeddings thường dùng
- [ ] Docker Compose full stack

---

## 10. Open Questions cần xác nhận trước khi code

1. **Server deployment**: Dự án chạy local (development) hay cần deploy lên server (VPS/cloud)?
2. **Multi-user**: Hỗ trợ nhiều người dùng hay chỉ single-user (bản thân học)?
3. **Document scope**: Chỉ tài liệu user tự upload hay có sẵn kho tài liệu mặc định (Cambridge, Oxford)?
4. **Embedding model language**: Tài liệu học chủ yếu là tiếng Anh hay có cả giải thích bằng tiếng Việt?
5. **GPU availability**: Máy chạy Ollama có GPU không? (ảnh hưởng đến lựa chọn model size)
