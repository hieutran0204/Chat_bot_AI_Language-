# 🧠 Brainstorm Master — Language AI Backend

> **Mode**: BRAINSTORM  
> **Cập nhật**: 2026-09-27  
> **Phiên bản hiện tại**: Sprint 2 hoàn tất — Hệ thống Cá nhân hóa & Đo lường Độ trễ sẵn sàng.

---

## 1. Tầm nhìn & Mục tiêu sản phẩm

**Language AI Backend** là máy chủ phục vụ ứng dụng **Gia sư Tiếng Anh luyện nói bằng Giọng nói (Voice-first AI English Speaking Tutor)** dành cho người học Việt Nam.

**3 bài toán cốt lõi đang giải quyết:**

| Bài toán | Giải pháp hiện tại | Trạng thái |
| :--- | :--- | :---: |
| **Low-Latency Conversation** | Redis sliding-window buffer + SSE streaming | ✅ Hoàn tất |
| **Document RAG / Intent Retrieval** | pgvector + HuggingFace embeddings | ✅ Hoàn tất |
| **Learner Personalization** | 12-taxonomy weakness extraction + async background task | ✅ Hoàn tất |

---

## 2. Kiến trúc Hệ thống (Hexagonal / Ports & Adapters)

```
                          ┌───────────────────────────────────────┐
                          │          FastAPI Routers               │
                          │    /auth  /chat  /documents  /health   │
                          └─────────────────┬─────────────────────┘
                                            │
                                            ▼
                          ┌───────────────────────────────────────┐
                          │          Application Services          │
                          │  ChatService      ExtractionService    │
                          │  DocumentService  ProfileLoader        │
                          └──────┬───────────────┬────────────────┘
                                 │               │
          ┌──────────────────────┼───────────────┼─────────────────────────┐
          ▼                      ▼               ▼                         ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  ILLMProvider    │  │  IVectorStore    │  │  ILearnerMemory  │  │  IRagPipeline    │
│  (OllamaLLM,     │  │  (PgVectorStore) │  │  (Redis Chat +   │  │  (NaiveRAG,      │
│   HuggingFaceLLM)│  │                  │  │   Redis Profile) │  │   ConversationRAG│
└──────────────────┘  └──────────────────┘  └──────────────────┘  └──────────────────┘
```

### Cấu trúc thư mục hiện tại (đã ổn định)

```
backend/app/
├── core/
│   ├── config.py                    # Pydantic-Settings (từ .env)
│   ├── database.py                  # SQLAlchemy AsyncEngine + Session
│   ├── security.py                  # JWT encode/decode, bcrypt
│   ├── profiler.py                  # Latency profiler (TTFT, retrieval, embedding)
│   ├── interfaces/                  # Ports (Pure Python ABCs)
│   │   ├── chunker.py               # IChunker
│   │   ├── embedder.py              # IEmbedder
│   │   ├── llm_provider.py          # ILLMProvider
│   │   ├── memory.py                # IMemoryStore, ILearnerProfileStore
│   │   ├── pipeline.py              # IRagPipeline
│   │   ├── vector_store.py          # IVectorStore, RetrievedChunk
│   │   └── voice.py                 # ISpeechToText, ITextToSpeech (placeholder)
│   ├── primitives/                  # Domain Value Objects
│   │   ├── messages.py              # BaseMessage, HumanMessage, AIMessage (+ Correction, WeaknessType)
│   │   ├── prompts.py               # ChatPromptTemplate
│   │   └── tools.py                 # BaseTool, ToolCall, ToolResult
│   └── redis/                       # Self-hosted Redis modules
│       ├── client.py                # Async connection pool
│       ├── cache.py                 # RedisCacheManager (TTL cache)
│       ├── memory.py                # RedisChatMessageHistory (IMemoryStore)
│       └── profile.py               # RedisLearnerProfileManager (ILearnerProfileStore)
│
├── adapters/                        # Driven Adapters (Port implementations)
│   ├── chunkers/recursive_chunker.py
│   ├── embedders/huggingface_embedder.py, ollama_embedder.py
│   ├── llm/ollama_llm.py, huggingface_llm.py
│   └── vector_stores/pgvector_store.py
│
├── pipelines/                       # RAG Pipeline orchestration
│   ├── base_pipeline.py             # BasePipeline (prompt builder, context formatter)
│   └── naive_rag/pipeline.py        # NaiveRagPipeline (retrieval → generation)
│
├── factory/
│   └── pipeline_factory.py          # create_pipeline() — DI entry point
│
├── services/
│   ├── chat_service.py              # Orchestrator: chat lifecycle, SSE streaming
│   ├── extraction_service.py        # Async background: weakness extraction (12 taxonomy)
│   ├── document_service.py          # Document CRUD (DB + file storage)
│   ├── ingest_service.py            # Chunk + embed + store vector
│   └── user_service.py             # User CRUD, level management
│
├── api/v1/
│   ├── chat.py                      # Thin router: JWT auth → ChatService
│   ├── documents.py                 # Thin router: DocumentService
│   ├── auth.py                      # JWT login/register
│   ├── users.py                     # User profile + weakness dashboard
│   └── health.py                    # GET /health (Redis + failure counters)
│
└── models/                          # SQLAlchemy ORM Models
    ├── user.py
    ├── conversation.py
    ├── message.py                   # + corrections JSONB, extraction_status, extraction_error
    ├── document.py + document_chunk.py
    ├── user_weakness_log.py         # 12-taxonomy weakness log
    ├── user_strength_log.py
    └── session_insight.py
```

---

## 3. 4 Chế độ Trò chuyện (ConversationMode)

| Mode | Hành vi | RAG? |
| :--- | :--- | :---: |
| `conversation` | Bạn luyện nói tự nhiên, câu ngắn, luôn hỏi lại | ❌ |
| `grammar` | Chuyên gia ngữ pháp, giải thích + ví dụ, so sánh tiếng Việt | ✅ |
| `vocabulary` | Nghĩa từ, collocation, từ đồng nghĩa, sắc thái | ✅ |
| `document_qa` | Hỏi đáp tài liệu cá nhân (PDF/DOCX), kèm nguồn trích dẫn | ✅ |

---

## 4. Hệ thống Cá nhân hóa Học viên

### 4.1 Bộ phân loại 12 nhóm lỗi (WeaknessType Taxonomy)

**Ngữ pháp (7):** `tense_past_simple` · `tense_present_perfect` · `subject_verb_agreement` · `article_usage` · `preposition` · `word_order` · `conditional`

**Từ vựng (3):** `wrong_word` · `false_friend` · `collocation`

**Phát âm (2):** `final_consonant` · `th_sound`

### 4.2 Luồng xử lý 2 chiều

```
[ĐỒNG BỘ — User không phải chờ]
User gửi tin ──► ProfileLoader (top 3 weaknesses từ Redis)
             ──► Inject vào System Prompt
             ──► LLM streaming (SSE)
             ──► Lưu message với extraction_status='pending'

[BẤT ĐỒNG BỘ — Background Task]
ExtractionService.extract_and_log()
  ├── Idempotency Guard (status='done' → skip)
  ├── LLM phân loại 12 taxonomy (temperature=0.0)
  ├── Ghi user_weakness_log + UPDATE messages.corrections
  └── Redis cache invalidation (DELETE profile keys)
```

---

## 5. Database Schema (PostgreSQL + pgvector)

```
users (id, email, username, hashed_pw, level A1-C2)
  │
  ├── conversations (id, user_id, mode, title)
  │       └── messages (id, conversation_id, role, content, audio_url,
  │                     stt_confidence, extraction_status, extraction_error,
  │                     corrections JSONB, sources JSONB)
  │
  ├── user_weakness_log (id, user_id, conversation_id, weakness_type,
  │                      original, suggestion, explanation, is_repeated)
  │
  ├── user_strength_log (id, user_id, conversation_id, example)
  │
  ├── session_insights (id, user_id, conversation_id unique,
  │                     score 0-100, memo, next_goal)
  │
  └── documents (id, user_id, filename, file_type, status)
          └── document_chunks (id, document_id, content, embedding vector(768),
                               chunk_index, metadata JSONB)
```

**Index quan trọng:**
- `ix_user_weakness_log_user_id_created_at` (composite) — query top điểm yếu siêu tốc
- `ivfflat` (embedding cosine) — pgvector similarity search

---

## 6. Cấu hình LLM & Stack Hạ tầng (100% Self-hosted)

| Thành phần | Công nghệ | Vai trò |
| :--- | :--- | :--- |
| **LLM Primary** | Ollama `llama3.1:8b` | Chat generation, tool calling |
| **Embedding** | HuggingFace `multilingual-mpnet-base-v2` | Document & query embedding |
| **LLM Fallback** | HuggingFace Inference API | Khi Ollama không khả dụng |
| **Vector Store** | PostgreSQL 16 + pgvector | Similarity search (cosine `<=>`) |
| **Short-term Memory** | Redis 7 (List, LPUSH/LRANGE) | Chat sliding window (24h TTL) |
| **Long-term Profile** | Redis 7 (Hash) + PostgreSQL | Learner weaknesses & level cache |

---

## 7. Đo lường Độ trễ (Latency Profiler)

Ghi vào `backend/logs/chat_latency.log` qua `app/core/profiler.py`:

| Metric | Mục tiêu |
| :--- | :--- |
| `db_user_msg_and_history` | < 10ms |
| `query_embedding` | 15–35ms |
| `vector_retrieval` | 10–30ms |
| `profile_load` (Redis hit) | 2–5ms |
| `prompt_preparation` | < 2ms |
| `llm_time_to_first_token` | 300–800ms |

---

## 8. Chuỗi Migration Alembic (trạng thái hiện tại)

1. `bd98b2e236e5` — Khởi tạo bảng hệ thống ban đầu
2. `c1a2b3d4e5f6` — Thêm voice fields + corrections vào `messages`
3. `a1b2c3d4e5f7` — Thêm 3 bảng analytics người học
4. **`b2c3d4e5f6a7` ← HEAD** — Chuẩn hóa `session_id→conversation_id`, composite index, `extraction_status`, `extraction_error`

---

## 9. Tính năng còn lại & Roadmap tiếp theo

### 🎯 Sprint 3 — Voice Integration (Chưa bắt đầu)
- [ ] Implement `ISpeechToText` adapter (Whisper local)
- [ ] Implement `ITextToSpeech` adapter (Piper TTS / Kokoro)
- [ ] Audio upload endpoint + `audio_url` wiring
- [ ] STT confidence threshold (`stt_confidence < 0.7` → hỏi lại user)

### 📊 Sprint 4 — Analytics & Quiz Engine
- [ ] API `GET /api/v1/users/me/analytics` — báo cáo tiến độ học
- [ ] `QuizGeneratorTool` — sinh 5–10 câu trắc nghiệm từ điểm yếu tích lũy
- [ ] Alembic: thêm bảng `generated_quizzes`
- [ ] Spaced Repetition System (SM-2 algorithm) cho từ vựng

### 🛠️ Bổ sung Tools (khi số lượng tool > 5)
- [ ] `DictionaryTool` (IPA, definition, collocations)
- [ ] `GrammarAnalyzerTool` (deep-dive single-sentence analysis)
- [ ] `ToolRegistry` (dynamic loading khi tool count > 5)

---

## 10. Rủi ro & Giải pháp Phòng ngừa

| Rủi ro | Mức độ | Giải pháp |
| :--- | :---: | :--- |
| Redis mất kết nối | 🟡 Medium | Graceful fallback về PostgreSQL (đã implement) |
| Local LLM sinh tool-call JSON sai | 🟡 Medium | Regex fallback parser trong `BaseTool` |
| Redis profile hash phình to | 🟡 Medium | Giữ top 10 điểm yếu, decay lỗi cũ |
| Audio field null gây crash pipeline | 🟡 Medium | Guard `audio_url is not None` toàn bộ pipeline |
| Extraction background task retry storm | 🔴 High | Idempotency Guard (`status='done'` → skip) — đã implement |
| LLM hallucination trong doc_qa | 🔴 High | Strict prompt: "Answer ONLY from context", hiển thị sources |
