# 🧱 Core Build Plan: AI Speaking Tutor (Voice-First, Self-Hosted)

> **Mục tiêu sản phẩm**: Gia sư tiếng Anh dạng AI Agent, nói chuyện như một người bạn để rèn luyện Speaking.
> **Nguyên tắc**: Build core trước — nhưng core phải được định nghĩa đúng theo domain (voice conversation + learner memory), không phải core LLM-app chung chung.
> **Ngày lập**: 2026-09-12

---

## 0. Nguyên tắc thiết kế Core lần này

1. **Voice là first-class citizen ngay từ schema đầu tiên**, không phải thứ thêm vào sau. Nếu giờ chỉ code `content: str` rồi sau mới thêm audio field, toàn bộ chỗ serialize/deserialize sẽ phải sửa lại.
2. **Short-term memory (Redis) và Long-term learner profile là 2 khái niệm khác nhau**, phải tách key/schema từ đầu — vì đây chính là thứ phân biệt "gia sư" với "chatbot thường".
3. **Tool interface giữ tối giản** — không build `ToolRegistry` tổng quát khi mới có 2-3 tool cố định. Defer phần này tới khi thực sự cần nhiều tool động.
4. **Correction strategy phải chốt trước khi code chat pipeline**, vì nó quyết định schema của `AIMessage`.

---

## 1. Thứ tự xây dựng Core (Step-by-Step)

### Bước 1 — Message Primitives (`app/core/primitives/messages.py`)
Xây trước tiên vì mọi thứ khác phụ thuộc vào đây.

- `BaseMessage`:
  ```python
  content: str
  audio_url: str | None = None       # bản ghi âm gốc (user nói hoặc bot đọc)
  audio_duration_ms: int | None = None
  additional_kwargs: dict = {}
  ```
- `HumanMessage`: thêm `stt_confidence: float | None` (độ tin cậy STT, để biết khi nào cần hỏi lại "bạn nói lại được không?")
- `AIMessage`: thêm `corrections: list[Correction] | None` — tách riêng lỗi ngữ pháp/phát âm ra khỏi `content` chính, để UI có thể hiển thị "gợi ý sửa" riêng mà không làm gián đoạn câu trả lời tự nhiên của bot.
- `ToolMessage`: giữ như bản gốc (`tool_call_id`, `name`, `content`).
- Định nghĩa `Correction` (Pydantic model): `original: str`, `suggestion: str`, `type: Literal["grammar", "pronunciation", "vocab"]`.

**Output**: `messages.py` hoàn chỉnh, có unit test convert `BaseMessage` ↔ dict cho Ollama.

---

### Bước 2 — Redis Core: 2 tầng bộ nhớ tách biệt (`app/core/redis/`)

**2a. Short-term: Chat Sliding Memory (`memory.py`)**
- Giữ nguyên thiết kế gốc: `LPUSH`/`LRANGE`, key `langai:conversation:{id}:messages`, TTL 24h.
- Không đổi gì so với bản trước — vẫn hợp lý cho session hiện tại.

**2b. Long-term: Learner Profile (`profile.py`) — MỚI, cần thêm ngay ở bước này**
- Key format:
  ```
  langai:profile:{user_id}:level          # string, e.g. "B1"
  langai:profile:{user_id}:weaknesses     # hash: {"past_tense": 5, "th_sound": 3, ...}
  langai:profile:{user_id}:last_session   # timestamp
  ```
- Không TTL (hoặc TTL rất dài) — vì đây là dữ liệu học tập lâu dài, khác bản chất với chat buffer.
- Đây chính là data để inject vào `ChatPromptTemplate` (`{user_level}`, `{user_weaknesses}`) ở Bước 1 của plan cũ.
- PostgreSQL vẫn là nguồn sự thật cuối cùng — Redis chỉ cache bản tổng hợp mới nhất để đọc nhanh mỗi khi mở phiên chat.

**2c. Cache Manager (`cache.py`)**: giữ nguyên như bản gốc.

**Output**: `client.py`, `memory.py`, `profile.py`, `cache.py`.

---

### Bước 3 — Prompt Template Engine (`app/core/primitives/prompts.py`)
- `ChatPromptTemplate` như bản gốc, nhưng thêm 1 template chuyên biệt cho persona "bạn nói chuyện" thay vì "trợ lý":
  ```python
  CONVERSATION_PERSONA = ChatPromptTemplate.from_messages([
      ("system", "You are a friendly English-speaking friend, not a formal assistant. "
                  "Keep responses short and conversational, like real speech."),
      ("system", "User level: {user_level}. Watch for: {user_weaknesses}."),
      ("system", "Correction style: {correction_style}"),  # xem Bước 4
      ("history", "{chat_history}"),
      ("human", "{user_input}")
  ])
  ```
- Đây là chỗ chốt luôn **tone/persona** — quyết định sản phẩm, nên làm sớm chứ không để tới lúc chat pipeline mới nghĩ.

---

### Bước 4 — Quyết định Correction Strategy (chốt trước khi code pipeline)
Không phải code, mà là 1 quyết định thiết kế cần ghi lại trong core:

- **Option A — Real-time inline**: bot sửa lỗi ngay trong câu trả lời (ngắt flow tự nhiên nhưng học nhanh).
- **Option B — Silent tracking**: bot không sửa giữa chừng, chỉ ghi nhận lỗi vào `corrections` field, tổng hợp feedback cuối buổi (giữ hội thoại tự nhiên hơn, đúng tinh thần "nói chuyện như bạn bè").

→ Khuyến nghị: **Option B làm mặc định**, có `correction_style` là biến trong prompt để dễ đổi sau. Quyết định này ảnh hưởng trực tiếp lên field `corrections` đã thêm ở Bước 1.

**Output**: 1 đoạn ghi chú ADR (Architecture Decision Record) ngắn trong `docs/decisions/correction_strategy.md`.

---

### Bước 5 — Tool Base Interface tối giản (`app/core/primitives/tools.py`)
- `BaseTool` (ABC) giữ như bản gốc: `name`, `description`, `args_schema`, `async def arun()`.
- **Không build `ToolRegistry` tổng quát** ở giai đoạn này. Thay vào đó hardcode danh sách 2-3 tool cố định trực tiếp trong Agent Engine:
  - `dictionary_lookup`
  - `grammar_checker`
  - `pronunciation_feedback` (placeholder, chưa cần implement đầy đủ)
- Khi nào số lượng tool tăng lên (>5-6 tool, hoặc cần dynamic loading), mới quay lại build `ToolRegistry`.

---

### Bước 6 — Ports & Adapters (`app/core/interfaces/`)
- Giữ như bản gốc: `ILLMProvider`, `IVectorStore`, `IEmbedder`.
- **Thêm mới**: `ISpeechToText` và `ITextToSpeech` port, kể cả khi chưa implement adapter thật (Whisper/Piper) — để interface đã sẵn sàng, tránh phải sửa core khi thêm voice sau:
  ```python
  class ISpeechToText(ABC):
      async def transcribe(self, audio_bytes: bytes) -> tuple[str, float]:  # (text, confidence)
          ...

  class ITextToSpeech(ABC):
      async def synthesize(self, text: str) -> bytes:
          ...
  ```

---

### Bước 7 — Tích hợp vào FastAPI Lifespan & Config
- Redis connect/disconnect trong lifespan như bản gốc.
- Thêm placeholder connection cho STT/TTS service (dù chưa dùng thật, khai báo trong config để không phải sửa cấu trúc app sau này).

---

### Bước 8 — Nối Redis Memory + Learner Profile vào Chat Pipeline
- Đọc `chat_history` từ Redis sliding window (như bản gốc).
- Đọc `user_level`, `user_weaknesses` từ `langai:profile:{user_id}` để inject vào prompt template (Bước 3).
- Sau mỗi turn: cập nhật `corrections` vào profile nếu lỗi lặp lại (tăng counter trong `weaknesses` hash).
- Lưu song song xuống PostgreSQL theo chu kỳ (giữ nguyên nguyên tắc PostgreSQL = single source of truth).

---

## 2. Điều gì được lược bỏ / defer ra khỏi Core lần này

| Hạng mục | Lý do defer |
| :--- | :--- |
| `pgvector` + HNSW vector search | Không phải core loop của speaking practice; chỉ cần khi làm tính năng tra cứu tài liệu/RAG phụ trợ sau này |
| `ToolRegistry` tự sinh system prompt | Premature abstraction khi mới có 2-3 tool cố định |
| STT/TTS adapter thật (Whisper, Piper) | Chưa implement, nhưng **interface đã có sẵn** ở Bước 6 để không phải sửa core sau |

---

## 3. Rủi ro & Giải pháp (giữ nguyên + bổ sung)

1. **Redis mất kết nối** → Graceful fallback về PostgreSQL (giữ nguyên từ plan gốc).
2. **Local LLM sinh tool-call sai JSON** → Regex fallback parser trong `BaseTool` (giữ nguyên).
3. **Redis vs PostgreSQL đồng bộ** → PostgreSQL là Single Source of Truth (giữ nguyên).
4. **MỚI — Audio field để null quá lâu**: nếu giai đoạn đầu chỉ làm text-only, cần đảm bảo toàn bộ pipeline vẫn chạy đúng khi `audio_url = None`, tránh để code giả định audio luôn tồn tại.
5. **MỚI — Learner profile phình to không kiểm soát**: `weaknesses` hash cần có cơ chế decay hoặc giới hạn số lượng key (ví dụ chỉ giữ top 10 lỗi phổ biến nhất) để tránh Redis hash phình vô hạn theo thời gian.

---

## 4. Tóm tắt thứ tự ưu tiên

```
1. messages.py (có audio + corrections field)
2. redis/memory.py + redis/profile.py (2 tầng bộ nhớ)
3. prompts.py (persona "bạn nói chuyện")
4. ADR: correction strategy
5. tools.py (tối giản, không ToolRegistry)
6. interfaces/ (thêm ISpeechToText, ITextToSpeech placeholder)
7. FastAPI lifespan integration
8. Nối Redis + Profile vào Chat Pipeline
```
