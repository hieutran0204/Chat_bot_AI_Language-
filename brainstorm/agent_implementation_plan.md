# 📋 Implementation Plan: AI Agent Platform (Speaking, Memory & Tools)

> **Mode**: BRAINSTORM / PLANNING  
> **Ngày lập**: 2026-09-05  
> **Mục tiêu**: Xây dựng toàn bộ hệ thống AI Agent học tiếng Anh bao gồm Speaking Partner theo chủ đề, Long-Term Memory ghi nhớ điểm yếu, và Agent Toolset (Từ điển, Sửa lỗi, Quiz tự động).

---

## 1. Kiến trúc Tổng thể & Phân rã Component

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FASTAPI BACKEND AGENT                             │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                            AGENT ENGINE                               │  │
│  │  - Intent Classifier & Tool Router                                    │  │
│  │  - ReAct Execution Loop                                               │  │
│  └───────┬───────────────────────────────┬───────────────────────────────┘  │
│          │                               │                                  │
│          ▼                               ▼                                  ▼
│  ┌───────────────┐               ┌───────────────┐               ┌───────────────┐  │
│  │ SPEAKING      │               │ MEMORY        │               │ TOOLSET       │  │
│  │ PARTNER AGENT │               │ EXTRACTOR     │               │ EXECUTOR      │  │
│  │ - Topic Role  │               │ - Background  │               │ - Dictionary  │  │
│  │ - Gentle Fix  │               │   Extractor   │               │ - Grammar Fix │  │
│  └───────────────┘               │ - SRS Engine  │               │ - Auto Quiz   │  │
│                                  └───────────────┘               │ - RAG Vector  │  │
│                                                                  └───────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Kế hoạch Chi tiết theo 3 Phase

### 🚀 PHASE 1: Agent Core Engine, Speaking Partner & Basic Toolset
> **Mục tiêu**: Dựng khung Agent Engine, cho phép luyện nói theo chủ đề và chạy các Tool cơ bản (Từ điển, Sửa lỗi).

#### Task 1.1: Thiết kế Core Interfaces cho Tools & Agent
- Tạo interface `ITool` (`name`, `description`, `parameters`, `execute()`).
- Tạo class `ToolRegistry` để quản lý và kích hoạt các Tool tự động.
- Tạo class `AgentEngine` điều phối luồng suy nghĩ ReAct.

#### Task 1.2: Xây dựng Bộ Toolset cơ bản
- **`DictionaryTool`**:
  - Tích hợp Free Dictionary API (hoặc WordNet / Ollama Dictionary Prompt).
  - Trả về: Phiên âm IPA, từ loại, định nghĩa, ví dụ câu, collocations.
- **`GrammarAnalyzerTool`**:
  - Nhận vào 1 câu tiếng Anh ➔ Soi lỗi sai chi tiết (Spelling, Tense, Article, Preposition).
  - Đưa ra giải thích tiếng Việt ngắn gọn + 3 cách diễn đạt tự nhiên chuẩn bản xứ.
- **`RagRetrievalTool`**:
  - Đóng gói `PgVectorStore` thành 1 Tool để Agent tự quyết định khi nào cần gọi tra cứu tài liệu.

#### Task 1.3: Speaking Partner Agent (Luyện nói theo chủ đề)
- Thêm mode `speaking_topic` vào `ConversationMode`.
- Tạo cấu trúc Prompt linh hoạt cho các chủ đề (*Job Interview, Travel, Airport, Daily Talk, IELTS*).
- Tích hợp **Real-time Gentle Correction**: Tự động chèn 1 đoạn nhận xét nhỏ ở cuối phản hồi nếu phát hiện lỗi sai nghiêm trọng.

---

### 💾 PHASE 2: Long-Term Memory & Spaced Repetition (SRS)
> **Mục tiêu**: Ghi nhớ thói quen, điểm yếu ngữ pháp và từ vựng của từng User qua thời gian.

#### Task 2.1: Mở rộng CSDL & Database Models
- Bổ sung bảng CSDL `user_memories`: Lưu vết các điểm yếu (ví dụ: *Hay quên chia động từ Quá khứ*, *Hay nhầm lẫn giữa Since/For*).
- Bổ sung bảng CSDL `user_vocabulary`: Lưu danh sách từ vựng kèm chỉ số ôn tập SRS (`mastery_level`, `next_review_at`).
- Tạo Alembic migration script để cập nhật DB.

#### Task 2.2: Memory Extractor Agent (Trích xuất ký ức bất đồng bộ)
- Xây dựng background task chạy sau mỗi lượt chat (không làm chậm tốc độ phản hồi của User).
- Phân tích đoạn chat ➔ Rút ra các lỗi sai ngữ pháp lặp lại hoặc từ vựng mới ➔ Lưu/Cập nhật vào `user_memories`.

#### Task 2.3: Spaced Repetition System (SRS Engine)
- Áp dụng thuật toán tính khoảng cách thời gian ôn tập (SuperMemo SM-2 hoặc Leitner).
- Tự động tính toán ngày cần ôn tập tiếp theo cho từng từ vựng trong `user_vocabulary`.
- Inject thông tin `user_memories` vào System Prompt của Agent ở các phiên chat tiếp theo để Agent "nhớ mặt điểm tên" điểm yếu của User.

---

### 🎯 PHASE 3: Automated Quiz Generator & Personalized Dashboard
> **Mục tiêu**: Tự động sinh bài tập cá nhân hoá dựa trên lỗi sai quá khứ và cung cấp báo cáo tiến độ.

#### Task 3.1: Model & Engine Sinh Quiz Tự động (`QuizGeneratorAgent`)
- Bổ sung bảng CSDL `generated_quizzes` (lưu danh sách câu hỏi, đáp án, câu giải thích).
- Xây dựng Agent tự đọc `user_memories` (điểm yếu) ➔ Tự thiết kế bộ bài tập 5-10 câu (Trắc nghiệm, Điền từ, Sắp xếp câu).
- API `POST /api/v1/quizzes/generate`: Cho phép User yêu cầu tạo bài tập ôn luyện bất cứ lúc nào.
- API `POST /api/v1/quizzes/{id}/submit`: Chấm điểm bài làm của User ➔ Cập nhật chỉ số `mastery_level` trong `user_vocabulary`.

#### Task 3.2: Analytics & Progress Dashboard API
- API `GET /api/v1/users/me/analytics`: Trả về báo cáo toàn diện:
  - Danh sách top 5 điểm yếu ngữ pháp cần khắc phục.
  - Số từ vựng đã thuộc vs. Số từ vựng cần ôn tập hôm nay (SRS Due).
  - Biểu đồ điểm số Quiz qua các tuần.

---

## 3. Lịch trình Thực hiện (Timeline & Checkpoints)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: Agent Core Engine, Speaking Partner & Tools (Tuần 1 - 2)          │
│  - Tool System (Dictionary, Grammar, RAG)                                   │
│  - Topic Speaking Mode + Gentle Correction                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  PHASE 2: Long-Term Memory & SRS Engine (Tuần 3 - 4)                        │
│  - DB Migration (user_memories, user_vocabulary)                            │
│  - Async Memory Extractor & Spaced Repetition Algorithm                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  PHASE 3: Automated Quiz Generator & Analytics (Tuần 5 - 6)                 │
│  - Auto Quiz Generator from Weaknesses                                      │
│  - Quiz Submit/Grading API + Progress Analytics                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Tiêu chuẩn Đánh giá Hoàn thành (Definition of Done)

1. **Speaking Partner**: User nói chuyện mượt mà theo bất kỳ chủ đề lựa chọn nào, nhận được nhận xét sửa lỗi nhẹ nhàng.
2. **Toolset**: Gọi được Tool tra từ điển chuẩn IPA và Tool phân tích ngữ pháp chuyên sâu.
3. **Memory**: Hệ thống ghi nhận được điểm yếu vào DB sau các buổi chat và nhớ lại điểm yếu đó ở lần chat tiếp theo.
4. **Auto Quiz**: Sinh ra bộ câu hỏi trắc nghiệm sát với chính các lỗi sai mà User từng mắc phải.
