# 🤖 AI Agent Architecture & Scope Design: English Learning Platform

> **Mode**: BRAINSTORM (Duyệt kiến trúc & Phạm vi dự án — Không tạo/sửa source code)  
> **Ngày**: 2026-09-05  
> **Mục tiêu**: Đẩy mạnh từ hệ thống RAG cơ bản lên **Multi-Capability AI Agent** tích hợp Speaking Partner theo chủ đề, Long-Term Memory đánh giá người học, và Bộ Toolset tự động (Tra từ điển, Tạo bài tập, Sửa lỗi).

---

## 1. Phân tích bài toán & Mục tiêu mới

Hệ thống cũ là RAG tĩnh (Hỏi ➔ Tra cứu ➔ Trả lời). Hệ thống mới chuyển đổi thành một **Multi-Agent System** có khả năng:

```
                               ┌──────────────────────────────────────────────┐
                               │           USER (FLUTTER APP)                 │
                               └──────────────────────┬───────────────────────┘
                                                      │
                                                      ▼
                               ┌──────────────────────────────────────────────┐
                               │               ORCHESTRATOR AGENT             │
                               │          (ReAct Loop & Intent Router)        │
                               └──────┬───────────────┬───────────────┬───────┘
                                      │               │               │
            ┌─────────────────────────┘               │               └─────────────────────────┐
            ▼                                         ▼                                         ▼
┌───────────────────────┐         ┌───────────────────────┐                 ┌───────────────────────┐
│ 🗣️ SPEAKING TUTOR     │         │ 💾 MEMORY AGENT       │                 │ 🛠️ AGENT TOOLSET      │
│ - Topic-based Talk    │         │ - Profile & Weakness  │                 │ - Dictionary Search   │
│ - Persona Adaptation  │         │ - Progress Tracking   │                 │ - Grammar Fixer       │
│ - Real-time Correct   │         │ - Spaced Repetition   │                 │ - Quiz Generator      │
└───────────────────────┘         └───────────────────────┘                 └───────────────────────┘
```

---

## 2. Chi tiết 3 Trụ cột Agent

### 🏛️ Trụ cột 1: Speaking Partner Agent (Luyện nói theo chủ đề)
- **Tính năng**:
  - Người dùng chọn hoặc tự đề xuất chủ đề trò chuyện (ví dụ: *Job Interview*, *Travel at Airport*, *Daily Routine*, *IELTS Speaking Part 2*).
  - Agent đóng vai nhân vật phù hợp (Interviewer, Receptionist, Friend) với phong thái giao tiếp tự nhiên.
  - **Dynamic Feedback Loop**: Trả lời câu hỏi ➔ Phân tích câu nói của User ➔ Đưa ra nhận xét ngắn nhẹ nhàng ở cuối tin nhắn (ví dụ: *"Good point! Notice that you said 'he go', it should be 'he goes'."*).

### 💾 Trụ cột 2: Long-Term Memory Agent (Hồ sơ & Đánh giá người học)
- **Tính năng**:
  - Lưu giữ **Ký ức ngắn hạn (Short-term context)**: Lịch sử cuộc trò chuyện hiện tại.
  - Lưu giữ **Ký ức dài hạn (Long-term memory)**:
    - **Lịch sử lỗi sai thường gặp**: Nhầm thì (Tenses), sai giới từ (Prepositions), nhầm lẫn từ đồng nghĩa.
    - **Vốn từ vựng cần ôn tập (Spaced Repetition)**: Các từ vựng đã học nhưng hay quên.
    - **Đánh giá trình độ liên tục (Dynamic Proficiency Rating)**: Cập nhật trình độ thực tế (A1 -> C2) dựa trên chỉ số từ vựng và ngữ pháp qua từng tuần.

### 🛠️ Trụ cột 3: Agent Toolset (Bộ công cụ phụ trợ)
- **Các Tool Agent có thể kích hoạt**:
  1. `dictionary_tool`: Tra cứu từ điển chuẩn (Từ loại, phiên âm IPA, định nghĩa, ví dụ, collocations).
  2. `grammar_analyzer_tool`: Phân tích chuyên sâu 1 câu của User (Chỉ ra lỗi sai, giải thích lý do, đưa ra 3 câu sửa lại tốt hơn).
  3. `quiz_generator_tool`: Tự động tạo bài tập (Trắc nghiệm, điền từ vào chỗ trống, sắp xếp câu) dựa chính trên **lỗi sai trong quá khứ** của User từ Long-Term Memory.
  4. `rag_retrieval_tool`: Tra cứu tài liệu PDF/DOCX đã upload.

---

## 3. Kiến trúc CSDL Bổ sung (Database Schema Additions)

Để hỗ trợ Long-Term Memory và Quiz/Tools, CSDL PostgreSQL sẽ được mở rộng thêm các bảng:

```sql
-- 1. Hồ sơ ký ức dài hạn người học (User Long-term Memory)
CREATE TABLE user_memories (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID REFERENCES users(id) ON DELETE CASCADE,
    category      VARCHAR(50),  -- 'grammar_weakness', 'preferred_topics', 'vocab_gap'
    content       TEXT NOT NULL, -- Ví dụ: "User frequently confuses Present Perfect with Past Simple"
    occurrence_count INTEGER DEFAULT 1,
    last_observed TIMESTAMPTZ DEFAULT now()
);

-- 2. Kho từ vựng cần ôn tập (Spaced Repetition Vocabulary)
CREATE TABLE user_vocabulary (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID REFERENCES users(id) ON DELETE CASCADE,
    word          VARCHAR(100) NOT NULL,
    definition    TEXT,
    example       TEXT,
    mastery_level INTEGER DEFAULT 1, -- 1 (Mới) -> 5 (Thành thạo)
    next_review_at TIMESTAMPTZ DEFAULT now(),
    created_at    TIMESTAMPTZ DEFAULT now()
);

-- 3. Bài tập trắc nghiệm được Agent tạo tự động (Generated Quizzes)
CREATE TABLE generated_quizzes (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID REFERENCES users(id) ON DELETE CASCADE,
    title         VARCHAR(255) NOT NULL,
    questions     JSONB NOT NULL, -- List các câu hỏi, đáp án A/B/C/D, giải thích
    status        VARCHAR(20) DEFAULT 'pending', -- pending | completed
    score         INTEGER DEFAULT 0,
    created_at    TIMESTAMPTZ DEFAULT now()
);
```

---

## 4. ReAct Loop & Agent Flowchart

```
[User Message] 
      │
      ▼
[Orchestrator Agent]
      │
      ├───▶ [Load User Memory & Level from DB]
      │
      ├───▶ [Decision Engine (ReAct / Function Calling)]
      │         │
      │         ├─ Need Dictionary?  ──▶ Run `dictionary_tool`
      │         ├─ Need Grammar Fix? ──▶ Run `grammar_analyzer_tool`
      │         ├─ Need Course Doc?  ──▶ Run `rag_retrieval_tool`
      │         └─ Need Quiz Gen?    ──▶ Run `quiz_generator_tool`
      │
      ▼
[Response Builder & Evaluation Agent]
      │
      ├─ Synthesize LLM Answer + Tool Output
      ├─ Update User Weakness in `user_memories`
      └─ Stream Response back to User
```

---

## 5. Danh sách Rủi ro & Edge Cases

| Rủi ro | Mức độ | Phương án xử lý |
| :--- | :--- | :--- |
| **Agent bị lặp vô tận (Tool Loop)** | 🔴 High | Giới hạn tối đa 3 lượt gọi Tool (Max Iterations = 3) cho 1 request. |
| **Memory bị phình to (Too many memories)** | 🟡 Medium | Gộp các memory tương tự nhau (Memory Consolidation) và xoá ký ức cũ ít lặp lại. |
| **LLM latency tăng khi dùng Tool** | 🟡 Medium | Sử dụng Stream SSE ngay khi Agent ra quyết định; Chạy ghi Memory bất đồng bộ (Background Task). |
| **Tạo Quiz không đúng trình độ** | 🟡 Medium | Truyền tham số `user_level` (A1-C2) trực tiếp vào prompt tạo Quiz. |

---

## 6. Lộ trình thực hiện từng Bước (Phase)

- **Phase 1 (Agent Core & Toolsets)**:
  - Dựng Agent Tool Execution Engine (Tra từ điển, Sửa lỗi ngữ pháp chi tiết).
  - Tích hợp Topic-based Speaking Partner Mode.

- **Phase 2 (Long-term Memory & SRS)**:
  - Xây dựng hệ thống lưu ký ức dài hạn `user_memories` & theo dõi từ vựng `user_vocabulary`.
  - Agent tự rút ra điểm yếu sau mỗi cuộc hội thoại.

- **Phase 3 (Auto Quiz & Personalized Dashboard)**:
  - Tự động sinh Quiz từ điểm yếu.
  - API báo cáo tiến bộ học tập cá nhân hóa.
