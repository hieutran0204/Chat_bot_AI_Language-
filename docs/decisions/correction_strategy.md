# ADR 001: Correction Strategy for AI Speaking Tutor

* **Status**: Accepted
* **Date**: 2026-09-12
* **Deciders**: Language AI Core Architecture Team

---

## 1. Context & Problem Statement
In an AI-powered conversational speaking tutor, feedback can either be:
* **Option A (Real-time Inline)**: The bot immediately interrupts or begins its response by correcting the learner's grammatical mistakes before answering.
* **Option B (Silent Tracking & Decoupled Feedback)**: The bot responds smoothly and naturally like a real human conversational partner in spoken dialogue, while errors are captured independently in structured form (`AIMessage.corrections`) and persisted to the learner's profile.

Learners practicing spoken English often suffer from speaking anxiety. Constant immediate interruptions break conversation momentum and discourage fluent speech.

---

## 2. Decision
We adopt **Option B (Silent Tracking & Decoupled Feedback)** as the primary default strategy.

1. **Natural Dialogue Flow**:
   - The primary LLM streaming turn responds to the learner's thoughts warmly, enthusiastically, and naturally in 1–3 spoken sentences.
2. **Decoupled Error Extraction**:
   - Linguistic slips (grammar, word choice, pronunciation phonetics) are extracted into the `corrections` list in `AIMessage`.
   - The UI displays these corrections in an unobtrusive secondary drawer or bubble so the learner can review them at their own pace without derailing the spoken conversation.
3. **Configurable Flexibility**:
   - The prompt template exposes `{correction_style}` so users can toggle to "Direct Correction Mode" (Option A) if they are specifically doing exam cramming (e.g. IELTS mock interview).
