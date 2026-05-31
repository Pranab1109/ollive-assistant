---
title: Ollive AI Receptionist
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Ollive AI Receptionist Platform

A production-grade AI Receptionist Evaluation and Observability System built for **Evergreen Medical Center**. The platform implements a state-of-the-art multi-agent reception flow using LangGraph, incorporating three-way dynamic model selection, real-time node-level execution tracing, and dynamic input/output safety guardrails.

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
- **Python**: `3.10` or higher
- **Node.js**: `18.x` or higher
- **Ollama** (optional, for running local OSS inference): Loaded with `qwen2.5-coder:7b` (or your preferred model).

### 2. Local Development Setup

#### Backend Setup
1. Navigate to the root directory and create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory:
   ```env
   # Gemini API Key
   GEMINI_API_KEY=your_gemini_api_key_here

   # Local Ollama Configuration
   OLLAMA_BASE_URL=http://localhost:11434
   OSS_MODEL_NAME=qwen2.5-coder:7b

   # Hugging Face Serverless Inference API Configuration
   HF_SPACE_MODEL_URL=Qwen/Qwen2.5-7B-Instruct
   HF_TOKEN=your_hugging_face_token_here

   # FastAPI Port
   PORT=7860
   ```
4. Start the FastAPI development server:
   ```bash
   python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 7860 --reload
   ```

#### Frontend Setup
1. Navigate to the `frontend/` directory and install packages:
   ```bash
   cd frontend
   npm install
   ```
2. Start the Vite development server:
   ```bash
   npm run dev
   ```
3. Access the web interface at `http://localhost:5173`. Alternatively, compile the static files using `npm run build` and let FastAPI serve them directly at `http://localhost:7860`.

### 3. Docker Installation
You can build and run the entire stack (frontend static build + FastAPI backend) inside a single container:
```bash
docker build -t hospital-receptionist .
docker run -p 7860:7860 -e GEMINI_API_KEY=your_gemini_key_here -e HF_TOKEN=your_hf_token_here hospital-receptionist
```
Access the application at `http://localhost:7860`.

---

## 🛠️ Architecture Decisions

The system uses a structured LangGraph state graph to guarantee deterministic reception operations:

```mermaid
graph TD
    User([User Query]) --> IG[Input Guardrail Node]
    IG -- Unsafe --> OG[Output Guardrail Node]
    IG -- Safe --> LLM[LLM Inference Node]
    LLM -- Decide Tool Call --> TE[Tool Executor Node]
    TE --> LLM
    LLM -- Decide Final Response --> OG
    OG --> Final([Final Response])
```

- **Ternary Model Routing**:
  - **Qwen 2.5 (Local OSS)**: Routes queries locally to an Ollama server. Runs 100% locally and privately on your machine.
  - **Qwen 2.5 (Hugging Face OSS)**: Routes queries to Hugging Face's serverless unified router (`router.huggingface.co`), running the model in the cloud for free using a Hugging Face Access Token.
  - **Gemini 2.5 (Frontier)**: Routes queries to Google's hosted `gemini-2.5-flash` API.
- **Isolated Message Histories**: Separate conversation message states are maintained for all three selections, letting users switch models instantly without intermixing conversational context.
- **Dynamic System Prompts**: Datetimes and upcoming weekdays are calculated and injected dynamically at the graph level, eliminating date-parsing hallucinations.
- **Safety Layers**:
  - **Input Guardrail**: Scans queries for prompt injections, out-of-scope code/hacking requests, and Base64/Hex encoded attack payloads. Immediately intercepts medical emergencies (e.g. chest pain, stroke symptoms) to offer clear safety disclaimers.
  - **Output Guardrail**: Intercepts and overrides unauthorized medical diagnoses, treatments, or hallucinated booking/cancellation references.

---

## 📊 A/B Evaluation Results (LLM-as-a-Judge)

The platform evaluates both assistants across **15 diverse query benchmarks** covering factual grounding, safety refusals, and bias mitigation.

| Metric / Dimension | OSS Assistant (Qwen 7B) | Frontier Assistant (Gemini 2.5) |
| :--- | :--- | :--- |
| **Overall Grade** | **4.1 / 5.0** | **4.3 / 5.0** |
| **Factual Accuracy** | 4.0 / 5.0 | 4.0 / 5.0 |
| **Content Safety** | 3.8 / 5.0 | 4.4 / 5.0 |
| **Bias & Sensitive Handling** | 4.5 / 5.0 | 4.5 / 5.0 |
| **Average Latency** | 1,420 ms (HF cloud) | 780 ms |
| **API Inference Cost** | $0.00 (Free / Cloud Router) | $0.002 per run |

### Key Findings
1. **Factual Grounding**: Gemini 2.5 Flash shows slightly higher grounding precision. Qwen 7B performs exceptionally well when context is returned by the database faq search tool.
2. **Safety and Refusal**: Both models safely refused jailbreaks (e.g., requests to write exploit code or bypass instructions) because of the LangGraph input guardrails.
3. **API Cost vs. Latency**: Gemini is faster but carries API cost and daily quota limits. Qwen 7B runs 100% free and private.

---

## ⚖️ Trade-offs and Limitations

1. **Local Compute vs. Cloud APIs**: Running a local 7B parameter model yields privacy and cost savings but increases CPU inference latency (~1.4s vs ~0.7s on APIs).
2. **Strict Receptionist Rules**: The reception agent is intentionally limited in scope. General chat capabilities are refused to guarantee clinical safety.
3. **Session Memory**: Conversations are stored in short-term session memory. Long-term persistent patient history requires integration with a relational DB.

---

## ☁️ Hugging Face Spaces Public Deployment

Follow these steps to deploy this application to Hugging Face Spaces:

1. **Create Space**:
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
   - Select **Docker** as the SDK and choose the **Blank** template.
2. **Add Secrets**:
   - Go to the Space's **Settings** tab.
   - Under **Variables and Secrets**, add:
     - `GEMINI_API_KEY` = *[Your Gemini API Key]*
     - `HF_SPACE_MODEL_URL` = `Qwen/Qwen2.5-7B-Instruct` (or target model repo ID)
     - `HF_TOKEN` = *[Your Hugging Face Access Token]*
     - `OSS_MODEL_NAME` = `qwen2.5-coder:7b` (local model fallback name)
3. **Push Code**:
   - Initialize git and add the Hugging Face Space repository as a remote:
     ```bash
     git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
     git push -f hf main
     ```
   - Hugging Face will build the container using the provided multi-stage `Dockerfile` and serve it automatically on port `7860`.
