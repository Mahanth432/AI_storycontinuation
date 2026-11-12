# AI Story Continuation System

An interactive NLP web application that generates creative story continuations from a user-provided opening prompt using **DistilGPT2**, **PyTorch**, and **Hugging Face Transformers**, featuring a **Streamlit** frontend and a containerized **TorchServe** REST inference service with automatic local fallback.

---

## Overview

The **AI Story Continuation System** enables writers, students, and enthusiasts to enter the opening lines of a story, configure response length and creativity (sampling temperature), and generate a coherent continuation in real time. The system decouples the user interface from model serving by routing inference requests to a **TorchServe** REST API (deployable via Docker or Linux/WSL2) while maintaining an in-process **Local DistilGPT2** inference path (`model/model_loader.py`) as an automatic fallback or standalone mode.

---

## Problem Statement

Creative writers and learners often encounter writer's block when developing narrative openings, while deploying transformer-based language models directly inside monolithic UI scripts tightly couples presentation logic with heavy model loading and makes scaling or containerizing inference difficult. Furthermore, exposing raw causal language models without parameter bounds or context-window safeguards can produce repetitive text, runtime context-length errors, or poor user experience.

---

## Objectives

- Build an intuitive, responsive web interface for interactive story generation using **Streamlit**.
- Separate model loading, tokenization, validation, and decoding logic into a modular PyTorch/Transformers package (`model/`).
- Serve the pretrained **DistilGPT2** model through a dedicated **TorchServe** custom handler (`torchserve/handler.py`) and model archive (`.mar`) for production-style REST inference.
- Containerize the TorchServe inference server with **Docker** for reproducible CPU deployment.
- Support controlled text generation (`Short`, `Medium`, `Long` token presets, temperature bounds, nucleus sampling, and repetition penalties) alongside comprehensive offline unit tests.

---

## Technologies Used

| Category | Technology / Version | Purpose |
| --- | --- | --- |
| **Language** | Python 3.10 | Core runtime for application, model logic, and tests |
| **Deep Learning** | PyTorch (`2.2.2`) | Tensor computation and CPU inference (`torch.inference_mode`) |
| **NLP / LLM** | Hugging Face Transformers (`4.44.2`), DistilGPT2 | Pretrained causal language model and BPE tokenizer |
| **Frontend UI** | Streamlit (`1.39.0`) | Interactive web interface and session state management |
| **Model Serving** | TorchServe (`0.11.0`), `torch-model-archiver` | REST inference server, model archive packaging, metrics & management APIs |
| **Containerization** | Docker (`python:3.10-slim-bookworm` + OpenJDK 17) | Isolated Linux container runtime for TorchServe |
| **HTTP Client & Numerics** | `requests`, NumPy (`>=1.26.0,<2.0`) | Frontend-to-TorchServe API communication and array compatibility |
| **Testing** | Python `unittest` & `unittest.mock` | Offline unit testing of model loader, handler, and UI helpers |

---

## System Architecture

```text
+-------------------------------------------------------------------+
|                   Streamlit Frontend (Port 8501)                  |
|                      app/streamlit_app.py                         |
|  - Prompt & parameter validation                                  |
|  - Backend selector:                                              |
|      1. TorchServe API (Auto Fallback) [Default]                  |
|      2. TorchServe API (Strict REST mode)                         |
|      3. Local DistilGPT2 (In-process)                             |
+-------------------------------+-----------------------------------+
                                |
           HTTP POST JSON       |        Fallback / Local Mode
   {"prompt","length","temperature"}    (model/model_loader.py)
                                |                 |
                                v                 v
+---------------------------------------+   +-----------------------+
|   TorchServe Server (Linux / Docker)  |   | Local CPU Inference   |
|   - Inference  : localhost:8090       |   | - @lru_cache model    |
|   - Management : localhost:8091       |   | - AutoTokenizer /     |
|   - Metrics    : localhost:8092       |   |   AutoModelForCausalLM|
+-------------------+-------------------+   +-----------------------+
                    |
                    v
+---------------------------------------+
|      story_continuation.mar           |
|  - torchserve/handler.py              |
|    (StoryContinuationHandler)         |
|  - Packaged DistilGPT2 weights &      |
|    tokenizer files                    |
+---------------------------------------+
```

### Request Lifecycle (`StoryContinuationHandler`)

1. **`initialize(context)`**: Loads the DistilGPT2 tokenizer and `AutoModelForCausalLM` once at worker startup from `context.system_properties["model_dir"]` (falling back to `"distilgpt2"` if needed), assigns `pad_token = eos_token`, and sets the model to evaluation mode on CPU.
2. **`preprocess(data)`**: Parses UTF-8 JSON payloads, validates `prompt` (non-empty string), `length` (`Short`, `Medium`, `Long`), and `temperature` (`[0.1, 1.5]`), and truncates input tokens to `1024 - max_new_tokens` so prompts never exceed DistilGPT2's 1024-token context limit. Invalid inputs raise `PredictionException(..., 400)`.
3. **`inference(model_input)`**: Runs `model.generate()` under `torch.inference_mode()` with `do_sample=True`, `top_p=0.92`, `repetition_penalty=1.15`, `no_repeat_ngram_size=3`, and `min_new_tokens=min(15, max_new_tokens)`, slicing and decoding only the newly generated tokens.
4. **`postprocess(inference_output)`**: Returns structured JSON containing `generated_text`, `length`, and `temperature`.

*(Ports `8090`, `8091`, and `8092` are configured in `torchserve/config.properties` to avoid conflicts with services commonly bound to port `8080`.)*

---

## Features

- **Three Story Length Presets**:
  - `Short`: up to `50` new tokens
  - `Medium`: up to `100` new tokens (default)
  - `Long`: up to `180` new tokens
- **Creativity Control**: Temperature slider from `0.1` (focused) to `1.5` (creative), defaulting to `0.8`.
- **Repetition Safeguards**: Nucleus sampling (`top_p=0.92`), `repetition_penalty=1.15`, and `no_repeat_ngram_size=3`.
- **Flexible Inference Backends**: Switch dynamically in the **Inference Backend Settings** expander between `TorchServe API (Auto Fallback)`, `TorchServe API`, and `Local DistilGPT2`, or customize the TorchServe endpoint URL.
- **Natural Spacing & Highlighted Story Card**: Visually distinguishes the user's opening prompt from the AI-generated continuation with smart punctuation spacing.
- **One-Click Copy & Reset**: Client-side **Copy Story** button and **Reset** / **Reset Story** controls for clearing state.
- **Input Validation & AI Disclaimer**: Graceful handling of empty prompts, unreachable APIs, and invalid parameters, plus a clear fictional AI-content disclaimer.

---

## Project Folder Structure

```text
AI-Story-Continuation/
├── app/
│   ├── __init__.py
│   └── streamlit_app.py           # Streamlit UI, backend routing, copy/reset controls
├── model/
│   ├── __init__.py
│   └── model_loader.py            # Local DistilGPT2 loading, validation, and generation
├── torchserve/
│   ├── __init__.py
│   ├── config.properties          # TorchServe ports (8090/8091/8092) and worker settings
│   ├── handler.py                 # Custom StoryContinuationHandler for TorchServe
│   └── package_model.py           # Script to download DistilGPT2 & build .mar archive
├── tests/
│   ├── __init__.py
│   ├── test_model_loader.py       # Unit tests for model_loader.py
│   ├── test_torchserve_handler.py # Unit tests for StoryContinuationHandler
│   └── test_ui_helpers.py         # Unit tests for Streamlit state & backend routing
├── .dockerignore
├── .gitignore
├── Dockerfile                     # Python 3.10 + OpenJDK 17 + TorchServe 0.11.0 image
├── requirements.txt               # Python dependencies for Streamlit & local inference
└── README.md
```

> **Note:** Generated model archives (`model_store/*.mar`), Hugging Face caches, `venv/`, and `__pycache__/` are intentionally excluded from version control via `.gitignore`. Follow the packaging steps below to generate `model_store/story_continuation.mar` locally before running TorchServe or building the Docker image.

---

## Installation Instructions

### Prerequisites

- **Python 3.10**
- **Git**
- *(Optional for TorchServe local/Docker serving)* **Docker Desktop** or a **Linux / WSL2 Ubuntu** environment with **OpenJDK 17**, `torchserve==0.11.0`, and `torch-model-archiver==0.11.0`

### 1. Clone the Repository

```bash
git clone https://github.com/Mahanth432/AI_storycontinuation.git
cd AI_storycontinuation
```

### 2. Create & Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

**Linux / macOS / WSL2:**
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

*(Note: On first run of local inference or model packaging, Hugging Face Transformers downloads the `distilgpt2` model weights automatically and stores them in your local Hugging Face cache.)*

---

## Running the Streamlit Application

Start the Streamlit frontend from the project root:

```bash
streamlit run app/streamlit_app.py
```

By default, the app opens at `http://localhost:8501` and uses `TorchServe API (Auto Fallback)`, which queries `http://localhost:8090/predictions/story_continuation` and automatically falls back to local DistilGPT2 if TorchServe is not running.

### Optional Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `TORCHSERVE_URL` | `http://localhost:8090/predictions/story_continuation` | TorchServe prediction endpoint URL |
| `TORCHSERVE_TIMEOUT` | `60` | HTTP request timeout in seconds |
| `INFERENCE_BACKEND` | `TorchServe API (Auto Fallback)` | Initial backend (`torchserve`, `torchserve_only`/`api`, or `local`) |

---

## Generating & Running the TorchServe Model Archive

TorchServe requires a Linux runtime (native Linux, WSL2, or Docker).

### 1. Package the `.mar` Archive

Install `torchserve==0.11.0` and `torch-model-archiver==0.11.0` (along with the packages in `requirements.txt` and OpenJDK 17) in your Linux/WSL2 environment, then run:

```bash
python torchserve/package_model.py
```

This script downloads `distilgpt2`, stages its tokenizer and weights (`model.safetensors`), packages them alongside `torchserve/handler.py` into `model_store/story_continuation.mar`, and cleans up the temporary staging folder.

### 2. Start TorchServe (Linux / WSL2)

```bash
torchserve --start --ncs \
  --model-store ./model_store \
  --models story_continuation=story_continuation.mar \
  --ts-config ./torchserve/config.properties \
  --foreground
```

---

## Docker Build & Execution Instructions

Ensure `model_store/story_continuation.mar` has been generated first (via `python torchserve/package_model.py`), because `Dockerfile` copies `model_store/story_continuation.mar` into `/app/model_store/story_continuation.mar`.

### 1. Build the Docker Image

```bash
docker build -t ai-story-torchserve .
```

### 2. Run the TorchServe Container

```bash
docker run --rm -p 8090:8090 -p 8091:8091 -p 8092:8092 ai-story-torchserve
```

The container exposes:
- **Inference API:** `http://localhost:8090`
- **Management API:** `http://localhost:8091`
- **Metrics API:** `http://localhost:8092`

Once the container passes its `/ping` healthcheck, launch the Streamlit frontend (`streamlit run app/streamlit_app.py`) to connect to it on port `8090`.

---

## API Endpoints

### Service Endpoints

| API | Method & Path | Port | Description |
| --- | --- | --- | --- |
| **Health Check** | `GET /ping` | `8090` | Returns TorchServe worker health status |
| **Predict** | `POST /predictions/story_continuation` | `8090` | Generates a story continuation from JSON input |
| **List Models** | `GET /models` | `8091` | Lists loaded models and versions |
| **Metrics** | `GET /metrics` | `8092` | Exposes Prometheus-formatted TorchServe metrics |

### Prediction Request (`POST /predictions/story_continuation`)

```json
{
  "prompt": "One day, a student discovered a mysterious box in the college library...",
  "length": "Medium",
  "temperature": 0.8
}
```

- `prompt` *(string, required)*: Non-empty story opening.
- `length` *(string, optional, default `"Medium"`)*: `"Short"` (`50` tokens), `"Medium"` (`100` tokens), or `"Long"` (`180` tokens).
- `temperature` *(float, optional, default `0.8`)*: Sampling temperature in `[0.1, 1.5]`.

### Success Response (`200 OK`)

```json
{
  "generated_text": "and found a tarnished brass compass resting atop a folded parchment map.",
  "length": "Medium",
  "temperature": 0.8
}
```

### Validation Error Response (`400 Bad Request`)

```json
{
  "code": 400,
  "type": "InternalServerException",
  "message": "Field 'prompt' must be a non-empty string."
}
```

### Example `curl` Verification

```bash
curl http://localhost:8090/ping
curl http://localhost:8091/models
curl -X POST http://localhost:8090/predictions/story_continuation \
  -H "Content-Type: application/json" \
  -d '{"prompt":"One day, a student discovered a mysterious box in the college library...","length":"Medium","temperature":0.8}'
```

---

## Testing Instructions

All unit tests run offline using mocked Hugging Face and HTTP responses so no model download or running TorchServe instance is required.

```bash
python -m unittest discover -s tests -v
```

### Test Coverage Summary

- `tests/test_model_loader.py`: Prompt validation, generation parameter bounds, model caching, tokenization, CPU decoding, and punctuation joining.
- `tests/test_torchserve_handler.py`: Model directory resolution, `Short`/`Medium`/`Long` token limits, long-prompt truncation, 400 validation errors, and `handle()` entrypoint.
- `tests/test_ui_helpers.py`: Syntax verification, Streamlit session state initialization/reset, TorchServe API success/error handling, and automatic local fallback routing.

---

## Screenshots

*(Add screenshots of your running application here, e.g., by placing images in a `docs/screenshots/` folder and linking them below.)*

- **Main Interface & Story Prompt Controls** — *Streamlit web UI with Response Length selector, Creativity slider, and Inference Backend Settings.*
- **Generated Story Output** — *Continued narrative card showing metadata pills (`Medium`, `Creativity`, backend badge), Copy Story, and Reset Story buttons.*
- **TorchServe Health & Prediction Verification** — *Terminal output from `/ping`, `/models`, and `/predictions/story_continuation`.*

---

## Limitations and Future Enhancements

### Current Limitations

- **Pretrained Base Model**: Uses general-purpose `distilgpt2` without domain-specific fine-tuning on curated fiction datasets, so long-range plot coherence can vary.
- **CPU Inference Latency**: Configured for CPU execution (`minWorkers=1`, `maxWorkers=1`), which keeps hardware requirements low but makes longer generations slower than GPU inference.
- **Context Window Limit**: Bounded by DistilGPT2's maximum context length of `1024` tokens (prompt tokens + `max_new_tokens`).
- **Single-Turn Continuation**: Generates one continuation per prompt rather than multi-turn interactive chapter state management.
- **Cloud Deployment Status**: Local WSL2 and Docker containerization are implemented; **AWS SageMaker deployment has not been performed** and remains a future enhancement.

### Future Enhancements

- **AWS SageMaker Deployment**: Deploy the TorchServe container image and `story_continuation.mar` to an AWS SageMaker real-time inference endpoint (not yet implemented).
- **Fine-Tuned Story Models & Genre Conditioning**: Fine-tune DistilGPT2 or larger causal LMs on creative writing corpora with genre/tone presets (Fantasy, Mystery, Sci-Fi).
- **GPU & Multi-Worker Scaling**: Enable CUDA support and dynamic TorchServe worker scaling for higher throughput.
- **Streaming & Multi-Turn Continuation**: Stream tokens incrementally to the UI and allow users to iteratively continue or branch from generated paragraphs.
- **Docker Compose Setup**: Add a `docker-compose.yml` orchestrating both the TorchServe container and the Streamlit frontend container together.
