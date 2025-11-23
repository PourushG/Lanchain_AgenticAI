**Project Snapshot**:

- **Purpose**: A small LangChain demo/experimentation workspace focused on embeddings, vectorstores, and simple GenAI demos (Streamlit examples + notebooks).
- **Key folders**: `Lanchain_AgenticAI/langchain` (examples & notebooks); `Lanchain_AgenticAI/1-python_basics` (learning notebooks).

**How to run / developer workflows**:

- **Install deps**: `pip install -r Lanchain_AgenticAI/requirements.txt` (project uses Python >= 3.11 per `pyproject.toml`).
- **Run example app**: `streamlit run Lanchain_AgenticAI/langchain/1-langchain/ollama_app.py` — uses `.env` for keys and `langchain_ollama`.
- **Notebooks**: Open the notebooks under `Lanchain_AgenticAI/langchain` with Jupyter/VS Code. They are the primary place for experiments (embeddings, FAISS, Chroma).

**Important environment/config patterns**:

- Secrets live in `Lanchain_AgenticAI/langchain/.env`. Common variables: `OPENAI_API_KEY`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`, `HF_TOKEN`, `EMBEDDING_MODEL`.
- Code loads env via `dotenv.load_dotenv()` and then reads `os.getenv(...)`. Do not hardcode keys into files; prefer `.env` or CI secrets.

**Major components & data flow**:

- **Prompting / LLMs**: Example entrypoint `Lanchain_AgenticAI/langchain/1-langchain/ollama_app.py` builds a `ChatPromptTemplate` → `OllamaLLM` → `StrOutputParser` chain. Agents should treat that file as canonical example of prompt→LLM→parser wiring.
- **Embeddings & Vectorstores**: Demos and notebooks in `Lanchain_AgenticAI/langchain/embeddings` and `.../vectorstores` show usage of `sentence_transformers`, `faiss-cpu`, and `chromadb`. Expect data flow: raw text → splitter → embeddings → vectorstore index (FAISS/Chroma).
- **Notebooks are source-of-truth** for exploratory code. Productionizable parts are small scripts (e.g., `ollama_app.py`) and should be refactored into modules if made reusable.

**Project-specific conventions and patterns**:

- Notebooks for exploration: changes to algorithms are reflected first in notebooks; port stable code to `.py` files when needed.
- Lightweight scripts use `langchain_*` extension packages (e.g., `langchain_ollama`, `langchain_chroma`). Follow the import style in `ollama_app.py` as the canonical example.
- Environment variables are set for LangSmith tracing in `ollama_app.py` (see `os.environ["LANGCHAIN_TRACING_V2"] = "true"`) — keep tracing toggles consistent across demo scripts.

**Integration points & external deps**:

- External APIs: OpenAI (via `openai`), Hugging Face tokens (`HF_TOKEN`), and Ollama (`langchain_ollama`). Ensure API keys are available in environment.
- Vector stores: `faiss-cpu` (local FAISS indices in `faiss_index/`), and `chromadb` (subfolder `chromadb/chroma_db`). Inspect these folders for existing persisted indexes before rebuilding.

**When modifying or adding code**:

- Preserve notebook experiments. When converting notebook logic to modules, add a small runnable example script and update `requirements.txt` accordingly.
- Update `Lanchain_AgenticAI/requirements.txt` and `pyproject.toml` when adding new runtime deps. Prefer pinning only when necessary for reproducibility.
- Avoid committing secrets. If a secret accidentally appears, rotate it and remove from history.

**Hints for AI agents working on this repo**:

- Start from `Lanchain_AgenticAI/langchain/1-langchain/ollama_app.py` for LLM wiring. Use the notebooks for embedding/vectorstore usage examples.
- Look for persistent artifacts in `Lanchain_AgenticAI/langchain/vectorstores/faiss_index` and `Lanchain_AgenticAI/langchain/vectorstores/chromadb/chroma_db` before rebuilding indexes.
- If adding features that call external APIs, add optional environment checks (fail-fast with clear messages) and document required env vars at top of the script.

**Files to reference during edits**:

- `Lanchain_AgenticAI/langchain/1-langchain/ollama_app.py` — canonical prompt/LLM example.
- `Lanchain_AgenticAI/langchain/.env` — expected environment variables.
- `Lanchain_AgenticAI/requirements.txt` and `pyproject.toml` — dependency & Python version constraints.
- `Lanchain_AgenticAI/langchain/embeddings/*` and `Lanchain_AgenticAI/langchain/vectorstores/*` — sample implementations for embeddings and indexing.

If any section is unclear or you'd like more examples (e.g., a small refactor of a notebook cell into a reusable module), tell me which area to expand and I'll iterate.
