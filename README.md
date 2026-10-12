# Voyage AI Semantic Search Demo

A small Python package that embeds sample documents with Voyage AI, selects the highest dot-product matches, reranks that shortlist with Voyage AI, and uses Groq to answer a question from the top reranked result.

## Requirements

- Python 3.10 or newer
- A Voyage AI API key
- A Groq API key

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
Copy-Item .env.example .env
```

Alternatively, install the runtime dependencies from `requirements.txt` with `python -m pip install -r requirements.txt`.

Set `VOYAGE_API_KEY` and `GROQ_API_KEY` in `.env`. `GROQ_MODEL` is optional and defaults to `openai/gpt-oss-120b`. `EMBEDDING_CANDIDATE_COUNT` controls how many embedding matches are sent to the reranker; it defaults to `2` and must be a positive integer.

Documents with blank-line paragraph breaks are chunked by paragraph. If a document has no paragraph breaks, it is split at sentence boundaries. Token counts are used only to batch embedding requests; they do not determine chunk boundaries. Voyage AI may truncate an individual chunk that exceeds its model context limit.

## Run

```powershell
python -m voyageai_project --demo
# Or, after editable installation:
voyage-demo --demo
```

Run either CLI command with `--help` to see usage.

## Development

Install the development tools and run Ruff:

```powershell
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
```

Do not commit `.env` or API credentials. The `.env.example` file documents the supported settings.
