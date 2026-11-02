# FAQ Chatbot Demo

A small **document-grounded FAQ assistant** you can show to clients: it reads your business documents and answers customer questions from that content only. When it cannot find a relevant answer, it says **I don't know** instead of making things up.

Built as a portfolio piece by **Tushar Gupta** — Python, Streamlit, and a simple retrieval-augmented generation (RAG) pipeline with no paid embedding API required.

## What it does

- Loads FAQ and policy documents from the `data/` folder (Markdown, plain text, or PDF).
- Splits them into passages and finds the best matches for each question (TF-IDF, runs locally).
- Sends those passages to an OpenAI-compatible chat API with strict instructions to stay on-document.
- Shows **citations** (which file and section were used).
- Refuses off-topic or unsupported questions when similarity scores are too low.

The included **BeanVoyage Coffee** content is **fictional** and labelled as such — swap in your own files to demo your client's brand.

## Quick start (no API key)

For demos and tests, use mock mode — the app quotes retrieved text without calling an external API:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
```

In `.env`, set `MOCK_LLM=1`, then run:

```bash
streamlit run app.py
```

On Windows (Command Prompt), you can set mock mode for one session without editing `.env`:

```bat
set MOCK_LLM=1
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

## Live answers with OpenRouter

### Get an OpenRouter key (beginner-friendly)

1. Sign up at [openrouter.ai](https://openrouter.ai/).
2. Open **Keys** in the dashboard and create an API key.
3. Add a small amount of credit to your account, or pick a model tagged **:free** on OpenRouter if you prefer not to pay for testing.
4. Copy `.env.example` to `.env` and paste your key into `OPENAI_API_KEY`. That variable name is shared with OpenAI-compatible tools — here it holds your **OpenRouter** key, not an OpenAI Platform key.

Example `.env` for live chat:

```env
OPENAI_API_KEY=sk-or-v1-...
OPENAI_BASE_URL=https://openrouter.ai/api/v1
MODEL=openai/gpt-4o-mini
MOCK_LLM=0
```

Run:

```bash
streamlit run app.py
```

You can point `OPENAI_BASE_URL` at OpenAI or any compatible provider and set `MODEL` to whatever they support.

## Add your own documents

1. Put `.md`, `.txt`, or `.pdf` files under `data/` (subfolders are fine).
2. Restart the Streamlit app so the index rebuilds.
3. Ask questions in the chat — answers should cite the files you added.

Keep one topic per section (Markdown headings help). Remove or replace the fictional BeanVoyage samples when presenting a real client.

## Run tests

```bash
MOCK_LLM=1 pytest
```

## Project layout

| Path | Purpose |
|------|---------|
| `app.py` | Streamlit chat UI |
| `rag/` | Document loading, chunking, retrieval, LLM client |
| `data/` | Your knowledge base (sample fiction included) |
| `tests/` | Retrieval, refusal, and citation tests |

## License

MIT — see [LICENSE](LICENSE).
