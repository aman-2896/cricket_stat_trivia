# Cricket Stat Trivia

A cricket assistant that answers two kinds of questions:

1. **Stats** — ball-by-ball statistics for the ICC Men's T20 World Cup 2024 (runs, wickets, strike rate, economy, etc.), answered by translating the question into SQL and running it against a Postgres database loaded from [Cricsheet](https://cricsheet.org) data.
2. **Rules** — questions about the Laws of Cricket, answered via hybrid (vector + keyword) retrieval over the official Laws of Cricket document.

It's served three ways: a FastAPI HTTP endpoint, a LangGraph-based agent, and an MCP (Model Context Protocol) server.

## Architecture

```
                     ┌───────────────────────┐
                     │   main.py (FastAPI)   │  POST /chat
                     │  or mcp/mcp_server.py │  (MCP tools)
                     └──────────┬────────────┘
                                │
                     ┌──────────▼────────────┐
                     │ services/agent.py     │  LangGraph agent — decides
                     │ (or raw_sdk_agent.py) │  which tool(s) to call
                     └──────────┬────────────┘
                 ┌──────────────┴──────────────┐
                 ▼                              ▼
    ┌────────────────────────┐     ┌─────────────────────────────┐
    │ services/text2sql.py   │     │ services/ingest.py           │
    │ NL question → SQL →    │     │ hybrid retrieval over the    │
    │ safety-checked SELECT  │     │ Laws of Cricket (pgvector +  │
    │ on a read-only DB user │     │ BM25 via ParadeDB)            │
    └───────────┬────────────┘     └───────────────┬───────────────┘
                │                                   │
                ▼                                   ▼
        ┌───────────────────────────────────────────────────┐
        │                  Postgres (db/)                    │
        │  matches / players / innings / deliveries /        │
        │  wickets / reviews   +   langchain_pg_embedding     │
        └───────────────────┬─────────────────────────────────┘
                            │ loaded by
                            ▼
                    db/loader.py (Cricsheet JSON → tables)
```

## Project layout

```
config/
  config.py        # env-based DB config (full + read-only) and DB_URL
db/
  db.py            # psycopg connection helpers (full-access & read-only)
  create_schema.py # creates the Postgres schema (enums + 6 tables)
  loader.py        # loads Cricsheet match JSON files into that schema
mcp/
  mcp_server.py    # exposes get_cricket_stats / get_cricket_rules as MCP tools
services/
  agent.py         # LangGraph agent (used by main.py) — the primary agent
  raw_sdk_agent.py # equivalent agent built on the raw OpenAI SDK, for comparison
  text2sql.py      # NL → SQL generation, safety checks, execution
  ingest.py        # Laws of Cricket ingestion + hybrid retrieval
  eval.py          # golden-set eval for text2sql (LLM-as-judge)
  agent_eval.py    # tool-routing eval for the agent
main.py            # FastAPI app exposing POST /chat
cricsheets/        # raw Cricsheet match JSON files (input to db/loader.py)
documents/         # Laws of Cricket docx/images + golden eval sets
```

## How a question is answered

- **Stats questions** go through `services/text2sql.py`:
  1. `introspect_schema()` reads the live table/column/FK/enum definitions from Postgres.
  2. `generate_sql()` prompts an LLM with that schema plus hand-written domain hints (`SCHEMA_HINTS` — e.g. use `batter_runs` for a batter's own score but `total_runs` for a bowler's runs conceded, exclude super overs, exclude run-outs from bowler wickets, match player names by surname) to produce a single SQL `SELECT`.
  3. `is_safe_sql()` rejects anything that isn't a single `SELECT` statement (blocks `DROP`/`DELETE`/`INSERT`/etc. and multiple statements) — defense against prompt injection trying to get destructive SQL executed.
  4. `execute_sql()` runs the query using a **read-only** database role (`DB_READONLY_CONFIG`), so even a query that slipped past the safety check can't mutate data.

- **Rules questions** go through `services/ingest.py`:
  1. The Laws of Cricket `.docx` is cleaned and chunked, embedded, and stored in a `pgvector`-backed table, along with LLM-generated text descriptions of the diagrams/images in the document.
  2. At query time, `get_hybrid_retrieval_results()` combines dense vector similarity search with BM25 keyword search (via ParadeDB), merging the two rankings with Reciprocal Rank Fusion, so both paraphrased and exact-term questions are handled well.

- **Routing** between the two tools (and deciding when neither applies) is handled by the LangGraph agent in `services/agent.py`, which is grounded to only answer using its tools and to decline anything else — including declining just the out-of-scope half of a multi-part question.

## Database schema

Six tables, populated by `db/loader.py` from Cricsheet JSON:

| Table        | Purpose                                                             |
|--------------|----------------------------------------------------------------------|
| `matches`    | One row per match: teams, venue, dates, result, toss, etc.          |
| `players`    | One row per player per match.                                       |
| `innings`    | One row per innings: powerplay overs, chase target, super-over flag.|
| `deliveries` | One row per ball bowled — the core fact table for statistics.       |
| `wickets`    | One row per wicket, linked to the delivery that produced it.        |
| `reviews`    | One row per DRS review taken on a delivery.                         |

## Setup

1. Install dependencies (this project uses [uv](https://github.com/astral-sh/uv)):
   ```bash
   uv sync
   ```
2. Copy `.env.example` to `.env` and fill in your Postgres credentials and API keys:
   ```bash
   cp .env.example .env
   ```
   You'll need a Postgres database with a full-access user (`DB_USER`) and a **read-only** user (`DB_READONLY_USER`) that only has `SELECT` privileges — the read-only user is what actually executes LLM-generated SQL.
3. Create the schema:
   ```bash
   python -m db.create_schema
   ```
4. Load match data (drop Cricsheet `*.json` files into `./cricsheets/` first):
   ```bash
   python -m db.loader
   ```
5. Ingest the Laws of Cricket document (populates the vector store used for rules questions):
   ```bash
   python -c "from services.ingest import split_chunks_embed_store, extract_laws_images, get_image_description, ingest_image_content; split_chunks_embed_store(); extract_laws_images(); ingest_image_content(get_image_description())"
   ```

## Running

- **HTTP API**:
  ```bash
  fastapi dev main.py
  ```
  Then `POST /chat` with `{"question": "..."}`.

- **MCP server**:
  ```bash
  python mcp/mcp_server.py
  ```

- **Agent directly** (for local testing):
  ```bash
  python -m services.agent
  ```

## Evaluation

- `python -m services.eval` — runs the golden question set (`documents/golden_set/`) through `text2sql` and grades each answer with an LLM judge.
- `python -m services.agent_eval` — checks that the agent calls the correct tool(s) for a fixed set of routing test cases.

## Environment variables

See `.env.example`:

| Variable            | Purpose                                             |
|---------------------|------------------------------------------------------|
| `DB_HOST`/`DB_PORT`/`DB_NAME` | Postgres connection details                  |
| `DB_USER`/`DB_PASSWORD`       | Full-access DB credentials                   |
| `DB_READONLY_USER`            | Read-only DB role used to run generated SQL  |
| `ANTHROPIC_API_KEY`           | Used by MCP/Claude-based tooling             |
| `OPENAI_API_KEY`              | Used for LLM calls (SQL generation, agent, embeddings, eval judge) |
