# Lebanon Outing Planner: an MCP toolkit with FastMCP and the OpenAI Agents SDK

**EECE 503P/798S: Agentic Systems** · C5 – Agent Tools · Adam Hijazi

A weather-aware outing planner for Lebanon, built as an **MCP server with exactly three tools** (FastMCP) and used by an **agent built with the OpenAI Agents SDK**.

🎥 **Demo video:** https://youtu.be/zjD50nKjQZE

| Category | Tool | What it does | Changes state? |
|---|---|---|---|
| **Data** | `get_weather_forecast(city, days=3)` | Daily forecast for a city (condition, min/max °C, rain chance, wind) from the free Open-Meteo API | No |
| **Action** | `save_outing(city, date, activity, note="")` | Saves a planned outing to `data/outings.json` | **Yes** |
| **Orchestration** | `plan_outing(city, activity, days=5)` | Gets the forecast (Data step), picks the best day for the activity, then saves it (Action step) | Yes, through the Action step |

**Resources** (read-only, `@mcp.resource`): `outings://saved` (all saved outings) and `outings://activities` (the weather rules for hiking, beach, picnic, cycling and skiing).

## Architecture

```text
 You ──► agent.py  (MCP HOST: OpenAI Agents SDK Agent + Gemini LLM)
            │  MCPServerStdio = MCP CLIENT (JSON-RPC over stdio)
            ▼
         server.py (MCP SERVER: FastMCP, 3 tools + 2 resources) ──HTTP──► Open-Meteo API
            │
            └──► data/outings.json
```

The agent prints every step of the round trip:

```text
🧑 Find me the best day this week to go hiking in Faraya and save it.
   🔧 agent → MCP client → server: plan_outing({"city": "Faraya", "activity": "hiking"})
   📦 server → MCP client → agent: {"status": "saved", "best_day": {"date": "...", ...}}
🤖 The best day is ... I saved it for you!
```

## Files

| File | Purpose |
|---|---|
| `server.py` | FastMCP server: the 3 tools, 2 resources, typed (Pydantic) outputs, and clear errors |
| `agent.py` | Agents SDK host/client: interactive chat, or `--demo` for a scripted demo |
| `test_tools.py` | 15 tests: each tool, each failure case, the MCP protocol (in-memory), and the Agents SDK stdio client |
| `test_agent_roundtrip.py` | 2 end-to-end tests: Agent → MCP client → server → back, with a scripted LLM (no API key needed) |
| `run_in_colab.ipynb` | One-click way to run everything in Google Colab |
| `SUMMARY.md` | Part 1: MCP article and the Fetch MCP server critique |
| `REFLECTION.md` | Reflection: tool boundaries, why three tools, next steps |

## Setup

### Option A: Google Colab (easiest, nothing to install)

1. Open `run_in_colab.ipynb` in Colab (**File → Upload notebook**, or open it from GitHub).
2. Add your Gemini key: **🔑 Secrets** → `GEMINI_API_KEY` → turn on **Notebook access**. Get a free key at <https://aistudio.google.com/apikey>.
3. **Runtime → Run all.** It clones this repo, installs the requirements, runs the tests, runs the demo, and opens a chat cell.

### Option B: your own computer (Python 3.10+)

```bash
git clone https://github.com/adammhijazi/c5_mcp_toolkit.git
cd c5_mcp_toolkit
pip install -r requirements.txt
cp .env.example .env            # then paste your key after GEMINI_API_KEY=
python -m pytest -v             # 17 tests, no API key or internet needed
python agent.py --demo          # scripted demo of all three tools + a failure case
python agent.py                 # interactive chat
```

If the weather API can't be reached (firewall, rate limit), add `--offline` to use the server's built-in sample weather.

### Why Gemini with the OpenAI Agents SDK?

The Agents SDK works with any OpenAI-compatible endpoint. `agent.py` points an `AsyncOpenAI` client at Gemini's OpenAI-compatible API (`https://generativelanguage.googleapis.com/v1beta/openai/`) and wraps it in `OpenAIChatCompletionsModel`, so a free Gemini key is enough. To use OpenAI instead, set `PROVIDER=openai` and `OPENAI_API_KEY` in `.env`. If one Gemini model is busy or rate-limited, the agent automatically tries the next one.

## Example queries

| Query | Expected tool |
|---|---|
| "What's the weather in Byblos for the next 3 days?" | `get_weather_forecast` |
| "Save a picnic in Beirut on 2026-12-05, note: bring the frisbee." | `save_outing` |
| "Find the best day this week to hike in Faraya and save it." | `plan_outing` |
| "What's the weather in Xyzzyville?" | `get_weather_forecast` → clear "city not found" error |
| "Save a beach day in Tyre on 2020-01-01." | `save_outing` → clear "date is in the past" error |

## Failure handling (predictable, never silent)

| Tool | Failure | Message |
|---|---|---|
| Data | Unknown city | "City 'X' was not found. Check the spelling or try a nearby larger town." |
| Data | `days` outside 1–7 | "days must be between 1 and 7 (got 12)." |
| Data | API down / timeout | "Weather service is unavailable right now (ConnectError). Please try again later." |
| Action | Bad date format / past date | "Invalid date '05/12/2026'. Use the format YYYY-MM-DD…" / "…is in the past." |
| Action | Duplicate outing | "An outing for picnic in Beirut on … is already saved." |
| Orchestration | Unknown activity | "Unknown activity 'skydiving'. Choose one of: hiking, beach, picnic, cycling, skiing." |
| Orchestration | No good day | Returns `status: "no_suitable_day"` with the reason, and **saves nothing** |

Errors are raised as FastMCP `ToolError`s, so the MCP client receives `isError: true` with the message, and the agent explains it to the user.
