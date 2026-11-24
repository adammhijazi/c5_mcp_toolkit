"""
MCP Host / Client: an OpenAI Agents SDK agent that uses the Lebanon Outing Planner MCP server.

    agent.py (Host)  --MCPServerStdio (Client)-->  server.py (MCP Server: 3 tools)  --HTTP-->  Open-Meteo

The LLM is Google Gemini, reached through Gemini's OpenAI-compatible endpoint, so a free Gemini key works
with the OpenAI Agents SDK. (With an OpenAI key, set PROVIDER=openai.)

Usage:
    python agent.py            interactive chat
    python agent.py --demo     scripted demo: Data, Action, Orchestration and a failure case
    add --offline              to use built-in sample weather if the weather API can't be reached
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled
from agents.mcp import MCPServerStdio

load_dotenv(Path(__file__).parent / ".env")
set_tracing_disabled(True)                       # tracing uploads to OpenAI; not needed here


def _make_stdio_safe_for_notebooks() -> None:
    """In Jupyter/Colab, sys.stderr has no real file descriptor, but the MCP stdio client needs one for the
    server's log output. In that case, send the server's log to mcp_server.log instead."""
    try:
        sys.stderr.fileno()
    except Exception:
        import agents.mcp.server as sdk_server
        from mcp.client.stdio import stdio_client as original_stdio_client
        log = open(Path(__file__).parent / "mcp_server.log", "a", encoding="utf-8")
        sdk_server.stdio_client = lambda params, *args, **kwargs: original_stdio_client(params, errlog=log)


_make_stdio_safe_for_notebooks()

SERVER_PATH = str(Path(__file__).parent / "server.py")
PROVIDER = os.getenv("PROVIDER", "gemini")
GEMINI_MODELS = [m for m in [os.getenv("GEMINI_MODEL"), "gemini-3.5-flash", "gemini-3.6-flash", "gemini-3-flash-preview",
                             "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-2.5-flash", "gemini-2.5-flash-lite"] if m]

INSTRUCTIONS = f"""You are a friendly outing planner for Lebanon. Today is {__import__('datetime').date.today().isoformat()}.
You have three MCP tools. Pick the ONE that matches the request:
- get_weather_forecast (Data): the user only wants to know the weather. Never saves anything.
- save_outing (Action): the user already chose a city, a date and an activity and wants it saved.
- plan_outing (Orchestration): the user wants you to FIND the best day for an activity (it checks the
  forecast and saves the best day in one step). Supported activities: hiking, beach, picnic, cycling, skiing.
If a tool returns an error, explain it to the user in plain words and suggest a fix. Never invent weather data.
Keep answers short and friendly; mention temperatures in °C."""


def make_model(name: str):
    if PROVIDER == "openai":
        return name or "gpt-4.1-mini"
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        sys.exit("Missing GEMINI_API_KEY. Put it in a .env file (see .env.example) or set it as an environment variable.")
    client = AsyncOpenAI(api_key=key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                         max_retries=0, timeout=60)  # don't wait on a busy model; switch to the next one instead
    return OpenAIChatCompletionsModel(model=name, openai_client=client)


def _output_text(output) -> str:
    # Tool outputs arrive as a string or a list of content parts; get the plain text
    if isinstance(output, list):
        output = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in output)
    if isinstance(output, dict) and "text" in output:
        output = output["text"]
    try:
        output = json.loads(output)
        if isinstance(output, dict) and "text" in output and len(output) <= 2:
            output = output["text"]
    except (TypeError, ValueError):
        pass
    return output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)


def print_round_trip(result) -> None:
    # Show the full round trip: model -> MCP client -> MCP server -> back
    for item in result.new_items:
        if item.type == "tool_call_item":
            raw = item.raw_item
            name = getattr(raw, "name", "?")
            args = getattr(raw, "arguments", "")
            print(f"   🔧 agent → MCP client → server: {name}({args})")
        elif item.type == "tool_call_output_item":
            text = _output_text(item.output)
            print(f"   📦 server → MCP client → agent: {text[:400]}{'...' if len(text) > 400 else ''}")


async def ask(agent_factory, question: str, history: list | None = None):
    """Run one user turn, trying the next Gemini model if one is busy or out of free quota.
    The model that worked is moved to the front, so later turns try it first."""
    global GEMINI_MODELS
    models = GEMINI_MODELS if PROVIDER == "gemini" else [os.getenv("OPENAI_MODEL", "gpt-4.1-mini")]
    last = None
    for attempt in range(2):
        for name in list(models):
            try:
                agent = agent_factory(make_model(name))
                result = await Runner.run(agent, (history or []) + [{"role": "user", "content": question}])
                if PROVIDER == "gemini" and GEMINI_MODELS[0] != name:
                    GEMINI_MODELS = [name] + [m for m in GEMINI_MODELS if m != name]
                return result, name
            except Exception as e:                # rate limit (429), overloaded (503), unknown model (404)
                text = str(e)
                if any(c in text for c in ("429", "500", "503", "404", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "NOT_FOUND",
                                           "timed out", "Timeout")):
                    code = next((c for c in ("429", "503", "500", "404") if c in text), "timeout")
                    reason = {"429": "rate limit / free quota used up", "503": "overloaded", "500": "server error",
                              "404": "not available for this key", "timeout": "no answer within 60 s"}.get(code, "error")
                    print(f"   [model] {name}: {code} {reason}, trying the next one...")
                    last = e
                    continue
                raise
        if attempt == 0:
            print("   [model] all models busy, waiting 30 s and trying again...")
            await asyncio.sleep(30)
    raise last


DEMO = [
    ("DATA tool",          "What's the weather going to be like in Byblos over the next 3 days?"),
    ("ACTION tool",        "Save a picnic in Beirut on {date}, note: bring the frisbee."),
    ("ORCHESTRATION tool", "Find me the best day this week to go hiking in Faraya and save it."),
    ("FAILURE case",       "What's the weather in Xyzzyville?"),
    ("RESOURCE check",     "Thanks! Which outings do I have saved now?"),
]


async def main(demo: bool):
    if demo:                                     # fresh outings file so the demo can be re-run (e.g. for a recording)
        demo_file = Path(__file__).parent / "data" / "demo_outings.json"
        demo_file.unlink(missing_ok=True)
        os.environ["OUTINGS_FILE"] = str(demo_file)
    async with MCPServerStdio(name="lebanon-outing-planner",
                              params={"command": sys.executable, "args": [SERVER_PATH], "env": dict(os.environ)},
                              cache_tools_list=True, client_session_timeout_seconds=30) as server:
        tools = await server.list_tools()
        print("Connected to MCP server. Tools:", ", ".join(t.name for t in tools))

        def factory(model):
            return Agent(name="Outing Planner", instructions=INSTRUCTIONS, mcp_servers=[server], model=model)

        history: list = []
        if demo:
            from datetime import date, timedelta
            for label, q in DEMO:
                q = q.format(date=(date.today() + timedelta(days=2)).isoformat())
                if label == "RESOURCE check":
                    saved = await server.read_resource("outings://saved")
                    print(f"\n=== {label} ===\n🧑 {q}\n   📚 resource outings://saved →\n{saved.contents[0].text}")
                    continue
                print(f"\n=== {label} ===\n🧑 {q}")
                result, used = await ask(factory, q, history)
                print_round_trip(result)
                print(f"🤖 ({used}) {result.final_output}")
                history = result.to_input_list()
            return

        print("Ask about the weather or plan an outing (type 'quit' to exit).")
        while True:
            q = input("\n🧑 You: ").strip()
            if q.lower() in ("quit", "exit", ""):
                break
            result, used = await ask(factory, q, history)
            print_round_trip(result)
            print(f"🤖 ({used}) {result.final_output}")
            history = result.to_input_list()


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)  # print each line immediately (otherwise output can be lost in notebooks)
    if "--offline" in sys.argv:              # use the server's built-in sample weather (no internet needed)
        os.environ["WEATHER_OFFLINE"] = "1"
    asyncio.run(main(demo="--demo" in sys.argv))
