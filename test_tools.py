"""
Tests for the three MCP tools, including the failure cases.

    pytest -v test_tools.py

Part A tests the tool logic directly (fast, no network: the HTTP call is replaced by sample data).
Part B goes through the real MCP protocol with an in-memory FastMCP client, exactly as an agent would.
Part C launches server.py as a subprocess over stdio with the OpenAI Agents SDK's MCP client.
"""
import asyncio
import datetime as dt
import json
import os
import sys
from pathlib import Path

import httpx
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

import server


def field(obj, new, old):
    # MCP SDK v2 renamed some fields (inputSchema -> input_schema ...); support both
    return getattr(obj, new, None) if getattr(obj, new, None) is not None else getattr(obj, old, None)


TODAY = dt.date.today()
ORIGINAL_HTTP_GET = server._http_get


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    # every test gets its own empty outings file and offline sample weather
    monkeypatch.setattr(server, "OUTINGS_FILE", tmp_path / "outings.json")
    monkeypatch.setattr(server, "_http_get", server._sample_response)


# ----------------------------------------------------------------------------- Part A: tool logic
def test_data_tool_returns_typed_forecast():
    f = server.fetch_forecast("Byblos", days=3)
    assert f.city == "Byblos" and f.country == "Lebanon"
    assert len(f.days) == 3 and f.days[0].date == TODAY.isoformat()
    assert f.days[0].condition == "rain" and f.days[2].rain_chance_pct == 5


def test_data_tool_unknown_city_fails_clearly():
    with pytest.raises(ToolError, match="not found"):
        server.fetch_forecast("Xyzzyville")


def test_data_tool_rejects_bad_days():
    with pytest.raises(ToolError, match="between 1 and 7"):
        server.fetch_forecast("Beirut", days=12)


def test_data_tool_service_down_fails_clearly(monkeypatch):
    def down(url, params, timeout):
        raise httpx.ConnectError("no route to host")
    monkeypatch.setattr(server, "OFFLINE", False)
    monkeypatch.setattr(server, "_http_get", ORIGINAL_HTTP_GET)      # the real network function
    monkeypatch.setattr(server.httpx, "get", down)                   # ...but the network is down
    with pytest.raises(ToolError, match="unavailable"):
        server.fetch_forecast("Beirut")


def test_action_tool_saves_outing():
    date = (TODAY + dt.timedelta(days=2)).isoformat()
    o = server.store_outing("Beirut", date, "Picnic", "bring the frisbee")
    assert o.id == 1 and o.activity == "picnic"
    assert json.loads(server.OUTINGS_FILE.read_text())[0]["note"] == "bring the frisbee"


@pytest.mark.parametrize("bad_date, message", [("05/12/2026", "Invalid date"), ("2020-01-01", "in the past")])
def test_action_tool_rejects_bad_dates(bad_date, message):
    with pytest.raises(ToolError, match=message):
        server.store_outing("Beirut", bad_date, "picnic")
    assert not server.OUTINGS_FILE.exists()          # nothing written on failure


def test_action_tool_rejects_duplicates():
    date = TODAY.isoformat()
    server.store_outing("Beirut", date, "picnic")
    with pytest.raises(ToolError, match="already saved"):
        server.store_outing("beirut", date, "PICNIC")


def test_orchestration_picks_best_day_and_saves():
    plan = server.plan_outing("Faraya", "hiking", days=5)
    # sample data: day 2 (19°C, 5% rain, 12 km/h wind) beats day 3 (21°C, 0% rain, 18 km/h)
    assert plan.status == "saved"
    assert plan.best_day.date == (TODAY + dt.timedelta(days=2)).isoformat()
    assert plan.saved_outing.city == "Faraya"
    assert len(json.loads(server.OUTINGS_FILE.read_text())) == 1


def test_orchestration_no_suitable_day_saves_nothing():
    plan = server.plan_outing("Byblos", "beach", days=3)       # sample days are far too cold for the beach
    assert plan.status == "no_suitable_day" and plan.saved_outing is None
    assert not server.OUTINGS_FILE.exists()


def test_orchestration_rejects_unknown_activity():
    with pytest.raises(ToolError, match="Unknown activity"):
        server.plan_outing("Beirut", "skydiving")


# ----------------------------------------------------------------------------- Part B: through the MCP protocol
def test_mcp_lists_exactly_three_tools_and_two_resources():
    async def run():
        async with Client(server.mcp) as c:
            tools = {t.name: t for t in await c.list_tools()}
            resources = {str(r.uri) for r in await c.list_resources()}
            return tools, resources
    tools, resources = asyncio.run(run())
    assert set(tools) == {"get_weather_forecast", "save_outing", "plan_outing"}
    assert field(tools["get_weather_forecast"], "input_schema", "inputSchema")["required"] == ["city"]
    assert field(tools["plan_outing"], "output_schema", "outputSchema") is not None                     # typed output
    assert resources == {"outings://saved", "outings://activities"}


def test_mcp_error_is_reported_not_silent():
    async def run():
        async with Client(server.mcp) as c:
            return await c.call_tool("get_weather_forecast", {"city": "Xyzzyville"}, raise_on_error=False)
    result = asyncio.run(run())
    assert result.is_error and "not found" in result.content[0].text


def test_mcp_orchestration_round_trip_and_resource():
    async def run():
        async with Client(server.mcp) as c:
            r = await c.call_tool("plan_outing", {"city": "Faraya", "activity": "hiking"})
            saved = await c.read_resource("outings://saved")
            return r, saved
    r, saved = asyncio.run(run())
    assert r.structured_content["status"] == "saved"
    assert json.loads(saved[0].text)[0]["activity"] == "hiking"


# ----------------------------------------------------------------------------- Part C: Agents SDK MCP client over stdio
def test_agents_sdk_stdio_client(tmp_path):
    from agents.mcp import MCPServerStdio

    async def run():
        env = {**os.environ, "WEATHER_OFFLINE": "1", "OUTINGS_FILE": str(tmp_path / "o.json")}
        params = {"command": sys.executable, "args": [str(Path(server.__file__))], "env": env}
        async with MCPServerStdio(params=params, name="test", client_session_timeout_seconds=30) as s:
            names = [t.name for t in await s.list_tools()]
            ok = await s.call_tool("get_weather_forecast", {"city": "Beirut", "days": 2})
            bad = await s.call_tool("save_outing", {"city": "Beirut", "date": "yesterday", "activity": "picnic"})
            return names, ok, bad
    names, ok, bad = asyncio.run(run())
    assert sorted(names) == ["get_weather_forecast", "plan_outing", "save_outing"]
    assert not field(ok, "is_error", "isError") and "Beirut" in ok.content[0].text
    assert field(bad, "is_error", "isError") and "Invalid date" in bad.content[0].text
