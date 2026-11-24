"""
End-to-end test of the agent's round trip WITHOUT an API key:
a scripted fake LLM (it decides which tool to call) + the real OpenAI Agents SDK + the real MCP server over stdio.

    pytest -v test_agent_roundtrip.py
"""
import asyncio
import json
import os
import sys
from pathlib import Path

from agents import Agent, Runner, set_tracing_disabled
from agents.items import ModelResponse
from agents.models.interface import Model
from agents.mcp import MCPServerStdio
from agents.usage import Usage
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

set_tracing_disabled(True)
SERVER = str(Path(__file__).parent / "server.py")


class ScriptedModel(Model):
    """Plays the LLM: first asks for a tool call, then answers using the tool result it was given."""
    def __init__(self, tool_name, tool_args):
        self.tool_name, self.tool_args, self.seen_tools = tool_name, tool_args, []

    async def get_response(self, system_instructions, input, model_settings, tools, output_schema, handoffs,
                           tracing, **kwargs):
        self.seen_tools = [t.name for t in tools]
        outputs = [i for i in input if isinstance(i, dict) and i.get("type") == "function_call_output"]
        if not outputs:
            call = ResponseFunctionToolCall(type="function_call", id="fc_1", call_id="call_1",
                                            name=self.tool_name, arguments=json.dumps(self.tool_args))
            return ModelResponse(output=[call], usage=Usage(), response_id=None)
        text = f"Tool said: {outputs[-1]['output'][:200]}"
        msg = ResponseOutputMessage(type="message", id="m1", role="assistant", status="completed",
                                    content=[ResponseOutputText(type="output_text", text=text, annotations=[])])
        return ModelResponse(output=[msg], usage=Usage(), response_id=None)

    async def stream_response(self, *args, **kwargs):
        raise NotImplementedError


def run_agent(tool_name, tool_args, tmp_path):
    async def run():
        env = {**os.environ, "WEATHER_OFFLINE": "1", "OUTINGS_FILE": str(tmp_path / "o.json")}
        async with MCPServerStdio(params={"command": sys.executable, "args": [SERVER], "env": env},
                                  name="planner", client_session_timeout_seconds=30) as server:
            model = ScriptedModel(tool_name, tool_args)
            agent = Agent(name="Outing Planner", instructions="test", mcp_servers=[server], model=model)
            result = await Runner.run(agent, "hi")
            return model, result
    return asyncio.run(run())


def test_agent_sees_three_mcp_tools_and_calls_orchestration(tmp_path):
    model, result = run_agent("plan_outing", {"city": "Faraya", "activity": "hiking"}, tmp_path)
    assert sorted(model.seen_tools) == ["get_weather_forecast", "plan_outing", "save_outing"]
    types = [i.type for i in result.new_items]
    assert types[:2] == ["tool_call_item", "tool_call_output_item"]
    assert '"status":"saved"' in result.final_output.replace(" ", "")
    assert json.loads((tmp_path / "o.json").read_text())[0]["city"] == "Faraya"


def test_agent_receives_clear_error(tmp_path):
    model, result = run_agent("get_weather_forecast", {"city": "Xyzzyville"}, tmp_path)
    assert "not found" in result.final_output
