# Part 1: MCP and the Fetch MCP Server

## Key MCP concepts

The Hugging Face article (and the full Turing Post piece it links to) presents the **Model Context Protocol** as an open standard, released by Anthropic in November 2024, for connecting AI applications to tools and data. Without it, M applications and N tools need M×N brittle custom integrations; MCP reduces this to M+N.

The architecture has three roles:

- The **Host** is the AI application the user interacts with (Claude Desktop, an IDE, or my Agents SDK program). It owns the LLM and decides when context or actions are needed.
- Inside the host, an **MCP Client** keeps a one-to-one connection with one server, speaking JSON-RPC over a transport such as stdio or HTTP.
- The **MCP Server** wraps a capability and advertises what it offers: **tools** (actions the model can call), **resources** (read-only context) and **prompts** (reusable templates).

Tools are discovered at runtime and chosen by the model, so their names and descriptions directly affect agent performance. The article also notes limitations: the overhead of running many servers, still-maturing security, and overkill for one or two simple APIs.

## The Fetch MCP server

The reference **Fetch** server (`mcp-server-fetch`) exposes a single tool, `fetch(url, max_length=5000, start_index=0, raw=false)`, plus a matching `fetch` prompt. It downloads a page and converts the HTML to markdown for the model.

**Strengths, judged against the tool-design principles from class:**

- **Well-defined interface:** typed arguments with defaults and bounds, and clear descriptions.
- **Predictable behavior on large pages:** content is truncated, and the response tells the model which `start_index` to request next, so pages can be read in chunks.
- **Clear errors:** it fails explicitly when `robots.txt` disallows a URL.

**Weaknesses:**

- **Several responsibilities in one tool:** downloading, HTML-to-markdown conversion, pagination and robots.txt policy are all bundled together.
- **A mode flag:** the `raw` boolean switches between two different output types, which makes the result format less predictable.
- **Context-dependent behavior:** its user-agent and robots.txt policy change depending on whether the model or the user triggered the request.
- **Security:** it can reach local and internal IP addresses, so network isolation has to be enforced outside the tool.

Overall, Fetch is a good example of a narrow, well-documented tool. A stricter design would separate "fetch raw content" from "convert to markdown".

*Sources:* https://huggingface.co/blog/Kseniase/mcp, https://www.turingpost.com/p/mcp, https://pypi.org/project/mcp-server-fetch
