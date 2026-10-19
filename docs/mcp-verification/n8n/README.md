# n8n MCP Verification

This directory contains independent verification evidence showing that n8n can
consume the SVHN Model Context Protocol (MCP) server as both:

- a direct no-code MCP client;
- an AI-agent-driven MCP client.

The verification also confirms that n8n may request and inspect controlled
runtime actions but cannot authorize them.

## Start n8n

n8n is run locally in Docker.

Create the persistent Docker volume once:

```bash
docker volume create n8n_data
```

Start n8n:

```bash
docker run -it --rm \
  --name n8n \
  -p 5678:5678 \
  -v n8n_data:/home/node/.n8n \
  docker.n8n.io/n8nio/n8n
```

Open the n8n editor:

```text
http://localhost:5678
```

The verified n8n version was:

```text
2.43.6
```

## Start the SVHN MCP Server

From the repository root:

```bash
source .venv/bin/activate
```

Start the MCP server using Streamable HTTP:

```bash
python -m mcp_server.server \
  --transport streamable-http \
  --host 0.0.0.0 \
  --port 8001
```

Expected startup endpoint:

```text
http://0.0.0.0:8001
```

Because n8n runs inside Docker, configure its MCP Client or MCP Client Tool
with:

```text
http://host.docker.internal:8001/mcp
```

## Verified Workflows

Direct no-code MCP invocation:

```text
Manual Trigger
    ↓
MCP Client
    ↓
SVHN MCP tool
```

AI-agent MCP invocation:

```text
Natural-language request
    ↓
AI Agent
    ↓
LLM
    ↓
MCP Client Tool
    ↓
SVHN MCP Server
```

Trusted controlled-action verification:

```text
n8n requests controlled change
    ↓
approval_required
    ↓
n8n cannot self-authorize
    ↓
trusted local human approval
    ↓
controlled execution
    ↓
n8n observes final state
```

## Verification Evidence

See:

```text
n8n_full_workflow_verification.txt
```

for the recorded end-to-end verification results.