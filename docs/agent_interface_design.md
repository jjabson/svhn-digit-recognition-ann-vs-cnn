# Agent Interface Design

## Purpose

Phase 5.3D introduces an intelligent agent as a consumer of the ML
system's MCP interface.

The agent interprets natural-language requests, discovers and selects
application capabilities through MCP, combines structured results when
appropriate, and communicates those results to the user.

The agent may decide what information or capability to request, but it
must not redefine application-owned inference, reliability, fallback,
or review policy.

The initial agent implementation will use LangGraph for explicit agent
state, control flow, tool routing, and multi-step reasoning.

---

## Architecture

The agent is a peer consumer of the MCP interface and does not access
application internals directly.

```text
User
 │
 ▼
LangGraph Agent
 │
 │ MCP Client
 ▼
MCP Server
 │
 ▼
Application / Service Layer
 │
 ├── Inference Orchestration
 ├── Evaluation Services
 ├── Reliability Analysis
 └── Configuration
 │
 ├── TensorFlow
 └── SQLite
```

The architectural responsibility of each layer is intentionally
separate:

- The agent decides which capability it needs.
- MCP defines and transports the capability contract.
- The application remains authoritative for ML behavior and policy.

---

## Agent Role

The initial agent acts as an **ML System Analyst Agent**.

Its responsibilities are to:

- interpret natural-language user intent;
- discover available capabilities through MCP;
- select appropriate MCP tools;
- supply valid tool arguments;
- inspect structured tool results;
- perform multi-step tool reasoning when useful;
- combine related application facts;
- communicate those facts clearly to the user.

The agent is a consumer of application capabilities, not a replacement
for application orchestration.

---

## Capability Ownership

### Agent

The agent owns:

- natural-language intent interpretation;
- capability selection;
- deciding whether another tool result would be useful;
- multi-tool reasoning;
- result synthesis;
- conversational response generation.

### MCP Interface

The MCP layer owns:

- capability discovery;
- tool schemas;
- tool argument validation;
- protocol transport;
- translation between application results and MCP contracts;
- MCP tool error semantics.

### Application

The application remains authoritative for:

- model execution;
- production model selection;
- confidence policy;
- confidence thresholds;
- ACCEPTED / UNCERTAIN / FAILED decisions;
- fallback behavior;
- fallback reasons;
- review requirements;
- authoritative evaluation metrics;
- reliability facts and analyses.

---

## Agent Permissions

The agent may:

- inspect the active inference configuration;
- inspect model architecture;
- retrieve evaluation summaries;
- retrieve evaluation insights;
- retrieve per-digit evaluation metrics;
- request an orchestrated prediction;
- use the result of one MCP tool as input to another MCP tool;
- combine multiple structured application results into a response.

For example:

```text
predict_digit(image)
        │
        ▼
predicted_digit = 3
        │
        ▼
get_digit_metrics(3)
        │
        ▼
combined explanation
```

This is valid agent reasoning because the agent is selecting and
combining capabilities without redefining application policy.

---

## Agent Prohibitions

The agent must not:

- override the production confidence threshold;
- change an ACCEPTED decision to UNCERTAIN;
- change an UNCERTAIN decision to ACCEPTED;
- suppress `review_required`;
- independently select fallback models;
- directly load TensorFlow model files;
- directly query the application SQLite database;
- import application internals to bypass MCP;
- call FastAPI as a substitute for MCP;
- manufacture evaluation or reliability metrics;
- reinterpret caller/input errors as model failures;
- claim inference succeeded when the MCP request failed.

If application inference returns:

```text
confidence = 0.87
status = UNCERTAIN
review_required = true
```

the agent may explain the result, but it may not decide that 0.87 is
sufficiently confident and change the result to ACCEPTED.

---

## Decision Boundaries

Three types of decisions remain distinct.

### Agent Decision

The agent answers:

> What information or capability do I need next?

Example:

```text
Need historical performance for predicted digit 3
→ get_digit_metrics(3)
```

### MCP Interface Decision

The MCP interface answers:

> Is this a valid invocation of this capability?

Example:

```text
get_digit_metrics(digit=12)
→ MCP tool error
```

### Application Decision

The application answers:

> What does this ML result mean operationally?

Example:

```text
confidence = 0.87
production threshold = 0.90

→ UNCERTAIN
→ review_required = true
```

---

## Required Agent Scenarios

The initial agent must support five representative scenarios.

### 1. Configuration Inquiry

Example:

> What confidence threshold is the system using?

Expected capability:

```text
get_inference_config
```

The agent should not execute unrelated inference or evaluation tools.

### 2. Evaluation Inquiry

Example:

> How accurate is the model?

Expected capability:

```text
get_evaluation_summary
```

The agent explains authoritative evaluation facts returned by the
application.

### 3. Evaluation Investigation

Example:

> Which digit performs worst and how does it perform?

Expected flow:

```text
get_evaluation_insights
        │
        ▼
worst_class = X
        │
        ▼
get_digit_metrics(X)
```

This is a required multi-tool scenario.

### 4. Prediction

Example:

> Predict this image.

Expected capability:

```text
predict_digit(image)
```

The agent preserves the application's inference decision semantics,
including status, reasons, fallback information, and review
requirements.

### 5. Prediction With Historical Context

Example:

> Predict this image and tell me how the model historically performs
> on that digit.

Expected flow:

```text
predict_digit(image)
        │
        ▼
predicted_digit = X
        │
        ▼
get_digit_metrics(X)
        │
        ▼
combined explanation
```

This is the primary demonstration of result-dependent multi-tool agent
reasoning.

---

## Failure Semantics

Tool/request failures and valid application inference failures are
different outcomes.

### Tool or Request Failure

Examples include:

- malformed Base64;
- invalid image data;
- unsupported tool arguments;
- unavailable capability.

These remain MCP tool errors.

### Application Inference Failure

A valid inference request may produce an application decision such as:

```text
status = FAILED
decision_reason = ...
review_required = true
```

This remains structured application data.

The agent must preserve this distinction.

Bad caller input must not be represented as a failed model inference.

---

## Agent State Requirements

The exact LangGraph state representation will be selected during
implementation.

Conceptually, agent state must support:

```text
Agent State
│
├── conversation/messages
├── optional image context
├── discovered MCP capabilities
├── structured tool observations
└── execution context required for the next reasoning step
```

Application-owned policy must not be independently recreated in agent
state.

If the agent requires application configuration or inference status,
it obtains those facts through MCP.

---

## MCP Capability Discovery

Agent-visible tool contracts must originate from MCP discovery rather
than manually duplicated SVHN application schemas.

Conceptually:

```text
MCP connection
      │
      ▼
session.initialize()
      │
      ▼
session.list_tools()
      │
      ▼
MCP tool contracts
      │
      ▼
agent-compatible tools
```

The agent integration must not import MCP server tool implementations
directly.

This preserves MCP as the interoperability boundary.

---

## Image Handling

The current `predict_digit` MCP capability accepts Base64-encoded image
data.

Large Base64 payloads should not become part of normal LLM reasoning
context.

Image transport belongs at the client/tool execution boundary:

```text
Image
  │
  ▼
Client / Tool Adapter
  │
  ├── read bytes
  ├── Base64 encode
  │
  ▼
predict_digit
```

The agent should reason primarily about the availability and purpose of
the image rather than its encoded representation.

---

## LangGraph Responsibility

LangGraph provides explicit agent state, control flow, tool routing,
and multi-step reasoning.

Conceptually:

```text
START
  │
  ▼
Agent Node
  │
  ├── final response ──────────────► END
  │
  └── tool request
          │
          ▼
      MCP Tool Node
          │
          ▼
      update state
          │
          └────────────────────────► Agent Node
```

LangGraph must not duplicate deterministic application orchestration.

The agent graph therefore does not implement separate confidence,
fallback, model-selection, or accept/reject policy nodes.

Those responsibilities remain inside the application.

---

## Testing Contract

Agent regression tests should verify deterministic behavioral facts
rather than exact LLM wording.

Tests should verify:

- correct MCP capability selection;
- correct tool arguments;
- correct use of intermediate tool results;
- expected state transitions;
- preservation of application inference status;
- preservation of `review_required`;
- preservation of fallback semantics;
- correct distinction between MCP tool errors and FAILED inference
  decisions;
- multi-tool behavior when the request requires multiple capabilities.

Tests should not require exact natural-language response strings from a
real LLM.

Deterministic agent tests should use controlled or fake model behavior
where appropriate.

Real-LLM execution should be treated as an integration demonstration,
not as a deterministic regression contract.

---

## Future Human Review

Human-in-the-loop execution is intentionally deferred from the initial
Phase 5.3D implementation.

The existing application field:

```text
review_required
```

provides a natural future LangGraph branch:

```text
InferenceDecision
       │
       ├── review_required = false → continue
       │
       └── review_required = true
                    │
                    ▼
               Human Review
                    │
                    ▼
                  resume
```

This capability should be added only after the basic LangGraph → MCP →
application integration is established.

---

## Engineering Principles

1. **The agent decides which capability to request; the application
   remains authoritative for what that capability means.**

2. **Agents consume application capabilities through MCP rather than
   bypassing the interoperability boundary.**

3. **MCP tool discovery is the source of agent-visible capability
   contracts.**

4. **Agent reasoning may combine application facts, but it must not
   manufacture or override application policy.**

5. **Tool/request failures and valid FAILED inference decisions are
   distinct outcomes and must remain distinct through the agent
   layer.**

6. **Binary transport details such as Base64 image data belong at the
   client/tool boundary, not in agent reasoning.**

7. **Agent tests verify tool selection, arguments, state transitions,
   and preserved application semantics rather than nondeterministic
   natural-language wording.**

8. **Agent autonomy should expand only where it adds a new capability;
   deterministic behavior remains deterministic.**