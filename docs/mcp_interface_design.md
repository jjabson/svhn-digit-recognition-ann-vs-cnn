# MCP Interface Design

## Phase 5.3A — Boundary and Tool Contracts

### Architectural Goal

Expose selected ML inference, evaluation, and system-inspection
capabilities to MCP clients and AI agents without coupling agent
frameworks to the core orchestration, evaluation, or persistence logic.

### Interface Model

FastAPI and MCP are peer interfaces over the same application/service
layer.

```text
HTTP Client ──► FastAPI ──┐
                          ├──► Application Services
Agent ────────► MCP ──────┘

```
## Tool Contract: `predict_digit`

### Purpose

Classify an SVHN-style digit image through the production inference
orchestration service.

The tool exposes the final orchestration decision rather than raw model
output so that agents consume the same confidence, fallback, and review
policy used by other application interfaces.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `image` | binary image data | yes | PNG or JPEG digit image |

The MCP adapter is responsible for validating and translating the
agent-supplied input into the image bytes expected by the application
service.

### Successful Tool Result

The tool returns structured decision information:

| Field | Type | Description |
|---|---|---|
| `predicted_digit` | `int \| null` | Final predicted digit |
| `confidence` | `float \| null` | Confidence of the selected prediction |
| `status` | `ACCEPTED \| UNCERTAIN \| FAILED` | Final orchestration status |
| `selected_model` | `str \| null` | Model selected by orchestration |
| `decision_reason` | `str \| null` | Reason for the final decision |
| `fallback_used` | `bool` | Whether fallback inference was used |
| `fallback_reason` | `str \| null` | Reason fallback was invoked |
| `review_required` | `bool` | Whether downstream review is required |

### Decision Semantics

#### ACCEPTED

The orchestration layer produced a prediction satisfying the configured
inference policy.

The agent may consume the prediction as an accepted system decision.

#### UNCERTAIN

Inference produced a usable prediction, but the result did not satisfy
the configured confidence policy.

The prediction and confidence remain available, but
`review_required` is true.

The agent must not reinterpret an UNCERTAIN result as ACCEPTED by applying
its own confidence threshold.

#### FAILED

Inference did not produce a usable final prediction.

`predicted_digit` and `confidence` may therefore be null and
`review_required` is true.

The agent must not manufacture or infer a replacement prediction.

### Failure Boundary

Invalid caller input is not an inference failure.

Malformed or unsupported image input must be represented as an MCP
tool/input error rather than converted into a FAILED inference decision.

A FAILED decision is reserved for failures occurring after valid input
has entered the inference execution boundary.

### Policy Ownership

The MCP client or agent does not choose the production confidence
threshold as part of `predict_digit`.

Inference policy remains owned by the application configuration and
orchestration layer.

Agents request decisions; they do not redefine inference policy.

## MCP Dependency Map

MCP tools are interface adapters over existing application and service
capabilities.

They must not bypass the application/service layer to call FastAPI routes,
SQLite persistence functions, TensorFlow models, or orchestration policy
functions directly.

| MCP Tool | Allowed Application Capability | Domain Meaning |
|---|---|---|
| `predict_digit` | `OrchestratedInferenceService` | Produce the final inference decision |
| `get_evaluation_summary` | evaluation service | Retrieve authoritative evaluation results |
| `get_evaluation_insights` | evaluation service | Retrieve derived evaluation intelligence |
| `get_digit_metrics` | evaluation service | Retrieve metrics for one digit class |
| `get_model_summary` | model service | Inspect the actively serving model |
| `get_inference_config` | inference service/configuration boundary | Inspect active serving configuration |

### Dependency Direction

```text
MCP Client / Agent
        │
        ▼
    MCP Tools
        │
        ▼
Application / Service Layer
        │
        ├──► Inference Orchestration
        ├──► Evaluation Logic
        ├──► Model Inspection
        └──► Configuration
                 │
                 ▼
       Domain / Persistence / Model
       
```

## Initial MCP Tool Surface

The first MCP interface intentionally exposes a small set of stable
application capabilities.

The goal is not to mirror every FastAPI endpoint. MCP tools are selected
based on whether the capability is useful and appropriate for an AI agent.

### Execution Tools

#### `predict_digit`

Runs production inference through the existing orchestration service.

This tool may cause model execution and fallback evaluation.

It returns the final `InferenceDecision` semantics rather than raw model
output.

### Read-Only Inspection Tools

#### `get_evaluation_summary`

Returns authoritative aggregate evaluation results and the protocol that
produced them.

#### `get_evaluation_insights`

Returns derived evaluation intelligence already owned by the evaluation
service.

#### `get_digit_metrics`

Returns evaluation metrics for one digit class.

Input:

- `digit: int`

The MCP adapter validates that the requested digit is within the supported
class domain before returning the appropriate evaluation result.

#### `get_model_summary`

Returns information describing the actively serving model and its
architecture/capacity.

#### `get_inference_config`

Returns the active inference configuration used by the serving system.

This is inspection only. The MCP client may not modify inference policy
through this tool.

### Tool Classification

| Tool | Classification | Causes Model Execution | Changes System State |
|---|---|---:|---:|
| `predict_digit` | execution | yes | no |
| `get_evaluation_summary` | read-only | no | no |
| `get_evaluation_insights` | read-only | no | no |
| `get_digit_metrics` | read-only | no | no |
| `get_model_summary` | read-only | no | no |
| `get_inference_config` | read-only | no | no |

### Deliberately Excluded From the Initial Surface

The first MCP version does not expose:

- raw model prediction that bypasses orchestration
- confidence-threshold overrides
- configuration mutation
- direct database access
- failure-injection controls
- test-only orchestration hooks
- direct TensorFlow/Keras execution
- persistence mutation
- reliability-policy mutation

These capabilities are either internal implementation details, testing
controls, or operations that would allow an agent to bypass established
system policy.

### Selection Principle

An MCP capability should be exposed because it provides meaningful
application behavior to an agent, not merely because a corresponding
Python function or HTTP endpoint exists.

## MCP Result and Error Semantics

MCP distinguishes application-domain outcomes from failures of the tool
request itself.

A valid application decision is returned as structured tool data.

A request that cannot validly enter or access the application capability
is represented as an MCP tool error.

### Result and Error Matrix

| Condition | MCP Behavior | Rationale |
|---|---|---|
| Successful application result | Structured result | Normal tool execution |
| `ACCEPTED` inference decision | Structured result | Valid orchestration outcome |
| `UNCERTAIN` inference decision | Structured result | Valid orchestration outcome requiring review |
| `FAILED` inference decision | Structured result | Valid orchestration outcome representing inference failure |
| Malformed image | Tool error | Invalid caller input |
| Unsupported image type | Tool error | Invalid caller input |
| Invalid digit argument | Tool error | Invalid caller input |
| Requested evaluation data unavailable | Tool error | Application capability cannot satisfy request |
| Inference service unavailable | Tool error | Tool cannot access required application service |
| Unexpected adapter/internal failure | Tool error | MCP operation itself could not complete |

### Inference Boundary

The following distinction must be preserved:

```text
Valid image
    │
    ▼
Inference orchestration
    │
    ├── ACCEPTED ──► structured MCP result
    │
    ├── UNCERTAIN ─► structured MCP result
    │
    └── FAILED ────► structured MCP result


Invalid image
    │
    └──────────────► MCP tool error
```
### Input Validation

Interface-specific validation belongs at the MCP boundary.

Examples include:

- required arguments
- supported image representation
- supported image format
- digit class range

Validation must not duplicate inference policy or model logic.

### Error Translation

MCP adapters may translate application exceptions into appropriate
MCP-facing tool errors.

They must not expose internal stack traces, database implementation
details, model filesystem paths, or other unnecessary implementation
details to clients.

### No HTTP Semantics

MCP contracts must not expose HTTP status codes such as:

- `400`
- `415`
- `503`

Those values belong to the FastAPI interface.

MCP uses MCP-native tool/result error semantics while preserving the same
underlying application meaning.

### Agent Responsibility

Agents may reason over valid structured results but must preserve
authoritative system semantics.

For example:

- `ACCEPTED` may be treated as an accepted inference decision.
- `UNCERTAIN` must remain uncertain and must preserve
  `review_required = true`.
- `FAILED` must remain a failed inference decision.
- An agent must not manufacture a prediction when the system returns
  `FAILED`.
- An agent must not reinterpret `UNCERTAIN` as `ACCEPTED` using its own
  confidence threshold.

### Error Design Principle

A domain failure and a tool failure answer different questions:

```text
Domain result:
"What did the application decide?"

Tool error:
"Could the requested application capability be invoked correctly?"
```

The MCP layer must preserve that distinction.