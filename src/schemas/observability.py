from pydantic import BaseModel


class InferenceMetricsResponse(BaseModel):
    total_attempts: int
    accepted: int
    uncertain: int
    failed: int

    total_latency_ms: float
    average_latency_ms: float
    max_latency_ms: float