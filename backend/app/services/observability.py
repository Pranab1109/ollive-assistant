import time
import uuid
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class StepTrace(BaseModel):
    node: str
    latency_ms: int
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)

class GuardrailsTrace(BaseModel):
    input_safe: bool = True
    input_reason: str = ""
    output_safe: bool = True
    output_reason: str = ""

class PerformanceMetrics(BaseModel):
    total_latency_ms: int = 0
    time_to_first_token_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0

class ExecutionTrace(BaseModel):
    session_id: str
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model: str
    query: str
    response: str = ""
    success: bool = True
    error_message: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    metrics: PerformanceMetrics = Field(default_factory=PerformanceMetrics)
    guardrails: GuardrailsTrace = Field(default_factory=GuardrailsTrace)
    steps: List[StepTrace] = Field(default_factory=list)
    # Store tool call details with outputs for easy access in traces
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)

class TraceRepository:
    def __init__(self):
        self._traces: Dict[str, ExecutionTrace] = {}
        self._sessions: Dict[str, List[str]] = {}

    def create_trace(self, session_id: str, model: str, query: str, query_id: Optional[str] = None) -> ExecutionTrace:
        trace = ExecutionTrace(session_id=session_id, model=model, query=query)
        if query_id:
            trace.query_id = query_id
        self._traces[trace.query_id] = trace
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append(trace.query_id)
        return trace

    def get_trace(self, query_id: str) -> Optional[ExecutionTrace]:
        return self._traces.get(query_id)

    def get_session_traces(self, session_id: str) -> List[ExecutionTrace]:
        query_ids = self._sessions.get(session_id, [])
        return [self._traces[qid] for qid in query_ids if qid in self._traces]

    def add_step(self, query_id: str, node: str, latency_ms: int, details: Dict[str, Any] = None):
        trace = self.get_trace(query_id)
        if trace:
            step = StepTrace(node=node, latency_ms=latency_ms, details=details or {})
            trace.steps.append(step)

    def add_tool_result(self, query_id: str, tool_name: str, arguments: Dict[str, Any], output: Any):
        """Record a tool execution with its output.
        This populates the `tool_calls` list on the ExecutionTrace for later retrieval.
        """
        trace = self.get_trace(query_id)
        if trace:
            trace.tool_calls.append({
                "name": tool_name,
                "arguments": arguments,
                "output": output,
                "timestamp": time.time()
            })

    def update_guardrails(self, query_id: str, input_safe: Optional[bool] = None, input_reason: Optional[str] = None,
                          output_safe: Optional[bool] = None, output_reason: Optional[str] = None):
        trace = self.get_trace(query_id)
        if trace:
            if input_safe is not None:
                trace.guardrails.input_safe = input_safe
            if input_reason is not None:
                trace.guardrails.input_reason = input_reason
            if output_safe is not None:
                trace.guardrails.output_safe = output_safe
            if output_reason is not None:
                trace.guardrails.output_reason = output_reason

    def update_metrics(self, query_id: str, total_latency_ms: int = 0, time_to_first_token_ms: int = 0,
                       prompt_tokens: int = 0, completion_tokens: int = 0, estimated_cost_usd: float = 0.0):
        trace = self.get_trace(query_id)
        if trace:
            if total_latency_ms:
                trace.metrics.total_latency_ms = total_latency_ms
            if time_to_first_token_ms:
                trace.metrics.time_to_first_token_ms = time_to_first_token_ms
            if prompt_tokens:
                trace.metrics.prompt_tokens = prompt_tokens
            if completion_tokens:
                trace.metrics.completion_tokens = completion_tokens
            if estimated_cost_usd:
                trace.metrics.estimated_cost_usd = estimated_cost_usd

    def finalize_trace(self, query_id: str, response: str, success: bool = True, error_message: str = None):
        trace = self.get_trace(query_id)
        if trace:
            trace.response = response
            trace.success = success
            trace.error_message = error_message
            if trace.metrics.total_latency_ms == 0:
                trace.metrics.total_latency_ms = int((time.time() - trace.created_at) * 1000)

trace_repo = TraceRepository()
