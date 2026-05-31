from typing import List, Dict, Any, TypedDict, Optional

class AgentState(TypedDict):
    # Core state
    messages: List[Dict[str, str]]
    session_id: str
    query_id: str
    model_type: str  # "oss" or "frontier"
    
    # Execution variables
    current_response: str
    tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    refusal_message: Optional[str]
    
    # Execution safety limits
    step_count: int
    
    # Retry handling fields
    should_retry: Optional[bool]
    retry_delay: Optional[float]
