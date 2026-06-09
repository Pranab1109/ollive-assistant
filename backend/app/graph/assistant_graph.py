import time
import re
import uuid
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END

from backend.app.config import settings
from backend.app.graph.state import AgentState
from backend.app.services.guardrails import GuardrailsService, session_guardrail
from backend.app.services.guardrail_engines import get_engine
from backend.app.services.assistant_oss import OSSAssistant
from backend.app.services.assistant_frontier import FrontierAssistant
from backend.app.services.observability import trace_repo
from backend.app.tools.hospital_tools import (
    list_all_specialties,
    search_doctors,
    check_doctor_availability,
    book_appointment,
    cancel_appointment,
    lookup_hospital_info,
    get_current_datetime,
    execute_math_calculation
)

logger = logging.getLogger(__name__)

# Initialize model helpers
oss_assistant = OSSAssistant()
frontier_assistant = FrontierAssistant()

# Mapping tool names to python functions
TOOL_MAPPING = {
    "list_all_specialties": list_all_specialties,
    "search_doctors": search_doctors,
    "check_doctor_availability": check_doctor_availability,
    "book_appointment": book_appointment,
    "cancel_appointment": cancel_appointment,
    "lookup_hospital_info": lookup_hospital_info,
    "get_current_datetime": get_current_datetime,
    "execute_math_calculation": execute_math_calculation
}

# Tools that take no arguments (called with no kwargs)
NO_ARG_TOOLS = {"get_current_datetime", "list_all_specialties"}

# Maximum tool-loop iterations to prevent infinite cycles
MAX_STEP_COUNT = 8


def is_confirmation_text(text: str) -> bool:
    if not text:
        return False
    text_lower = text.lower()
    if re.search(r'APT-[A-Z0-9]{4,}', text):
        return True
    
    # Patterns for confirmation
    patterns = [
        r'\b(?:appointment|booking|slot)?\s*(?:is|has\s+been)\s+(?:now\s+)?(?:booked|confirmed|scheduled)\b',
        r'\b(?:successfully|now)\s+(?:booked|confirmed|scheduled)\b',
        r'\b(?:booked|confirmed|scheduled)\s+successfully\b',
        r'\bconfirmed\s+(?:your|the)\s+appointment\b',
        r'\b(?:appointment|booking)\s+(?:confirmed|booked|scheduled)\b'
    ]
    for pattern in patterns:
        if re.search(pattern, text_lower):
            return True
    return False


def is_cancellation_text(text: str) -> bool:
    if not text:
        return False
    text_lower = text.lower()
    
    patterns = [
        r'\b(?:appointment|booking|slot|cancellation)?\s*(?:is|has\s+been)\s+(?:now\s+)?(?:cancelled|canceled)\b',
        r'\b(?:successfully|now)\s+(?:cancelled|canceled)\b',
        r'\b(?:cancelled|canceled)\s+successfully\b',
        r'\bcancelled\s+(?:your|the)\s+appointment\b',
        r'\b(?:appointment|booking|cancellation)\s+(?:cancelled|canceled)\b'
    ]
    for pattern in patterns:
        if re.search(pattern, text_lower):
            return True
    return False


def _extract_booking_from_hallucination(messages, hallucinated_response):
    """
    When the model hallucinates a booking confirmation (fake APT-ID, no tool call),
    extract patient_name, doctor_name, date, time from the conversation so we can
    inject a real book_appointment tool call.
    """
    doctor_name = None
    date_str = None
    time_slot = None
    patient_name = None

    tool_result_text = ""
    user_messages = []
    assistant_messages = []

    for msg in messages:
        content = msg.get("content", "") or ""
        if msg.get("role") == "user":
            user_messages.append(content.strip())
        elif msg.get("role") == "assistant":
            assistant_messages.append(content.strip())
        elif msg.get("role") == "tool":
            tool_result_text += content + "\n"

    # 1. DOCTOR NAME  — "Dr. First Last"
    doc_pattern = r'(Dr\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+)'
    doc_matches = re.findall(doc_pattern, hallucinated_response)
    if not doc_matches:
        for am in reversed(assistant_messages):
            doc_matches = re.findall(doc_pattern, am)
            if doc_matches:
                break
    if not doc_matches:
        doc_matches = re.findall(doc_pattern, tool_result_text)
    if doc_matches:
        doctor_name = doc_matches[-1]

    # 2. TIME  — "HH:MM AM/PM"  or bare "9AM"
    time_pattern = r'(\d{1,2}:\d{2}\s*(?:AM|PM))'
    time_matches = re.findall(time_pattern, hallucinated_response, re.IGNORECASE)
    if not time_matches:
        for am in reversed(assistant_messages):
            time_matches = re.findall(time_pattern, am, re.IGNORECASE)
            if time_matches:
                break
    if not time_matches:
        for um in reversed(user_messages):
            time_matches = re.findall(time_pattern, um, re.IGNORECASE)
            if time_matches:
                break
    if time_matches:
        raw = time_matches[-1].strip()
        m = re.match(r'^(\d{1,2}):(\d{2})\s*(AM|PM)$', raw, re.IGNORECASE)
        if m:
            time_slot = f"{int(m.group(1)):02d}:{m.group(2)} {m.group(3).upper()}"
        else:
            time_slot = raw
    else:
        # Bare hour like "9AM" or "9 AM"
        bare_pattern = r'\b(\d{1,2})\s*(AM|PM)\b'
        bare_match = re.search(bare_pattern, hallucinated_response, re.IGNORECASE)
        if not bare_match:
            for am in reversed(assistant_messages):
                bare_match = re.search(bare_pattern, am, re.IGNORECASE)
                if bare_match:
                    break
        if not bare_match:
            for um in reversed(user_messages):
                bare_match = re.search(bare_pattern, um, re.IGNORECASE)
                if bare_match:
                    break
        if bare_match:
            h = int(bare_match.group(1))
            ap = bare_match.group(2).upper()
            time_slot = f"{h:02d}:00 {ap}"

    # 3. DATE  — resolve from tool results, user day-name references, or hallucinated text
    # Try finding YYYY-MM-DD
    date_pattern = r'(\d{4}-\d{2}-\d{2})'
    all_date_texts = [hallucinated_response] + list(reversed(assistant_messages)) + [tool_result_text]
    for text in all_date_texts:
        dates = re.findall(date_pattern, text)
        if dates:
            date_str = dates[-1]
            break

    # If YYYY-MM-DD not found, look for "Month DD"
    if not date_str:
        month_day_pattern = r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?\b'
        all_month_texts = [hallucinated_response] + list(reversed(assistant_messages)) + list(reversed(user_messages))
        for text in all_month_texts:
            month_match = re.search(month_day_pattern, text, re.IGNORECASE)
            if month_match:
                try:
                    now = datetime.now()
                    parsed = datetime.strptime(f"{month_match.group(1)} {month_match.group(2)} {now.year}", "%B %d %Y")
                    date_str = parsed.strftime("%Y-%m-%d")
                    break
                except ValueError:
                    pass

    # If still not found, check day names (e.g. Friday, tomorrow)
    if not date_str:
        for text in list(reversed(user_messages)) + list(reversed(assistant_messages)):
            text_lower = text.lower()
            found_day = None
            for day in ["today", "tomorrow", "monday", "tuesday", "wednesday",
                        "thursday", "friday", "saturday", "sunday"]:
                if day in text_lower:
                    found_day = day
                    break
            if found_day:
                now = datetime.now()
                if found_day == "today":
                    date_str = now.strftime("%Y-%m-%d")
                elif found_day == "tomorrow":
                    date_str = (now + timedelta(days=1)).strftime("%Y-%m-%d")
                else:
                    tool_dates = re.findall(r'(\d{4}-\d{2}-\d{2})', tool_result_text)
                    for dm in tool_dates:
                        try:
                            dt = datetime.strptime(dm, "%Y-%m-%d")
                            if dt.strftime("%A").lower() == found_day:
                                date_str = dm
                                break
                        except ValueError:
                            pass
                    if not date_str:
                        for i in range(0, 8):
                            future = now + timedelta(days=i)
                            if future.strftime("%A").lower() == found_day:
                                date_str = future.strftime("%Y-%m-%d")
                                break
                if date_str:
                    break

    # 4. PATIENT NAME  — try assistant messages first (e.g. "Thank you, Pranab Saha!"), then fallback to user messages
    for am in reversed(assistant_messages):
        thanks_match = re.search(r'Thank\s+you,?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', am)
        if thanks_match:
            patient_name = thanks_match.group(1)
            break
        patient_match = re.search(r'Patient:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', am, re.IGNORECASE)
        if patient_match:
            patient_name = patient_match.group(1)
            break

    if not patient_name:
        skip_words = {"yes", "no", "confirm", "book it", "go ahead", "proceed",
                      "sure", "ok", "okay", "yeah", "yep", "please", "y", "book"}
        for um in reversed(user_messages):
            if um.lower() in skip_words:
                continue
            if "?" in um or len(um) > 60 or len(um) < 2:
                continue
            if any(kw in um.lower() for kw in
                   ["available", "cardio", "neuro", "doctor", "appointment",
                    "slot", "cancel", "department", "specialty", "am", "pm"]):
                continue
            words = um.split()
            if 1 <= len(words) <= 4 and all(w[0].isupper() for w in words if w):
                patient_name = um
                break

    if all([doctor_name, date_str, time_slot, patient_name]):
        return {
            "patient_name": patient_name,
            "doctor_name": doctor_name,
            "date": date_str,
            "time": time_slot
        }

    logger.warning(
        f"Auto-booking extraction incomplete: doctor={doctor_name}, "
        f"date={date_str}, time={time_slot}, patient={patient_name}"
    )
    return None


# --- NODES ---

async def input_guardrail_node(state: AgentState) -> Dict[str, Any]:
    start_time = time.time()
    query_id = state["query_id"]
    session_id = state["session_id"]
    
    # 1. Check Session rate limiting
    if session_guardrail.is_blocked(session_id):
        refusal = "Session temporarily blocked due to repeated policy violations."
        latency = int((time.time() - start_time) * 1000)
        trace_repo.add_step(query_id, "InputGuardrail", latency, {"safe": False, "reason": refusal})
        trace_repo.update_guardrails(query_id, input_safe=False, input_reason=refusal)
        return {
            "refusal_message": refusal
        }
        
    # Get last user message
    user_query = ""
    for msg in reversed(state["messages"]):
        if msg["role"] == "user":
            user_query = msg["content"]
            break
            
    if not user_query:
        return {
            "refusal_message": "No query found."
        }
        
    is_safe, reason = await get_engine().verify_input(user_query)
    
    # If not safe, record the flag on the session
    if not is_safe:
        session_guardrail.record_flag(session_id)
        
    latency = int((time.time() - start_time) * 1000)
    trace_repo.add_step(query_id, "InputGuardrail", latency, {"safe": is_safe, "reason": reason})
    trace_repo.update_guardrails(query_id, input_safe=is_safe, input_reason=reason)
    
    if not is_safe:
        if "emergency" in reason.lower() or "out-of-scope" in reason.lower() or "this may be a medical emergency" in reason.lower() or "policy violation" in reason.lower():
            refusal = reason
        else:
            refusal = f"I apologize, but I cannot assist with that request. Reason: {reason}"
        return {
            "refusal_message": refusal
        }
        
    return {
        "refusal_message": None
    }

def extract_retry_delay(exception: Exception) -> float:
    err_str = str(exception)
    
    # Check for "retry_delay { seconds: 43 }"
    match = re.search(r'retry_delay\s*\{\s*seconds:\s*(\d+)', err_str)
    if match:
        return float(match.group(1))
        
    # Check for "Please retry in 43.794388572s"
    match2 = re.search(r'Please\s+retry\s+in\s+([\d.]+)\s*s', err_str, re.IGNORECASE)
    if match2:
        return float(match2.group(1))
        
    # Check for "retry in 43 seconds" or "retry in 43s"
    match3 = re.search(r'retry\s+in\s+([\d.]+)\s*s(?:econds)?', err_str, re.IGNORECASE)
    if match3:
        return float(match3.group(1))
        
    # Check for generic numbers with seconds
    match4 = re.search(r'(\d+)\s*seconds', err_str, re.IGNORECASE)
    if match4:
        return float(match4.group(1))

    return 5.0  # default fallback delay


async def llm_inference_node(state: AgentState) -> Dict[str, Any]:
    query_id = state["query_id"]
    model_type = state["model_type"]
    messages = state["messages"]
    
    max_retries = 2
    retry_count = 0
    
    while True:
        try:
            # Map generic "oss" to the correct target for backward compatibility/evals
            active_model = model_type
            if active_model == "oss":
                active_model = "oss_hf" if settings.HF_SPACE_MODEL_URL else "oss_local"

            if active_model == "oss_hf":
                text_response, tool_calls = await oss_assistant.generate_response(messages, query_id, use_hf=True)
            elif active_model == "oss_local":
                text_response, tool_calls = await oss_assistant.generate_response(messages, query_id, use_hf=False)
            else:
                text_response, tool_calls = await frontier_assistant.generate_response(messages, query_id)
    
            # ── Hallucination interceptor: only trigger if book_appointment
            #    has NOT already succeeded in this turn's tool_results ──
            if not tool_calls and text_response:
                existing_results = state.get("tool_results", []) or []
                already_booked = any(
                    r.get("tool") == "book_appointment"
                    and "Error" not in str(r.get("result", ""))
                    and "confirmed" in str(r.get("result", "")).lower()
                    for r in existing_results
                )
                if is_confirmation_text(text_response) and not already_booked:
                    details = _extract_booking_from_hallucination(messages, text_response)
                    if details:
                        logger.info(f"Auto-booking interceptor: forcing book_appointment with {details}")
                        tool_calls = [{
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "name": "book_appointment",
                            "arguments": details
                        }]
                        text_response = ""  # clear hallucinated text; real response comes after tool
            # ─────────────────────────────────────────────────────────
    
            # Append this assistant response to message history in state
            updated_messages = list(messages)
            assistant_msg = {
                "role": "assistant",
                "content": text_response
            }
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
                
            updated_messages.append(assistant_msg)
                
            return {
                "current_response": text_response,
                "tool_calls": tool_calls,
                "messages": updated_messages,
                "step_count": state.get("step_count", 0) + 1,
                "should_retry": False,
                "retry_delay": 0.0
            }
        except Exception as e:
            logger.error(f"Inference error in LLM node (attempt {retry_count + 1}): {e}")
            err_str = str(e).lower()
            is_rate_limit = any(term in err_str for term in ["429", "quota", "exhausted", "rate limit"])
            
            if is_rate_limit:
                delay_sec = extract_retry_delay(e)
                
                # If backend retry is feasible (delay <= 3s) and we haven't exceeded max_retries:
                if delay_sec <= 3.0 and retry_count < max_retries:
                    retry_count += 1
                    logger.info(f"Rate limit hit. Retrying on backend in {delay_sec}s (attempt {retry_count}/{max_retries})...")
                    await asyncio.sleep(delay_sec)
                    continue
                
                # Otherwise, return to frontend for a visible user retry
                err_response = (
                    f"Evergreen Medical Center's hosted assistant (Gemini) is rate-limited. "
                    f"Retrying in {int(delay_sec)} seconds..."
                )
                
                updated_messages = list(messages)
                updated_messages.append({"role": "assistant", "content": err_response})
                
                return {
                    "current_response": err_response,
                    "tool_calls": [],
                    "refusal_message": err_response,
                    "messages": updated_messages,
                    "step_count": state.get("step_count", 0) + 1,
                    "should_retry": True,
                    "retry_delay": delay_sec
                }
            else:
                # Non-rate-limit error, return immediately
                err_response = "I encountered an error while processing your request. Please try again."
                refusal_message = f"System Error: {str(e)}"
                
                updated_messages = list(messages)
                updated_messages.append({"role": "assistant", "content": err_response})
                
                return {
                    "current_response": err_response,
                    "tool_calls": [],
                    "refusal_message": refusal_message,
                    "messages": updated_messages,
                    "step_count": state.get("step_count", 0) + 1,
                    "should_retry": False,
                    "retry_delay": 0.0
                }

async def tool_executor_node(state: AgentState) -> Dict[str, Any]:
    start_time = time.time()
    query_id = state["query_id"]
    tool_calls = state["tool_calls"]
    tool_results = state.get("tool_results", []) or []
    messages = list(state["messages"])
    
    new_results = []
    
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        args = tool_call["arguments"]
        tool_call_id = tool_call.get("id") or f"call_{str(uuid.uuid4())[:8]}"
        
        tool_start = time.time()
        logger.info(f"Executing tool {tool_name} with args {args}")
        
        # Resolve function
        func = TOOL_MAPPING.get(tool_name)
        if func:
            try:
                if tool_name in NO_ARG_TOOLS:
                    result = await func()
                else:
                    result = await func(**args)
            except Exception as e:
                result = f"Error executing tool {tool_name}: {str(e)}"
                logger.error(f"Tool execution error for {tool_name}: {e}")
        else:
            result = f"Error: Tool '{tool_name}' is not available. Available tools: {', '.join(TOOL_MAPPING.keys())}."
            
        tool_latency = int((time.time() - tool_start) * 1000)
        
        # Log to trace
        trace_repo.add_step(query_id, "ToolExecutor", tool_latency, {
            "tool": tool_name,
            "arguments": args,
            "output": result
        })
        # Record tool result separately for easy access in trace_output
        trace_repo.add_tool_result(query_id, tool_name, args, result)
        
        new_results.append({
            "tool": tool_name,
            "args": args,
            "result": result
        })
        
        # Append tool response in native formatting
        messages.append({
            "role": "tool",
            "name": tool_name,
            "tool_call_id": tool_call_id,
            "content": result
        })
        
    return {
        "tool_results": tool_results + new_results,
        "tool_calls": [],  # Clear pending tool calls
        "messages": messages
    }

async def output_guardrail_node(state: AgentState) -> Dict[str, Any]:
    start_time = time.time()
    query_id = state["query_id"]
    refusal = state.get("refusal_message")
    response = state.get("current_response", "") or ""
    messages = list(state["messages"])
    tool_results = state.get("tool_results", []) or []
    
    # ── FIX: If response is empty but tools ran, synthesize from results ──
    if not response.strip() and tool_results:
        last_result = tool_results[-1]
        last_tool_name = last_result.get("tool", "")
        last_tool_output = str(last_result.get("result", ""))
        
        if last_tool_name == "book_appointment" and "confirmed" in last_tool_output.lower():
            response = f"Your appointment has been booked successfully! Here are the details:\n\n{last_tool_output}"
        elif last_tool_name == "cancel_appointment" and "cancelled" in last_tool_output.lower():
            response = f"Your appointment has been cancelled. {last_tool_output}"
        elif last_tool_name == "book_appointment" and "Error" in last_tool_output:
            response = f"I wasn't able to complete the booking. {last_tool_output}\n\nWould you like to try a different time slot?"
        elif last_tool_output:
            response = last_tool_output
        
        if response.strip():
            if messages and messages[-1]["role"] == "assistant":
                messages[-1]["content"] = response
            logger.info(f"Synthesized response from tool result (was empty): {last_tool_name}")
    
    if refusal:
        latency = int((time.time() - start_time) * 1000)
        trace_repo.add_step(query_id, "OutputGuardrail", latency, {"status": "skipped", "reason": "Input already failed guardrails."})
        return {
            "current_response": refusal
        }
        
    is_safe, reason = await GuardrailsService.verify_output(response)
    
    latency = int((time.time() - start_time) * 1000)
    trace_repo.add_step(query_id, "OutputGuardrail", latency, {"safe": is_safe, "reason": reason})
    trace_repo.update_guardrails(query_id, output_safe=is_safe, output_reason=reason)
    
    if not is_safe:
        override_response = (
            "I apologize, but as Evergreen Medical Center's receptionist, I am not authorized to offer "
            "medical diagnosis or treatment advice. I highly recommend booking an appointment with "
            "Dr. David Kim (General Medicine) so he can evaluate your symptoms, or visiting the emergency ward."
        )
        if messages and messages[-1]["role"] == "assistant":
            messages[-1]["content"] = override_response
            
        return {
            "current_response": override_response,
            "messages": messages
        }
        
    # --- BOOKING & CANCELLATION HALLUCINATION GUARDRAILS ---
    is_confirming_booking = is_confirmation_text(response)
    is_confirming_cancellation = is_cancellation_text(response)
    
    booked_successfully = False
    cancelled_successfully = False
    
    for res in tool_results:
        tool_name = res.get("tool")
        result_text = str(res.get("result", ""))
        
        if tool_name == "book_appointment" and "Error" not in result_text and "confirmed" in result_text.lower():
            booked_successfully = True
        if tool_name == "cancel_appointment" and "Error" not in result_text and "cancelled" in result_text.lower():
            cancelled_successfully = True

    # If model claims booking but not successful, override
    if is_confirming_booking and not booked_successfully:
        tool_error = next((res.get("result") for res in tool_results if res.get("tool") == "book_appointment" and "Error" in str(res.get("result"))), None)
        if tool_error:
            override_response = f"I apologize, but I could not book the appointment. The system returned the following error: {tool_error} Please let me know how you would like to proceed."
        else:
            override_response = "I apologize, but I am unable to confirm the appointment at this moment because the booking tool was not executed in our database. Would you like me to book it for you now?"
        
        logger.warning(f"Guardrail intercepted booking hallucination. Overriding response.")
        if messages and messages[-1]["role"] == "assistant":
            messages[-1]["content"] = override_response
        return {
            "current_response": override_response,
            "messages": messages
        }

    # If model claims cancellation but not successful, override
    if is_confirming_cancellation and not cancelled_successfully:
        tool_error = next((res.get("result") for res in tool_results if res.get("tool") == "cancel_appointment" and "Error" in str(res.get("result"))), None)
        if tool_error:
            override_response = f"I apologize, but I could not cancel the appointment. The system returned the following error: {tool_error}"
        else:
            override_response = "I apologize, but I am unable to confirm the cancellation because the cancellation tool was not executed in our database. Please provide the Booking ID so I can cancel it for you."
        
        logger.warning(f"Guardrail intercepted cancellation hallucination. Overriding response.")
        if messages and messages[-1]["role"] == "assistant":
            messages[-1]["content"] = override_response
        return {
            "current_response": override_response,
            "messages": messages
        }
        
    return {
        "current_response": response
    }

# --- ROUTING FUNCTIONS ---

def route_input_guardrail(state: AgentState) -> str:
    if state.get("refusal_message"):
        return "unsafe"
    return "safe"

def route_llm_output(state: AgentState) -> str:
    if state.get("refusal_message"):
        return "finalize"
        
    tool_calls = state.get("tool_calls", [])
    step_count = state.get("step_count", 0)
    
    # Dedup guard: if book_appointment already succeeded, skip further tool calls
    # to that same tool to prevent double-booking
    if tool_calls:
        existing_results = state.get("tool_results", []) or []
        already_booked = any(
            r.get("tool") == "book_appointment"
            and "Error" not in str(r.get("result", ""))
            and "confirmed" in str(r.get("result", "")).lower()
            for r in existing_results
        )
        if already_booked:
            # Filter out duplicate book_appointment calls
            filtered = [tc for tc in tool_calls if tc["name"] != "book_appointment"]
            if not filtered:
                return "finalize"
            # Update state with filtered calls (can't mutate directly, but
            # the remaining calls will proceed)
    
    if tool_calls and step_count < MAX_STEP_COUNT:
        return "execute_tool"
    return "finalize"

# --- BUILD GRAPH ---

workflow = StateGraph(AgentState)

workflow.add_node("input_guardrail", input_guardrail_node)
workflow.add_node("llm_inference", llm_inference_node)
workflow.add_node("tool_executor", tool_executor_node)
workflow.add_node("output_guardrail", output_guardrail_node)

workflow.set_entry_point("input_guardrail")

workflow.add_conditional_edges(
    "input_guardrail",
    route_input_guardrail,
    {
        "safe": "llm_inference",
        "unsafe": "output_guardrail"
    }
)

workflow.add_conditional_edges(
    "llm_inference",
    route_llm_output,
    {
        "execute_tool": "tool_executor",
        "finalize": "output_guardrail"
    }
)

workflow.add_edge("tool_executor", "llm_inference")
workflow.add_edge("output_guardrail", END)

assistant_graph = workflow.compile()
