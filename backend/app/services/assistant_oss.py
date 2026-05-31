import time
import json
import re
import uuid
import logging
import httpx
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from backend.app.config import settings
from backend.app.services.observability import trace_repo

logger = logging.getLogger(__name__)

# --- DYNAMIC SYSTEM PROMPT BUILDER ---

def _build_system_prompt() -> str:
    """
    Builds the system instruction with the current datetime injected.
    Optimized for local Llama/Qwen model with custom JSON tool calling.
    """
    now = datetime.now()
    tomorrow = now + timedelta(days=1)
    
    # Compute next weekdays
    next_days = {}
    for i in range(1, 8):
        future = now + timedelta(days=i)
        day_name = future.strftime("%A")
        if day_name not in next_days:
            next_days[day_name] = future.strftime("%Y-%m-%d")
    
    next_days_str = ", ".join(f"next {d} = {dt}" for d, dt in next_days.items())
    
    return f"""You are Olive, the AI Receptionist at Evergreen Medical Center. Warm, professional, and concise.

=== ABSOLUTE RULES — NEVER BREAK THESE ===
1. NEVER mention doctor names unless you retrieved them from a tool response (search_doctors or list_all_specialties).
2. NEVER say an appointment is confirmed/booked unless the book_appointment tool returned a success result with a real ID starting with "APT-".
3. NEVER invent Booking IDs or make up time slots. Use actual tool results.
4. ALWAYS call search_doctors first before selecting/suggesting any doctor.
5. If a tool returns an error (e.g. check_doctor_availability or book_appointment says doctor is unavailable or has no hours), you MUST explain the issue to the patient and ask them to select another date/doctor. Never assume success.
6. You MUST call the search_doctors tool immediately in your very first response when the user mentions any medical specialty, department, symptom, or doctor query. NEVER answer with text or ask clarification questions before calling the search_doctors tool.
7. When you have ALL FOUR booking details (patient_name, doctor_name, date, time), you MUST call book_appointment IMMEDIATELY. Do NOT just generate a text summary — you MUST invoke the tool.
8. When the user says "yes", "confirm", "go ahead", "book it", "proceed", or any affirmation after you summarized booking details, you MUST call book_appointment with the details from the conversation. NEVER respond with text only.

=== CURRENT CONTEXT ===
Right now: {now.strftime('%A, %B %d, %Y %I:%M %p')}
Today: {now.strftime('%Y-%m-%d')} | Tomorrow: {tomorrow.strftime('%Y-%m-%d')}
Upcoming: {next_days_str}
(Convert relative dates yourself - do not ask patient for YYYY-MM-DD).

=== BOOKING FLOW ===
1. Search doctors: Call search_doctors immediately when specialty/symptom/informal term is mentioned.
2. Present doctors with their 3-day availability (already included in search results).
3. User picks a doctor, date, and time → ask for their full name.
4. User gives name → you now have all 4 details → IMMEDIATELY call book_appointment(patient_name, doctor_name, date, time). Do NOT ask for confirmation, do NOT generate a summary first, just CALL THE TOOL.

CRITICAL: The moment you know patient_name + doctor_name + date + time, your ONLY valid action is to call book_appointment. Generating text instead of calling the tool is WRONG.

=== TOOLS AVAILABLE ===
- Name: list_all_specialties
  Description: Lists all medical specialties available at Evergreen Medical Center.
  Parameters: None

- Name: search_doctors
  Description: Search for doctors by specialty. Returns doctor names, room, bio, extension, and their availability slots for the next 3 days.
  Parameters:
    - specialty: string (REQUIRED) - The medical specialty or informal term (e.g., 'cardio', 'heart', 'neuro', 'kids').

- Name: check_doctor_availability
  Description: Check available time slots for a specific doctor on a given date (YYYY-MM-DD format).
  Parameters:
    - doctor_name: string (REQUIRED) - Full name of the doctor (e.g. Dr. Alice Smith).
    - date: string (REQUIRED) - The date in YYYY-MM-DD format (e.g. 2026-05-29).

- Name: book_appointment
  Description: Book a confirmed appointment for a patient. Call this AFTER the details (name, doctor, date, time) are confirmed.
  Parameters:
    - patient_name: string (REQUIRED) - Full name of the patient.
    - doctor_name: string (REQUIRED) - Name of the doctor (e.g. Dr. Alice Smith).
    - date: string (REQUIRED) - Date in YYYY-MM-DD format.
    - time: string (REQUIRED) - Time slot (e.g. '10:00 AM').

- Name: cancel_appointment
  Description: Cancel an existing appointment by its Booking ID (e.g. APT-XXXXXXXX).
  Parameters:
    - booking_id: string (REQUIRED) - The Booking ID of the appointment to cancel.

- Name: lookup_hospital_info
  Description: Look up hospital FAQs including: policies, insurance accepted, room rates, contact details, parking fees, address, pharmacy hours, cafeteria info, and emergency info.
  Parameters:
    - query: string (REQUIRED) - A specific query about hospital services or policies.

- Name: get_current_datetime
  Description: Get the current date and time with computed upcoming dates.
  Parameters: None

- Name: execute_math_calculation
  Description: Safely evaluate basic mathematical expressions.
  Parameters:
    - expression: string (REQUIRED) - The math expression to compute (e.g. '150 * 3 + 20').

=== RESPONSE FORMAT ===
Your response MUST be a single, valid JSON object in one of the following two formats, and NOTHING ELSE. Do not wrap the JSON in code blocks (e.g., do not use ```json).

Format 1: When you need to call a tool:
{{
  "tool_calling_required": true,
  "tool_name": "tool_name",
  "args": {{
    "arg_name": "arg_value"
  }}
}}

Format 2: When you have the final answer or need to speak to the user:
{{
  "tool_calling_required": false,
  "response": "Your friendly text response here..."
}}

=== EXAMPLES OF TOOL TRIGGERS ===
- Specialty searches:
  User: "Who is available in cardiology?" -> Output:
  {{
    "tool_calling_required": true,
    "tool_name": "search_doctors",
    "args": {{
      "specialty": "Cardiology"
    }}
  }}

- When user picks a slot and gives name:
  User: "Book it for Friday 9AM" -> You ask for name:
  {{
    "tool_calling_required": false,
    "response": "Sure! What is your full name?"
  }}
  
  User: "Pranab Saha" -> Action: Call book_appointment:
  {{
    "tool_calling_required": true,
    "tool_name": "book_appointment",
    "args": {{
      "patient_name": "Pranab Saha",
      "doctor_name": "Dr. Alice Smith",
      "date": "{tomorrow.strftime('%Y-%m-%d')}",
      "time": "09:00 AM"
    }}
  }}

- When user confirms after a summary:
  User: "Yes" -> Action: Call book_appointment:
  {{
    "tool_calling_required": true,
    "tool_name": "book_appointment",
    "args": {{
      "patient_name": "Pranab Saha",
      "doctor_name": "Dr. Alice Smith",
      "date": "{tomorrow.strftime('%Y-%m-%d')}",
      "time": "09:00 AM"
    }}
  }}

- FAQ:
  User: "What insurance do you accept?" -> Output:
  {{
    "tool_calling_required": true,
    "tool_name": "lookup_hospital_info",
    "args": {{
      "query": "insurance accepted"
    }}
  }}

=== RESPONSE STYLE ===
- Warm, concise, one topic at a time.
- Format lists with bullet points.
- When presenting doctors from a search result, ALWAYS include their upcoming 3-day availability (dates and open time slots) as returned by the search_doctors tool. Never omit the availability section.
- Always end with a clear next step/question."""


class OSSAssistant:
    def __init__(self):
        self.model_name = settings.OSS_MODEL_NAME

    def _clean_response_text(self, text: str) -> str:
        """
        Aggressively clean any remaining JSON/tool-call artifacts from the model's text response.
        This handles cases where Qwen outputs partial tool call JSON inline.
        """
        if not text:
            return ""
        
        # Remove any remaining {"name": "...", "arguments": {...}} blocks
        text = re.sub(r'\{\s*"?name"?\s*:\s*"[^"]+"\s*,\s*"?arguments"?\s*:\s*\{[^}]*\}\s*\}', '', text)
        
        # Remove <tool_call>...</tool_call> XML wrappers
        text = re.sub(r'</?tool_call>', '', text)
        
        # Remove standalone JSON-like fragments (common Qwen artifacts)
        text = re.sub(r'```json\s*\{.*?\}\s*```', '', text, flags=re.DOTALL)
        text = re.sub(r'```\s*\{.*?\}\s*```', '', text, flags=re.DOTALL)
        
        # Clean up excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        
        return text

    async def generate_response(self, messages: List[Dict[str, Any]], query_id: str) -> Tuple[str, List[Dict[str, Any]]]:
        start_time = time.time()
        
        if settings.HF_SPACE_MODEL_URL:
            raw_url = settings.HF_SPACE_MODEL_URL.rstrip('/')
            if "api-inference.huggingface.co" in raw_url:
                base_url = "https://api-inference.huggingface.co/v1/chat/completions"
                match = re.search(r'/models/([^/]+/[^/]+)', raw_url)
                if match:
                    model_to_use = match.group(1)
                else:
                    model_to_use = "Qwen/Qwen2.5-7B-Instruct"
            else:
                if raw_url.endswith("/v1/chat/completions"):
                    base_url = raw_url
                elif raw_url.endswith("/v1"):
                    base_url = f"{raw_url}/chat/completions"
                else:
                    base_url = f"{raw_url}/v1/chat/completions"
                
                model_to_use = "tgi"
                match = re.search(r'/models/([^/]+/[^/]+)', raw_url)
                if match:
                    model_to_use = match.group(1)
        else:
            base_url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/v1/chat/completions"
            model_to_use = self.model_name

        # Build system prompt with current datetime injected
        system_prompt = _build_system_prompt()

        # Parse native history formatting into simplified user/assistant roles
        api_messages = [{"role": "system", "content": system_prompt}]
        
        for msg in messages:
            api_msg = {}
            if msg["role"] == "user":
                api_msg["role"] = "user"
                api_msg["content"] = msg["content"]
            elif msg["role"] == "assistant":
                api_msg["role"] = "assistant"
                if msg.get("tool_calls"):
                    # Format tool calls as the JSON structure
                    tc = msg["tool_calls"][0]
                    api_msg["content"] = json.dumps({
                        "tool_calling_required": True,
                        "tool_name": tc["name"],
                        "args": tc["arguments"]
                    })
                else:
                    api_msg["content"] = msg["content"]
            elif msg["role"] == "tool":
                api_msg["role"] = "user"
                api_msg["content"] = f"Tool result for {msg['name']}:\n{msg['content']}"
                
            api_messages.append(api_msg)

        payload = {
            "model": model_to_use,
            "messages": api_messages,
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }

        headers = {}
        if settings.HF_TOKEN:
            headers["Authorization"] = f"Bearer {settings.HF_TOKEN}"

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(base_url, json=payload, headers=headers)
                
                if response.status_code != 200:
                    err_msg = f"Inference server returned status code {response.status_code}: {response.text}"
                    logger.error(err_msg)
                    raise Exception(err_msg)

                data = response.json()
                latency_ms = int((time.time() - start_time) * 1000)

                # Parse token usage
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                
                trace_repo.update_metrics(
                    query_id,
                    time_to_first_token_ms=int(latency_ms * 0.4),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    estimated_cost_usd=0.0
                )

                # Extract content and parse our JSON structure
                choice = data["choices"][0]
                content = choice["message"].get("content") or ""
                
                text_response = ""
                tool_calls = []
                
                try:
                    # Find outer JSON boundaries
                    start_idx = content.find('{')
                    end_idx = content.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_str = content[start_idx:end_idx + 1]
                    else:
                        json_str = content
                        
                    response_json = json.loads(json_str)
                    
                    tool_calling_required = response_json.get("tool_calling_required", False)
                    if tool_calling_required:
                        tool_name = response_json.get("tool_name")
                        args = response_json.get("args", {})
                        if tool_name:
                            tool_calls.append({
                                "id": f"call_{uuid.uuid4().hex[:8]}",
                                "name": tool_name,
                                "arguments": args
                            })
                    else:
                        text_response = response_json.get("response", "")
                except Exception as e:
                    logger.error(f"Error parsing JSON response from OSS LLM: {e}. Content was: {content}")
                    # Fallback
                    text_response = content

                # Always clean the response text to remove any remaining artifacts
                text_response = self._clean_response_text(text_response)

                trace_repo.add_step(
                    query_id,
                    "LLMNode",
                    latency_ms,
                    {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "tool_calls_requested": [tc["name"] for tc in tool_calls],
                        "raw_response": content
                    }
                )

                return text_response, tool_calls

        except httpx.ConnectError as ce:
            is_hf = "huggingface" in base_url
            dest = "Hugging Face Inference API" if is_hf else "local Ollama"
            logger.error(f"Failed to connect to {dest}: {ce}")
            trace_repo.add_step(query_id, "LLMNode", int((time.time() - start_time) * 1000), {"error": f"{dest} connection failure."})
            if is_hf:
                raise Exception("Cannot connect to Hugging Face Inference API. Check the URL and token.")
            else:
                raise Exception("Cannot connect to local Ollama server. Check that Ollama is running.")
        except Exception as e:
            logger.error(f"Error calling OSS model: {e}")
            trace_repo.add_step(query_id, "LLMNode", int((time.time() - start_time) * 1000), {"error": str(e)})
            raise e
