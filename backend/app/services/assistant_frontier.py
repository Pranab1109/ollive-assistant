import time
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse
from backend.app.config import settings
from backend.app.services.observability import trace_repo

logger = logging.getLogger(__name__)

COST_INPUT_PER_1M = 0.075
COST_OUTPUT_PER_1M = 0.30

# --- DYNAMIC SYSTEM PROMPT BUILDER ---

def _build_system_prompt() -> str:
    """
    Builds the system instruction with the current datetime injected.
    Identical prompt to OSS for fair A/B comparison.
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
4. NEVER output raw JSON or code blocks.
5. ALWAYS call search_doctors first before selecting/suggesting any doctor.
6. If a tool returns an error (e.g. check_doctor_availability or book_appointment says doctor is unavailable or has no hours), you MUST explain the issue to the patient and ask them to select another date/doctor. Never assume success.
7. You MUST call the search_doctors tool immediately in your very first response when the user mentions any medical specialty, department, symptom, or doctor query. NEVER answer with text or ask clarification questions before calling the search_doctors tool.

=== CURRENT CONTEXT ===
Right now: {now.strftime('%A, %B %d, %Y %I:%M %p')}
Today: {now.strftime('%Y-%m-%d')} | Tomorrow: {tomorrow.strftime('%Y-%m-%d')}
Upcoming: {next_days_str}
(Convert relative dates yourself - do not ask patient for YYYY-MM-DD).

=== BOOKING FLOW ===
1. Search doctors: Call search_doctors immediately when specialty/symptom/informal term is mentioned.
2. Confirm doctor preference.
3. Get date, then call check_doctor_availability.
4. Show slots, confirm slot.
5. Ask full name.
6. Summarize details and ask to confirm.
7. Call book_appointment.

=== EXAMPLES OF TOOL TRIGGERS ===
- Specialty searches:
  User: "Who is available in cardiology?" -> Action: Call search_doctors(specialty="Cardiology")
  User: "Need a skin doctor next Monday" -> Action: Call search_doctors(specialty="Dermatology")
  User: "Is there a nerve specialist?" -> Action: Call search_doctors(specialty="Neurology")
  User: "Which doctors handle bone injuries?" -> Action: Call search_doctors(specialty="Orthopedics")
  User: "Who is in primary care?" -> Action: Call search_doctors(specialty="General Medicine")
  User: "What specialties do you have?" -> Action: Call list_all_specialties()

- Availability checks:
  User: "Is Dr. Alice Smith free tomorrow?" -> Action: Call check_doctor_availability(doctor_name="Dr. Alice Smith", date="{tomorrow.strftime('%Y-%m-%d')}")
  User: "When is Dr. David Kim available next Monday?" -> Action: Call check_doctor_availability(doctor_name="Dr. David Kim", date="[Insert next Monday's YYYY-MM-DD]")

- Bookings:
  User: "Book a slot for Dr. Clara Lee on 2026-05-30 at 10:30 AM. My name is Pranab." -> Action: Call book_appointment(patient_name="Pranab", doctor_name="Dr. Clara Lee", date="2026-05-30", time="10:30 AM")

- Cancellations:
  User: "Cancel my appointment APT-F73B90" -> Action: Call cancel_appointment(booking_id="APT-F73B90")

- Policy & FAQ lookup:
  User: "What insurance do you accept?" -> Action: Call lookup_hospital_info(query="insurance accepted")
  User: "What is your address?" -> Action: Call lookup_hospital_info(query="address")
  User: "How much is parking at the hospital?" -> Action: Call lookup_hospital_info(query="parking fees")

- Mathematical calculations:
  User: "What is the cost of a 3-day stay in a private room?" -> Action: Call execute_math_calculation(expression="150 * 3") (where 150 is the room rate from FAQ)

=== RESPONSE STYLE ===
- Warm, concise, one topic at a time.
- Format lists with bullet points.
- Always end with a clear next step/question."""


class FrontierAssistant:
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        if self.api_key:
            genai.configure(api_key=self.api_key)
        else:
            logger.warning("GEMINI_API_KEY is not set. Gemini assistant will operate in Mock mode.")
        
        self.model_name = "gemini-2.5-flash"

    def _get_gemini_tools(self) -> List[Any]:
        tools_declaration = {
            "function_declarations": [
                {
                    "name": "list_all_specialties",
                    "description": "Lists all medical specialties available at Evergreen Medical Center with their doctors. Call this when the patient asks what specialties or doctors are available.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {}
                    }
                },
                {
                    "name": "search_doctors",
                    "description": "Search for doctors by specialty. You can pass informal terms like 'cardio', 'heart', 'neuro', 'brain', 'peds', 'kids', 'bone', 'skin', 'gp' or canonical names like 'Cardiology', 'Neurology', 'Pediatrics', 'Orthopedics', 'Dermatology', 'General Medicine'. Returns doctor names, rooms, phone extensions, and bio.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "specialty": {
                                "type": "STRING",
                                "description": "The medical specialty or informal term to search by (e.g. 'cardio', 'Cardiology')."
                            }
                        },
                        "required": ["specialty"]
                    }
                },
                {
                    "name": "check_doctor_availability",
                    "description": "Check available time slots for a specific doctor on a given date. The date must be in YYYY-MM-DD format.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "doctor_name": {
                                "type": "STRING",
                                "description": "Full name of the doctor (e.g. Dr. Alice Smith)."
                            },
                            "date": {
                                "type": "STRING",
                                "description": "The date in YYYY-MM-DD format (e.g. 2026-05-29)."
                            }
                        },
                        "required": ["doctor_name", "date"]
                    }
                },
                {
                    "name": "book_appointment",
                    "description": "Book a confirmed appointment for a patient. Only call this AFTER the patient has confirmed all details.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "patient_name": {
                                "type": "STRING",
                                "description": "Full name of the patient."
                            },
                            "doctor_name": {
                                "type": "STRING",
                                "description": "Name of the doctor (e.g. Dr. Alice Smith)."
                            },
                            "date": {
                                "type": "STRING",
                                "description": "Date in YYYY-MM-DD format."
                            },
                            "time": {
                                "type": "STRING",
                                "description": "Time slot (e.g. '10:00 AM')."
                            }
                        },
                        "required": ["patient_name", "doctor_name", "date", "time"]
                    }
                },
                {
                    "name": "cancel_appointment",
                    "description": "Cancel an existing appointment by its Booking ID (e.g. APT-XXXXXXXX).",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "booking_id": {
                                "type": "STRING",
                                "description": "The Booking ID of the appointment to cancel (e.g. APT-1A2B3C4D)."
                            }
                        },
                        "required": ["booking_id"]
                    }
                },
                {
                    "name": "lookup_hospital_info",
                    "description": "Look up hospital FAQs including: policies, insurance accepted, room rates, contact details, parking fees, address, pharmacy hours, cafeteria info, and emergency info.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "query": {
                                "type": "STRING",
                                "description": "A specific query about hospital services or policies."
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "get_current_datetime",
                    "description": "Get the current date and time with computed upcoming dates. Use only if you need to verify or recalculate dates.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {}
                    }
                },
                {
                    "name": "execute_math_calculation",
                    "description": "Safely evaluate basic mathematical expressions for calculating prices, stay totals, or other numeric requests.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "expression": {
                                "type": "STRING",
                                "description": "The math expression to compute (e.g. '150 * 3 + 20')."
                            }
                        },
                        "required": ["expression"]
                    }
                }
            ]
        }
        return [tools_declaration]

    async def generate_response(self, messages: List[Dict[str, Any]], query_id: str) -> Tuple[str, List[Dict[str, Any]]]:
        start_time = time.time()
        
        # Build system prompt with current datetime
        system_prompt = _build_system_prompt()
        
        if not self.api_key:
            await asyncio.sleep(0.5)
            mock_text = "Hello! I am Olive, Evergreen Medical Center's AI Receptionist (Mock Mode). Please set your GEMINI_API_KEY to activate."
            trace_repo.add_step(query_id, "LLMNode", int((time.time() - start_time) * 1000), {"mock": True})
            return mock_text, []

        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_prompt,
                tools=self._get_gemini_tools()
            )

            contents = []
            for msg in messages:
                role = "user" if msg["role"] in ["user", "tool"] else "model"
                parts = []
                
                if msg["role"] == "tool":
                    parts.append({
                        "function_response": {
                            "name": msg["name"],
                            "response": {"result": msg["content"]}
                        }
                    })
                elif msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        parts.append({
                            "function_call": {
                                "name": tc["name"],
                                "args": tc["arguments"]
                            }
                        })
                    if msg.get("content"):
                        parts.append(msg["content"])
                else:
                    parts.append(msg["content"])
                    
                contents.append({
                    "role": role,
                    "parts": parts
                })

            response: GenerateContentResponse = await model.generate_content_async(contents)
            latency_ms = int((time.time() - start_time) * 1000)
            
            prompt_tokens = 0
            completion_tokens = 0
            cost = 0.0
            
            if response.usage_metadata:
                prompt_tokens = response.usage_metadata.prompt_token_count
                completion_tokens = response.usage_metadata.candidates_token_count
                cost = ((prompt_tokens / 1_000_000) * COST_INPUT_PER_1M) + ((completion_tokens / 1_000_000) * COST_OUTPUT_PER_1M)
                
            trace_repo.update_metrics(
                query_id,
                time_to_first_token_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                estimated_cost_usd=cost
            )
            
            tool_calls = []
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.function_call:
                        call = part.function_call
                        args = {key: val for key, val in call.args.items()}
                        tool_calls.append({
                            "name": call.name,
                            "arguments": args
                        })
                        
            # Check parts individually to compile text, avoiding calling response.text
            # which raises ValueError if a part contains a function_call instead of text.
            text_response = ""
            if response.candidates and response.candidates[0].content.parts:
                text_parts = []
                for part in response.candidates[0].content.parts:
                    if part.text:
                        text_parts.append(part.text)
                text_response = "".join(text_parts)
            
            trace_repo.add_step(
                query_id, 
                "LLMNode", 
                latency_ms, 
                {
                    "prompt_tokens": prompt_tokens, 
                    "completion_tokens": completion_tokens,
                    "tool_calls_requested": [tc["name"] for tc in tool_calls],
                    "raw_response": json.dumps({"tool_calls": tool_calls}) if tool_calls else text_response
                }
            )
            
            return text_response, tool_calls

        except Exception as e:
            logger.error(f"Error in Gemini API call: {e}")
            trace_repo.add_step(query_id, "LLMNode", int((time.time() - start_time) * 1000), {"error": str(e)})
            raise e
