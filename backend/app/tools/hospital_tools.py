import os
import json
import re
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# File paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
APPOINTMENTS_FILE = os.path.join(DATA_DIR, "appointments.json")
DOCTORS_DB_FILE = os.path.join(DATA_DIR, "doctors_db.json")

# --- DATA LOADERS ---

def _load_doctors_db() -> Dict[str, Any]:
    """Load the full hospital database from doctors_db.json."""
    try:
        with open(DOCTORS_DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"doctors": [], "hospital_faq": {}}

def _get_doctors() -> List[Dict[str, Any]]:
    """Get the list of doctors from the database."""
    db = _load_doctors_db()
    return db.get("doctors", [])

def _get_hospital_faq() -> Dict[str, str]:
    """Get hospital FAQ knowledge base."""
    db = _load_doctors_db()
    return db.get("hospital_faq", {})

def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(APPOINTMENTS_FILE):
        with open(APPOINTMENTS_FILE, "w") as f:
            json.dump([], f)

def _read_appointments() -> List[Dict[str, Any]]:
    _ensure_data_dir()
    try:
        with open(APPOINTMENTS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def _write_appointments(appointments: List[Dict[str, Any]]):
    _ensure_data_dir()
    with open(APPOINTMENTS_FILE, "w") as f:
        json.dump(appointments, f, indent=2)

def _get_day_of_week(date_str: str) -> str:
    """Convert YYYY-MM-DD to day name (e.g., 'Monday')."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%A")
    except ValueError:
        return ""

# Specialty alias map — maps informal terms → canonical specialty names
SPECIALTY_ALIASES = {
    "cardio": "Cardiology",
    "cardiology": "Cardiology",
    "heart": "Cardiology",
    "cardiac": "Cardiology",
    "neuro": "Neurology",
    "neurology": "Neurology",
    "brain": "Neurology",
    "nerve": "Neurology",
    "peds": "Pediatrics",
    "pediatrics": "Pediatrics",
    "pediatric": "Pediatrics",
    "children": "Pediatrics",
    "child": "Pediatrics",
    "kids": "Pediatrics",
    "ortho": "Orthopedics",
    "orthopedics": "Orthopedics",
    "orthopedic": "Orthopedics",
    "bone": "Orthopedics",
    "joint": "Orthopedics",
    "sports": "Orthopedics",
    "derm": "Dermatology",
    "dermatology": "Dermatology",
    "dermatologist": "Dermatology",
    "skin": "Dermatology",
    "general": "General Medicine",
    "general medicine": "General Medicine",
    "gp": "General Medicine",
    "internal medicine": "General Medicine",
    "primary care": "General Medicine",
    "family": "General Medicine",
}

def _normalize_specialty(specialty: str) -> str:
    """Normalize a specialty string to the canonical name using aliases."""
    normalized = specialty.strip().lower()
    if normalized in SPECIALTY_ALIASES:
        return SPECIALTY_ALIASES[normalized]
    # Try partial match — e.g. "neurolog" → "Neurology"
    for alias, canonical in SPECIALTY_ALIASES.items():
        if alias in normalized or normalized in alias:
            return canonical
    # Return title-cased version as fallback (handles "cardiology" → "Cardiology")
    return specialty.strip().title()

# --- TOOL 1: List All Specialties ---

async def list_all_specialties() -> str:
    """
    Lists all medical specialties available at Evergreen Medical Center,
    along with the doctors in each specialty.
    """
    await asyncio.sleep(0.05)
    doctors = _get_doctors()
    specialty_map = {}
    for doc in doctors:
        spec = doc["specialty"]
        if spec not in specialty_map:
            specialty_map[spec] = []
        specialty_map[spec].append(doc["name"])
    
    if not specialty_map:
        return "No specialties found in the database."
    
    result = "Available specialties at Evergreen Medical Center:\n"
    for spec, doc_names in specialty_map.items():
        result += f"- {spec}: {', '.join(doc_names)}\n"
    return result

# --- TOOL 2: Search Doctors ---

async def search_doctors(specialty: str) -> str:
    """
    Search for doctors by their specialty.
    Returns a list of matching doctors with their rooms, phone extensions, brief bio,
    and their availability (open slots) for the next 3 days.
    """
    await asyncio.sleep(0.1)
    
    # Normalize the specialty using alias map
    canonical_specialty = _normalize_specialty(specialty)
    
    doctors = _get_doctors()
    matches = [d for d in doctors if d["specialty"].lower() == canonical_specialty.lower()]
    
    if not matches:
        all_specialties = list(set(d["specialty"] for d in doctors))
        return (
            f"No doctors found for '{specialty}' (interpreted as '{canonical_specialty}'). "
            f"Available specialties are: {', '.join(all_specialties)}."
        )
    
    # Single-pass: build a lookup dict (doctor_lower, date) -> set(booked_times)
    # so each doctor×date check is O(1) instead of scanning every appointment
    appointments = _read_appointments()
    booked_map = {}
    for app in appointments:
        if app.get("status", "confirmed") == "confirmed":
            key = (app["doctor"].lower(), app["date"])
            booked_map.setdefault(key, set()).add(app["time"])
    
    today = datetime.now()
    # Pre-compute the 3 upcoming dates once
    upcoming_dates = []
    for offset in range(3):
        future = today + timedelta(days=offset)
        upcoming_dates.append((future.strftime("%Y-%m-%d"), future.strftime("%A")))
    
    result = f"Doctors available for {canonical_specialty}:\n"
    for doc in matches:
        result += (
            f"\n- {doc['name']} | {doc['room']} | Ext: {doc['phone_ext']}\n"
            f"  Bio: {doc['bio']}\n"
            f"  Insurance accepted: {', '.join(doc['accepted_insurance'])}\n"
        )
        # Add availability for the next 3 days
        result += "  Upcoming availability (next 3 days):\n"
        schedule = doc.get("weekly_schedule", {})
        doc_lower = doc["name"].lower()
        for date_str, day_name in upcoming_dates:
            day_slots = schedule.get(day_name, [])
            if not day_slots:
                result += f"    {date_str} ({day_name}): No office hours\n"
            else:
                booked = booked_map.get((doc_lower, date_str), set())
                open_slots = [s for s in day_slots if s not in booked]
                if open_slots:
                    result += f"    {date_str} ({day_name}): {', '.join(open_slots)}\n"
                else:
                    result += f"    {date_str} ({day_name}): Fully booked\n"
    return result

# --- TOOL 3: Check Doctor Availability ---

async def check_doctor_availability(doctor_name: str, date: str) -> str:
    """
    Checks the available appointment slots for a specific doctor on a given date (YYYY-MM-DD).
    Cross-references the doctor's weekly schedule with already-booked appointments.
    """
    await asyncio.sleep(0.1)
    doctors = _get_doctors()
    doc = next((d for d in doctors if d["name"].lower() == doctor_name.strip().lower()), None)
    
    if not doc:
        return f"Doctor '{doctor_name}' not found. Please use search_doctors to find the correct name."
    
    # Determine day of week for the requested date
    day_name = _get_day_of_week(date)
    if not day_name:
        return f"Invalid date format: '{date}'. Please use YYYY-MM-DD format."
    
    # Get schedule for that day
    schedule = doc.get("weekly_schedule", {})
    day_slots = schedule.get(day_name, [])
    
    if not day_slots:
        return (
            f"{doc['name']} does not have office hours on {day_name}s ({date}). "
            f"Their working days are: {', '.join(d for d, slots in schedule.items() if slots)}."
        )
    
    # Check already-booked slots
    appointments = _read_appointments()
    booked_slots = [
        app["time"] for app in appointments
        if app["doctor"].lower() == doc["name"].lower() 
        and app["date"] == date 
        and app.get("status", "confirmed") == "confirmed"
    ]
    
    available = [slot for slot in day_slots if slot not in booked_slots]
    
    if not available:
        return f"{doc['name']} has no available slots left on {date} ({day_name}). All slots are booked."
    
    return (
        f"{doc['name']} availability on {date} ({day_name}):\n"
        f"Open slots: {', '.join(available)}\n"
        f"Location: {doc['room']}"
    )

# --- TOOL 4: Book Appointment ---

async def book_appointment(patient_name: str, doctor_name: str, date: str, time: str) -> str:
    """
    Books an appointment for a patient with a doctor on a specific date (YYYY-MM-DD) and time slot.
    Validates the doctor exists, the time is a valid slot for that day, and the slot is not already taken.
    """
    await asyncio.sleep(0.2)
    doctors = _get_doctors()
    doc = next((d for d in doctors if d["name"].lower() == doctor_name.strip().lower()), None)
    
    if not doc:
        return f"Error: Doctor '{doctor_name}' not found in our system."
    
    # Validate date format
    day_name = _get_day_of_week(date)
    if not day_name:
        return f"Error: Invalid date format '{date}'. Please use YYYY-MM-DD format."
    
    # Check if the time is a valid slot for that day
    schedule = doc.get("weekly_schedule", {})
    day_slots = schedule.get(day_name, [])
    
    if not day_slots:
        return (
            f"Error: {doc['name']} does not have office hours on {day_name}s ({date}). "
            f"Working days: {', '.join(d for d, slots in schedule.items() if slots)}."
        )
    
    if time not in day_slots:
        return (
            f"Error: '{time}' is not a valid slot for {doc['name']} on {day_name}s. "
            f"Available slots on {day_name}s are: {', '.join(day_slots)}."
        )
    
    # Check if already booked
    appointments = _read_appointments()
    is_booked = any(
        app for app in appointments
        if app["doctor"].lower() == doc["name"].lower() 
        and app["date"] == date 
        and app["time"] == time
        and app.get("status", "confirmed") == "confirmed"
    )
    
    if is_booked:
        return f"Error: {doc['name']} is already booked at {time} on {date}. Please choose a different slot."
    
    # Create booking
    booking_id = f"APT-{str(uuid.uuid4())[:8].upper()}"
    new_booking = {
        "id": booking_id,
        "patient": patient_name,
        "doctor": doc["name"],
        "specialty": doc["specialty"],
        "room": doc["room"],
        "date": date,
        "time": time,
        "status": "confirmed",
        "booked_at": datetime.now().isoformat()
    }
    appointments.append(new_booking)
    _write_appointments(appointments)
    
    return (
        f"Appointment confirmed successfully!\n"
        f"- Booking ID: {booking_id}\n"
        f"- Patient: {patient_name}\n"
        f"- Doctor: {doc['name']} ({doc['specialty']})\n"
        f"- Location: {doc['room']}\n"
        f"- Date: {date} ({day_name})\n"
        f"- Time: {time}\n"
        f"Please arrive 15 minutes early with your ID and insurance card."
    )

# --- TOOL 5: Cancel Appointment ---

async def cancel_appointment(booking_id: str) -> str:
    """
    Cancels an existing appointment by its Booking ID (e.g., APT-XXXXXXXX).
    """
    await asyncio.sleep(0.1)
    appointments = _read_appointments()
    
    found = False
    for app in appointments:
        if app.get("id", "").upper() == booking_id.upper() and app.get("status") == "confirmed":
            app["status"] = "cancelled"
            app["cancelled_at"] = datetime.now().isoformat()
            found = True
            break
    
    if not found:
        return f"Error: No active appointment found with Booking ID '{booking_id}'. Please verify the ID."
    
    _write_appointments(appointments)
    return f"Appointment {booking_id} has been successfully cancelled."

# --- TOOL 6: Lookup Hospital Info ---

async def lookup_hospital_info(query: str) -> str:
    """
    Look up general hospital policies, location/contact, insurance coverage, room rates, parking fees, and more.
    """
    await asyncio.sleep(0.1)
    faq = _get_hospital_faq()
    q = query.lower()
    
    # Keyword matching with priority ordering
    keyword_map = [
        (["address", "location", "where", "directions", "find us"], "address"),
        (["visit", "hours", "ward", "icu", "visiting"], "visiting_hours"),
        (["insurance", "accept", "aetna", "cigna", "blueshield", "healthfirst", "coverage"], "insurance"),
        (["rate", "room", "cost", "price", "bed", "suite", "private", "ward rate"], "room_rates"),
        (["park", "parking", "fee", "valet", "car"], "parking"),
        (["phone", "contact", "call", "number", "reach"], "contact"),
        (["pharmacy", "prescription", "medicine", "pick up"], "pharmacy"),
        (["emergency", "er", "urgent", "911"], "emergency"),
        (["cafeteria", "food", "meal", "eat", "dining", "diet"], "cafeteria"),
    ]
    
    for keywords, faq_key in keyword_map:
        for kw in keywords:
            if kw in q:
                return faq.get(faq_key, "Information not available.")
    
    # If no specific match, return a comprehensive summary
    result = "Here is a summary of Evergreen Medical Center services:\n"
    for key, value in faq.items():
        label = key.replace("_", " ").title()
        result += f"- {label}: {value}\n"
    result += "\nPlease ask about a specific topic for detailed information."
    return result

# --- TOOL 7: Get Current Datetime ---

async def get_current_datetime() -> str:
    """
    Retrieve the current system date, time, and day of the week.
    Also provides computed dates for 'tomorrow' and 'next Monday' through 'next Sunday'.
    """
    await asyncio.sleep(0.05)
    now = datetime.now()
    tomorrow = now + timedelta(days=1)
    
    # Compute next occurrence of each day
    next_days = {}
    for i in range(1, 8):
        future = now + timedelta(days=i)
        day_name = future.strftime("%A")
        if day_name not in next_days:
            next_days[day_name] = future.strftime("%Y-%m-%d")
    
    result = (
        f"Current Date & Time: {now.strftime('%A, %B %d, %Y %I:%M %p')}\n"
        f"Today (YYYY-MM-DD): {now.strftime('%Y-%m-%d')}\n"
        f"Tomorrow: {tomorrow.strftime('%A, %Y-%m-%d')}\n"
        f"Upcoming dates:\n"
    )
    for day, date_str in next_days.items():
        result += f"  Next {day}: {date_str}\n"
    
    return result

# --- TOOL 8: Execute Math Calculation ---

async def execute_math_calculation(expression: str) -> str:
    """
    Safely execute a basic mathematical calculation (addition, subtraction, multiplication, division, parentheses).
    Example input: "150 * 3 + 20"
    """
    await asyncio.sleep(0.05)
    clean_expr = expression.strip()
    
    # Safe validation: allow only digits, spaces, and basic operators
    if not re.match(r'^[\d+\-*/().\s]+$', clean_expr):
        return "Error: Invalid characters. Only basic math operators (+, -, *, /, parentheses) are permitted."
    
    try:
        result = eval(clean_expr, {"__builtins__": None}, {})
        return f"Calculation Result: {clean_expr} = {result}"
    except Exception as e:
        return f"Error executing calculation: {str(e)}"
