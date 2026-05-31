import re
import base64
import asyncio
import unicodedata
import urllib.parse
import codecs
import time
from collections import defaultdict, deque
from typing import Tuple

# --- INPUT GUARDRAIL PATTERNS ---

EMERGENCY_PATTERNS = [
    r"\bchest\s+pain\b",
    r"\bdifficulty\s+breathing\b",
    r"\bcan'?t\s+breathe\b",
    r"\bstroke\b",
    r"\bheart\s+attack\b",
    r"\bunconscious\b",
    r"\bsevere\s+bleeding\b",
    r"\bsuicidal\b",
    r"\bsuicide\b",
    r"\bchoking\b",
    r"\banaphylaxis\b",
    r"\boverdos(e|ing)\b",
    r"\bnot\s+breathing\b",
    r"\bseizure\b",
    r"\bpassing\s+out\b",
    r"\bblood\s+everywhere\b",
    r"\bcan'?t\s+move\b",
    r"\bamputation\b",
    r"\bpoisoning\b",
    r"\bdiabetic\s+(shock|coma)\b",
    r"\ballergic\s+reaction\b",
    r"\blosing\s+consciousness\b"
]

ROLE_OVERRIDE_PATTERNS = [
    r"you are now",
    r"act as",
    r"pretend to be",
    r"roleplay as",
    r"ignore your role",
    r"stop being a receptionist",
    r"switch to doctor mode"
]

PROMPT_INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"override instructions",
    r"system prompt",
    r"developer message",
    r"hidden instructions",
    r"reveal prompt",
    r"system instructions",
    r"jailbreak"
]

OUT_OF_SCOPE_PATTERNS = [
    r"write\s+a\s+python\s+script",
    r"write\s+code",
    r"programming\b",
    r"\bjavascript\b",
    r"\bhtml\b",
    r"how\s+do\s+i\s+hack",
    r"hack\s+a\s+server",
    r"generate\s+malware",
    r"ddos\b",
    r"\bexploit\b",
    r"network\s+intrusion",
    r"reverse\s+shell"
]

# --- OUTPUT GUARDRAIL PATTERNS ---

MEDICAL_INFERENCE_PATTERNS = [
    r"sounds like",
    r"appears to be",
    r"likely have",
    r"probably have",
    r"consistent with",
    r"indicates that",
    r"you may be suffering from"
]

TREATMENT_PATTERNS = [
    r"treatment plan",
    r"recommended treatment",
    r"you should take",
    r"start taking"
]

DOSAGE_PATTERNS = [
    r"\b\d+\s?mg\b",
    r"\b\d+\s?ml\b",
    r"\b\d+\s?tablet",
    r"\bonce daily\b",
    r"\btwice daily\b",
    r"\bevery \d+ hours\b"
]

MEDICATION_PATTERNS = [
    r"\bacetaminophen\b",
    r"\bibuprofen\b",
    r"\baspirin\b",
    r"\bparacetamol\b",
    r"\bxanax\b",
    r"\bpenicillin\b",
    r"\bantibiotic(s)?\b",
    r"\bmedication(s)?\b",
    r"\bprescribe(d|s)?\b",
    r"\bprescription(s)?\b"
]

LEAKAGE_PATTERNS = [
    r"system prompt",
    r"developer instruction",
    r"hidden instruction",
    r"internal workflow",
    r"chain of thought"
]

SECRET_PATTERNS = [
    r"sk-[A-Za-z0-9]{20,}",
    r"AKIA[A-Z0-9]{16}",
]

OFFENSIVE_PATTERNS = [
    r"\bnigger\b", r"\bkike\b", r"\bchink\b", r"\bspic\b", r"\bgook\b", r"\bfaggot\b", r"\bdyke\b", r"\bretard\b"
]

SAFE_MEDICAL_CONTEXT_PATTERNS = [
    r"(cannot|unable to|not able to)\s.{0,60}(diagnose|prescribe|recommend|advise)",
    r"(consult|please\s+see|recommend\s+seeing)\s+a\s+(doctor|physician|specialist|professional)",
    r"(schedule|book)\s+(an\s+)?(appointment|consultation)",
    r"i('m|\s+am)\s+not\s+(a\s+)?(doctor|medical)",
]

# --- SEVERITY RISK MATRIX ---

RISK_WEIGHTS = {
    "role_override":    {"score": 6, "tier": "CRITICAL"},
    "prompt_injection": {"score": 5, "tier": "CRITICAL"},
    "pii_leak":         {"score": 5, "tier": "HIGH"},
    "out_of_scope":     {"score": 3, "tier": "MEDIUM"},
    "encoded_text":     {"score": 2, "tier": "LOW"},
}


def normalize_text(text: str) -> str:
    """
    Normalizes text to defeat spacing obfuscation, Unicode homoglyphs, and leet-speak.
    """
    # Normalize unicode (e.g. ａｃｔ ａｓ -> act as)
    text = unicodedata.normalize("NFKC", text)
    # Collapse whitespace injected between characters (y o u -> you)
    text = re.sub(r'(\w)\s+(\w)', r'\1\2', text)
    # Strip common leet-speak substitutions
    leet_map = str.maketrans("013456789@$", "oieashgtbas")
    return text.lower().translate(leet_map)


def decode_text(text: str) -> str:
    """
    Decodes potential Base64, Hex, URL-encoded, and ROT13 representations in the text for scanning.
    Supports recursive scanning up to 2 levels deep.
    """
    decoded_chunks = []
    
    # 1. Base64 Scan
    b64_matches = re.findall(r'\b[A-Za-z0-9+/]{8,}={0,2}\b', text)
    for word in b64_matches:
        try:
            padded = word + "=" * ((4 - len(word) % 4) % 4)
            decoded = base64.b64decode(padded).decode('utf-8', errors='ignore')
            if len(decoded) > 3 and any(c.isalpha() for c in decoded):
                decoded_chunks.append(decoded.lower())
        except Exception:
            pass
            
    # 2. Hex Scan
    hex_matches = re.findall(r'\b(?:[0-9a-fA-F]{2}[- :]*){3,}\b', text)
    for match in hex_matches:
        try:
            cleaned = re.sub(r'[^0-9a-fA-F]', '', match)
            decoded = bytes.fromhex(cleaned).decode('utf-8', errors='ignore')
            if len(decoded) > 3 and any(c.isalpha() for c in decoded):
                decoded_chunks.append(decoded.lower())
        except Exception:
            pass

    # 3. URL Decoding
    try:
        url_decoded = urllib.parse.unquote(text)
        if url_decoded != text:
            decoded_chunks.append(url_decoded.lower())
    except Exception:
        pass

    # 4. ROT13
    try:
        rot13 = codecs.decode(text, 'rot_13')
        if rot13 != text:
            decoded_chunks.append(rot13.lower())
    except Exception:
        pass

    # 5. Recursive pass (catches double-encoded payloads, one level deep)
    for chunk in list(decoded_chunks):
        second_pass_chunks = []
        b64_matches_recur = re.findall(r'\b[A-Za-z0-9+/]{8,}={0,2}\b', chunk)
        for word in b64_matches_recur:
            try:
                padded = word + "=" * ((4 - len(word) % 4) % 4)
                decoded = base64.b64decode(padded).decode('utf-8', errors='ignore')
                if len(decoded) > 3 and any(c.isalpha() for c in decoded):
                    second_pass_chunks.append(decoded.lower())
            except Exception:
                pass
        decoded_chunks.extend(second_pass_chunks)
            
    return " | ".join(decoded_chunks)


def is_safe_medical_response(response: str) -> bool:
    """
    Returns True only if medical terminology appears strictly inside a
    genuine refusal or redirect context — not just anywhere in the response.
    """
    response_lower = response.lower()
    return any(re.search(p, response_lower) for p in SAFE_MEDICAL_CONTEXT_PATTERNS)


class SessionGuardrail:
    def __init__(self, window_sec: int = 60, max_flags: int = 3):
        self.window = window_sec
        self.max_flags = max_flags
        self._flags = defaultdict(deque)

    def record_flag(self, session_id: str) -> bool:
        """
        Records a guardrail flag for the session.
        Returns True if the session should now be blocked.
        """
        now = time.time()
        q = self._flags[session_id]
        while q and q[0] < now - self.window:
            q.popleft()
        q.append(now)
        return len(q) >= self.max_flags

    def is_blocked(self, session_id: str) -> bool:
        """
        Checks if the session is currently rate-limited due to repeated flags.
        """
        now = time.time()
        q = self._flags[session_id]
        while q and q[0] < now - self.window:
            q.popleft()
        return len(q) >= self.max_flags


# Singleton instance for session-level safety rates
session_guardrail = SessionGuardrail(window_sec=60, max_flags=3)


class GuardrailsService:
    @staticmethod
    async def verify_input(query: str) -> Tuple[bool, str]:
        """
        Verify the input prompt using a normalized risk-scoring safety architecture.
        Returns: (is_safe: bool, reason: str)
        """
        await asyncio.sleep(0.05)
        
        query_clean = query.strip()
        query_lower = normalize_text(query_clean)
        
        # 1. Check for Emergency (High Priority Escalation)
        for pattern in EMERGENCY_PATTERNS:
            if re.search(pattern, query_lower):
                return False, "This may be a medical emergency. Please contact emergency services immediately or visit the nearest emergency department."
                
        # 2. Decode hidden encoded payloads (Base64/Hex/URL/ROT13) and scan them
        decoded = decode_text(query_clean)
        scan_text = query_lower
        if decoded:
            scan_text += " | decoded: " + decoded
            
        risk_score = 0
        reasons = []
        
        # Check Role Override attempts
        for pattern in ROLE_OVERRIDE_PATTERNS:
            if re.search(pattern, scan_text):
                risk_score += RISK_WEIGHTS["role_override"]["score"]
                reasons.append("role_override")
                break
                
        # Check Prompt Injection attempts
        for pattern in PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, scan_text):
                risk_score += RISK_WEIGHTS["prompt_injection"]["score"]
                reasons.append("prompt_injection")
                break
                
        # Check Out-of-Scope content
        for pattern in OUT_OF_SCOPE_PATTERNS:
            if re.search(pattern, scan_text):
                risk_score += RISK_WEIGHTS["out_of_scope"]["score"]
                reasons.append("out_of_scope")
                break
                
        # Check PII (Credit Cards / SSN)
        cc_pattern = r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"
        ssn_pattern = r"\b\d{3}-\d{2}-\d{4}\b"
        if re.search(cc_pattern, query_clean) or re.search(ssn_pattern, query_clean):
            risk_score += RISK_WEIGHTS["pii_leak"]["score"]
            reasons.append("pii_leak")
            
        if decoded:
            risk_score += RISK_WEIGHTS["encoded_text"]["score"]
            reasons.append("encoded_text")
            
        # Decision Logic based on Risk Score and Severity Tiers
        if any(RISK_WEIGHTS.get(r, {}).get("tier") == "CRITICAL" for r in reasons):
            return False, f"Flagged Input: Critical policy violation detected ({', '.join(reasons)})."
            
        if risk_score >= 5:
            return False, f"Flagged Input: Policy violation detected ({', '.join(reasons)})."
            
        return True, "Passed: Input is safe."

    @staticmethod
    async def verify_output(response: str) -> Tuple[bool, str]:
        """
        Verify the assistant response for safety.
        Ensure it does not attempt to diagnose symptoms, recommend treatments,
        disclose developer prompts, reveal keys, or output inappropriate language.
        Returns: (is_safe: bool, reason: str)
        """
        await asyncio.sleep(0.05)
        
        response_lower = response.lower()
        
        # 1. Check for Offensive Language / Racial Slurs
        for pattern in OFFENSIVE_PATTERNS:
            if re.search(pattern, response_lower):
                return False, "Flagged Output: Inappropriate language detected."
                
        # 2. Check for Secret Leakage
        for pattern in SECRET_PATTERNS:
            if re.search(pattern, response):
                return False, "Flagged Output: Credential or security token detected."
                
        # 3. Check for Prompt Leakage
        for pattern in LEAKAGE_PATTERNS:
            if re.search(pattern, response_lower):
                return False, "Flagged Output: Developer instructions leak detected."
                
        # 4. Check for Medical Advice (Diagnosis, Treatment, Dosage, Medications)
        medical_patterns = MEDICAL_INFERENCE_PATTERNS + TREATMENT_PATTERNS + DOSAGE_PATTERNS + MEDICATION_PATTERNS
        has_medical_terms = any(re.search(pattern, response_lower) for pattern in medical_patterns)
        
        if has_medical_terms:
            if not is_safe_medical_response(response):
                return False, "Flagged Output: Response appears to contain medical diagnosis, treatment recommendations, or dosage advice."
                    
        return True, "Passed: Output is safe."
