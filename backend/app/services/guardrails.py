import re
import base64
import asyncio
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
    r"\banaphylaxis\b"
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


def decode_text(text: str) -> str:
    """
    Decodes potential Base64 and Hex representations in the text for scanning.
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
            
    return " | ".join(decoded_chunks)


class GuardrailsService:
    @staticmethod
    async def verify_input(query: str) -> Tuple[bool, str]:
        """
        Verify the input prompt using a risk scoring safety architecture.
        Returns: (is_safe: bool, reason: str)
        """
        await asyncio.sleep(0.05)
        
        query_clean = query.strip()
        query_lower = query_clean.lower()
        
        # 1. Check for Emergency (High Priority Escalation)
        for pattern in EMERGENCY_PATTERNS:
            if re.search(pattern, query_lower):
                return False, "This may be a medical emergency. Please contact emergency services immediately or visit the nearest emergency department."
                
        # 2. Decode hidden encoded payloads (Base64/Hex) and combine with main search text
        decoded = decode_text(query_clean)
        scan_text = query_lower
        if decoded:
            scan_text += " | decoded: " + decoded
            
        risk_score = 0
        reasons = []
        
        # Check Role Override attempts
        for pattern in ROLE_OVERRIDE_PATTERNS:
            if re.search(pattern, scan_text):
                risk_score += 6
                reasons.append("role_override")
                break
                
        # Check Prompt Injection attempts
        for pattern in PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, scan_text):
                risk_score += 3
                reasons.append("prompt_injection")
                break
                
        # Check Out-of-Scope content
        for pattern in OUT_OF_SCOPE_PATTERNS:
            if re.search(pattern, scan_text):
                risk_score += 4
                reasons.append("out_of_scope")
                break
                
        # Check PII (Credit Cards / SSN)
        cc_pattern = r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"
        ssn_pattern = r"\b\d{3}-\d{2}-\d{4}\b"
        if re.search(cc_pattern, query_clean) or re.search(ssn_pattern, query_clean):
            risk_score += 5
            reasons.append("pii_leak")
            
        if decoded:
            risk_score += 2
            reasons.append("encoded_text")
            
        # Decision Logic based on Risk Score
        if risk_score >= 6:
            return False, f"Flagged Input: Policy violation detected ({', '.join(reasons)})."
        elif risk_score >= 4:
            # Out of scope / suspicious requests get escalated as a refusal
            return False, f"Out-of-scope query request concerning {reasons[0]}."
            
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
        for pattern in medical_patterns:
            if re.search(pattern, response_lower):
                # Standard receptionist disclaimers are allowed
                refusals = ["cannot", "not able to", "sorry", "unable", "should see a doctor", "consult", "appointment", "not authorized", "do not provide"]
                is_refusal = any(ref in response_lower for ref in refusals)
                if not is_refusal:
                    return False, "Flagged Output: Response appears to contain medical diagnosis, treatment recommendations, or dosage advice."
                    
        return True, "Passed: Output is safe."
