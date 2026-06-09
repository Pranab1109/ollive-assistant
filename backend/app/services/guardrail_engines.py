"""Three interchangeable input-guardrail engines, selected by GUARDRAIL_MODE."""
import os
import re
import time
from typing import Tuple, Dict
from functools import lru_cache

from backend.app.config import settings
# Regex pieces we KEEP in every mode (emergency + PII + decoding live in guardrails.py)
from backend.app.services.guardrails import (
    GuardrailsService, normalize_text, collapse_spaced_text,
    decode_text, EMERGENCY_PATTERNS,
)

CRITICAL = {"prompt_injection", "role_override"}     # block immediately
HIGH     = {"out_of_scope"}                           # block
ROUTE    = {"sensitive_bias"}                          # don't block — flag for FAQ routing

def _emergency_or_pii(query: str):
    ql = normalize_text(query)
    qc = collapse_spaced_text(ql)
    if any(re.search(p, ql) or re.search(p, qc) for p in EMERGENCY_PATTERNS):
        return False, "This may be a medical emergency. Please contact emergency services immediately."
    cc  = r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"
    ssn = r"\b\d{3}-\d{2}-\d{4}\b"
    if re.search(cc, query) or re.search(ssn, query):
        return False, "Flagged Input: PII detected (pii_leak)."
    return None  # not decided here


class RegexEngine:
    """Arm A: delegate fully to the existing service."""
    name = "regex"
    async def verify_input(self, query: str) -> Tuple[bool, str]:
        return await GuardrailsService.verify_input(query)


class SingleSetFitEngine:
    """Arm B: emergency/PII regex first, then one multiclass SetFit model."""
    name = "setfit_single"
    def __init__(self, model_dir="models/guardrail_single", threshold=0.5):
        self.model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", model_dir))
        self.threshold = threshold
        
        local_exists = os.path.exists(self.model_dir) and any(os.path.isfile(os.path.join(self.model_dir, f)) for f in os.listdir(self.model_dir))
        if local_exists:
            from setfit import SetFitModel
            self.model = SetFitModel.from_pretrained(self.model_dir)
            self.active = True
            print(f"SingleSetFitEngine loaded successfully from local path {self.model_dir}")
        else:
            hf_repo = f"{settings.HF_USERNAME}/ollive-guardrail-single"
            print(f"Local SetFit single model directory '{self.model_dir}' not found. Loading from Hugging Face Hub: '{hf_repo}'...")
            try:
                from setfit import SetFitModel
                self.model = SetFitModel.from_pretrained(hf_repo)
                self.active = True
                print(f"SingleSetFitEngine loaded successfully from Hugging Face Hub: {hf_repo}")
            except Exception as e:
                print(f"Error loading Single SetFit model from Hugging Face Hub ({hf_repo}): {e}. Falling back to RegexEngine.")
                self.active = False
                self.regex_fallback = RegexEngine()

    async def verify_input(self, query: str) -> Tuple[bool, str]:
        if not self.active:
            return await self.regex_fallback.verify_input(query)
            
        pre = _emergency_or_pii(query)
        if pre is not None:
            return pre
            
        # feed decoded payloads in too (mirrors regex preprocessor)
        decoded = decode_text(query)
        text = query if not decoded else f"{query} {decoded}"
        
        probs = self.model.predict_proba([text])[0]
        labels = self.model.labels
        scored = dict(zip(labels, [float(p) for p in probs]))
        top = max(scored, key=scored.get)
        
        if top in (CRITICAL | HIGH) and scored[top] >= self.threshold:
            return False, f"Flagged Input: {top} (p={scored[top]:.2f})."
        if top in ROUTE and scored[top] >= self.threshold:
            # not blocked: surface as a soft flag so the node can route to FAQ
            return True, f"sensitive_topic:{top}"
            
        return True, "Passed: Input is safe."


class EnsembleSetFitEngine:
    """Arm C: emergency/PII regex first, then N binary SetFit models."""
    name = "setfit_ensemble"
    THRESHOLDS = {
        "prompt_injection": 0.5, 
        "role_override": 0.5,
        "out_of_scope": 0.5, 
        "sensitive_bias": 0.4
    }
    
    def __init__(self, root="models/guardrail_ensemble"):
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", root))
        self.active = True
        self.models = {}
        
        # Check if all sub-models exist
        for cls in self.THRESHOLDS:
            cls_path = os.path.join(self.root_dir, cls)
            local_exists = os.path.exists(cls_path) and any(os.path.isfile(os.path.join(cls_path, f)) for f in os.listdir(cls_path))
            
            if local_exists:
                from setfit import SetFitModel
                self.models[cls] = SetFitModel.from_pretrained(cls_path)
            else:
                hf_repo = f"{settings.HF_USERNAME}/ollive-guardrail-ensemble-{cls}"
                print(f"Local model for '{cls}' not found. Loading from Hugging Face Hub: '{hf_repo}'...")
                try:
                    from setfit import SetFitModel
                    self.models[cls] = SetFitModel.from_pretrained(hf_repo)
                except Exception as e:
                    print(f"Error loading ensemble model '{cls}' from Hugging Face Hub ({hf_repo}): {e}")
                    self.active = False
                    break
                
        if self.active:
            print(f"EnsembleSetFitEngine loaded successfully.")
        else:
            print(f"Warning: SetFit ensemble models not found locally or on HF Hub. Falling back to RegexEngine.")
            self.regex_fallback = RegexEngine()

    async def verify_input(self, query: str) -> Tuple[bool, str]:
        if not self.active:
            return await self.regex_fallback.verify_input(query)
            
        pre = _emergency_or_pii(query)
        if pre is not None:
            return pre
            
        decoded = decode_text(query)
        text = query if not decoded else f"{query} {decoded}"
        
        fired = {}
        for cls, m in self.models.items():
            # Get P(positive)
            p = float(m.predict_proba([text])[0][1])
            if p >= self.THRESHOLDS[cls]:
                fired[cls] = p
                
        if not fired:
            return True, "Passed: Input is safe."
            
        top = max(fired, key=fired.get)
        if top in (CRITICAL | HIGH):
            return False, f"Flagged Input: {top} (p={fired[top]:.2f})."
            
        return True, f"sensitive_topic:{top}"   # bias -> route, don't block


@lru_cache(maxsize=1)
def get_engine():
    mode = settings.GUARDRAIL_MODE
    if mode == "setfit_single":
        return SingleSetFitEngine()
    if mode == "setfit_ensemble":
        return EnsembleSetFitEngine()
    return RegexEngine()
