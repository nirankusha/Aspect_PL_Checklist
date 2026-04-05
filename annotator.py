"""
annotator.py
Constructional-lattice few-shot annotator for Polish aspectual constructions.
Supports:
  - LLaMA 3.1-8B-Instruct (local via llama_cpp.server, Q4_K_M)
  - GPT-4o (OpenAI Chat Completions API)

Features annotated (CheckList constructional lattice):
  Construction       : existential | entity | ambiguous
  Aspect             : IPFV | PFV
  NPCase             : Nom | Gen | other
  VerbAgreement      : V-Neut | V-Plur | other
  Quantifier         : <surface form> | null
  LocativePP         : <surface form> | null
  ResultativeXP      : <surface form> | null
  EventPolarity      : appearance | decrease | other
"""

import json
import re
from typing import List, Dict, Optional

from openai import OpenAI
# AutoTokenizer is only needed in the __main__ demo block;
# importing it at module level pulls transformers.auto_factory which
# breaks on some Colab torch/transformers version mismatches.


# ─────────────────────────────────────────────────────────────────────────────
# JSON stream repair  (unchanged from original – handles llama_cpp truncation)
# ─────────────────────────────────────────────────────────────────────────────

def repair_streamed_json_array(text: str) -> str:
    start = text.find("[")
    if start == -1:
        raise ValueError("No JSON array start '[' found in streamed output")
    text = text[start:]
    last_obj = text.rfind("}")
    if last_obj != -1:
        text = text[: last_obj + 1]
    text = text.rstrip().rstrip(",")
    if not text.endswith("]"):
        text += "\n]"
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Shared prompt content
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a linguistic annotator specialising in Polish constructional grammar.

TASK
Annotate each input sentence for the features listed below.
Return ONLY a valid JSON array — no prose, no markdown, no code fences.

FEATURE SYSTEM (CheckList constructional lattice)
  Construction     : "existential"   — V-Neut + Gen NP (appearance/decrease frame)
                   | "entity"        — Nom NP + V-Plur (motion/arrival frame)
                   | "ambiguous"
  Aspect           : "IPFV"  (imperfective — ongoing/unbounded)
                   | "PFV"   (perfective   — completed/bounded)
  NPCase           : "Nom"   (nominative subject)
                   | "Gen"   (genitive, quantified or existential)
                   | "other"
  VerbAgreement    : "V-Neut"  (neuter default, existential trigger)
                   | "V-Plur"  (plural human/entity agreement)
                   | "other"
  Quantifier       : surface form of wiele/wielu/ile/kilka etc., or null
  LocativePP       : directional PP (na miejsce / do miasta …), or null
  ResultativeXP    : resultative / path XP (na jaw / na światło dzienne …), or null
  EventPolarity    : "appearance" | "decrease" | "other"

OUTPUT FORMAT — one object per sentence:
[
  {
    "Sentence":      "<original sentence>",
    "Roles": {
      "Subject":     "<surface subject or null>",
      "Verb":        "<finite verb form>",
      "Object":      "<NP complement or null>"
    },
    "UsageFeatures": {
      "Construction":   "existential | entity | ambiguous",
      "Aspect":         "IPFV | PFV",
      "NPCase":         "Nom | Gen | other",
      "VerbAgreement":  "V-Neut | V-Plur | other",
      "Quantifier":     "<string or null>",
      "LocativePP":     "<string or null>",
      "ResultativeXP":  "<string or null>",
      "EventPolarity":  "appearance | decrease | other"
    }
  }
]

FEW-SHOT EXAMPLES

Input: Przybyło wielu świadków.
Output:
[{"Sentence":"Przybyło wielu świadków.","Roles":{"Subject":null,"Verb":"przybyło","Object":"wielu świadków"},"UsageFeatures":{"Construction":"existential","Aspect":"PFV","NPCase":"Gen","VerbAgreement":"V-Neut","Quantifier":"wielu","LocativePP":null,"ResultativeXP":null,"EventPolarity":"appearance"}}]

Input: Świadkowie przybyli na miejsce.
Output:
[{"Sentence":"Świadkowie przybyli na miejsce.","Roles":{"Subject":"świadkowie","Verb":"przybyli","Object":null},"UsageFeatures":{"Construction":"entity","Aspect":"PFV","NPCase":"Nom","VerbAgreement":"V-Plur","Quantifier":null,"LocativePP":"na miejsce","ResultativeXP":null,"EventPolarity":"appearance"}}]

Input: Na jaw przybywało faktów.
Output:
[{"Sentence":"Na jaw przybywało faktów.","Roles":{"Subject":null,"Verb":"przybywało","Object":"faktów"},"UsageFeatures":{"Construction":"existential","Aspect":"IPFV","NPCase":"Gen","VerbAgreement":"V-Neut","Quantifier":null,"LocativePP":null,"ResultativeXP":"na jaw","EventPolarity":"appearance"}}]

Input: Ubyło wody.
Output:
[{"Sentence":"Ubyło wody.","Roles":{"Subject":null,"Verb":"ubyło","Object":"wody"},"UsageFeatures":{"Construction":"existential","Aspect":"PFV","NPCase":"Gen","VerbAgreement":"V-Neut","Quantifier":null,"LocativePP":null,"ResultativeXP":null,"EventPolarity":"decrease"}}]

Do NOT explain anything. Return only the JSON array.\
"""


def _build_user_message(sentences: List[str]) -> str:
    return (
        "Annotate the following sentences:\n\n"
        + "\n".join(f"- {s}" for s in sentences)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Base annotator
# ─────────────────────────────────────────────────────────────────────────────

class BaseAnnotator:
    """Shared parsing / validation logic."""

    REQUIRED_FEATURE_KEYS = {
        "Construction", "Aspect", "NPCase", "VerbAgreement",
        "Quantifier", "LocativePP", "ResultativeXP", "EventPolarity",
    }

    def _parse_response(self, raw: str) -> List[Dict]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            try:
                repaired = repair_streamed_json_array(raw)
                data = json.loads(repaired)
            except Exception as e:
                print(f"[WARN] JSON repair failed: {e}")
                return []

        if not isinstance(data, list):
            print("[WARN] Response is not a JSON array")
            return []

        cleaned = []
        for item in data:
            if not isinstance(item, dict):
                continue
            if not all(k in item for k in ("Sentence", "Roles", "UsageFeatures")):
                continue
            uf = item.get("UsageFeatures", {})
            if not self.REQUIRED_FEATURE_KEYS.issubset(uf.keys()):
                missing = self.REQUIRED_FEATURE_KEYS - uf.keys()
                print(f"[WARN] Missing features {missing} in: {item.get('Sentence')}")
            cleaned.append(item)
        return cleaned

    def annotate(self, sentences: List[str]) -> List[Dict]:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# LLaMA annotator  (streaming, local llama_cpp.server)
# ─────────────────────────────────────────────────────────────────────────────

class LlamaAnnotator(BaseAnnotator):
    """
    Uses joshnader/Meta-Llama-3.1-8B-Instruct-Q4_K_M-GGUF via llama_cpp.server.
    Tokenizer: NousResearch/Meta-Llama-3-8B-Instruct
    """

    def __init__(
        self,
        tokenizer=None,          # kept for API compatibility; not used internally
        base_url: str = "http://localhost:9001/v1",
        model_name: str = "local-model",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ):
        self.tokenizer   = tokenizer  # reserved for future token-budget checks
        self.model_name  = model_name
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self._client = OpenAI(base_url=base_url, api_key="not-needed")

    def annotate(self, sentences: List[str]) -> List[Dict]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_message(sentences)},
        ]

        completion = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        buffer, started = "", False
        for chunk in completion:
            delta = chunk.choices[0].delta.content
            if not delta:
                continue
            if not started and "[" in delta:
                started = True
            if started:
                buffer += delta

        print(f"[LLaMA] Raw preview: {buffer[:300]}")
        return self._parse_response(buffer)


# ─────────────────────────────────────────────────────────────────────────────
# GPT-4o annotator  (standard OpenAI Chat Completions)
# ─────────────────────────────────────────────────────────────────────────────

class GPTAnnotator(BaseAnnotator):
    """
    Uses OpenAI GPT-4o via standard Chat Completions endpoint.
    Set OPENAI_API_KEY in environment before instantiating.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ):
        import os
        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self._client     = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    def annotate(self, sentences: List[str]) -> List[Dict]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_message(sentences)},
        ]

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        raw = resp.choices[0].message.content or ""
        # strip markdown fences if model adds them despite instructions
        raw = re.sub(r"^```[a-z]*\n?|```$", "", raw.strip(), flags=re.MULTILINE)
        print(f"[GPT-4o] Raw preview: {raw[:300]}")
        return self._parse_response(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

def print_annotations(annotations: List[Dict], model_label: str = "") -> None:
    header = f"── Annotations{' — ' + model_label if model_label else ''} "
    print(f"\n{header}{'─' * max(0, 60 - len(header))}")
    for ann in annotations:
        print(f"\n  Sentence : {ann.get('Sentence')}")
        roles = ann.get("Roles", {})
        print(f"  Subject  : {roles.get('Subject')}  "
              f"Verb: {roles.get('Verb')}  Object: {roles.get('Object')}")
        uf = ann.get("UsageFeatures", {})
        print(f"  Construction : {uf.get('Construction'):<14}  "
              f"Aspect : {uf.get('Aspect')}")
        print(f"  NPCase       : {uf.get('NPCase'):<14}  "
              f"VerbAgr: {uf.get('VerbAgreement')}")
        print(f"  Quantifier   : {str(uf.get('Quantifier')):<14}  "
              f"LocPP  : {uf.get('LocativePP')}")
        print(f"  ResultXP     : {str(uf.get('ResultativeXP')):<14}  "
              f"Polarity: {uf.get('EventPolarity')}")


# ─────────────────────────────────────────────────────────────────────────────
# Demo / entry-point
# ─────────────────────────────────────────────────────────────────────────────

TEST_SENTENCES = [
    # W1 – Telicity logic
    "Przybywało wiele faktów, ale nie przybyło.",
    # W2 – Existential construction
    "Przybyło wielu świadków.",
    # W3 – Construction shift
    "Przybyło dowodów.",
    "Dowody przybyły.",
    # W4 – Quantifier → Case
    "Przybywało wiele faktów.",
    # W5 – Resultative XP
    "Na jaw przybywało faktów.",
    # W6 – Decrease
    "Ubyło wody.",
]

if __name__ == "__main__":
    import os
    from transformers import AutoTokenizer   # lazy: only needed for demo

    # ── LLaMA (local) ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        "NousResearch/Meta-Llama-3-8B-Instruct"
    )
    llama = LlamaAnnotator(tokenizer=tokenizer)
    llama_anns = llama.annotate(TEST_SENTENCES)
    print_annotations(llama_anns, "LLaMA-3.1-8B")

    with open("annotations_llama.json", "w", encoding="utf-8") as f:
        json.dump(llama_anns, f, ensure_ascii=False, indent=2)
    print("\nSaved: annotations_llama.json")

    # ── GPT-4o ───────────────────────────────────────────────────────────────
    # Reads OPENAI_API_KEY from environment automatically
    gpt = GPTAnnotator()
    gpt_anns = gpt.annotate(TEST_SENTENCES)
    print_annotations(gpt_anns, "GPT-4o")

    with open("annotations_gpt4o.json", "w", encoding="utf-8") as f:
        json.dump(gpt_anns, f, ensure_ascii=False, indent=2)
    print("Saved: annotations_gpt4o.json")