#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
przybyc_streamlined_workflow.py
================================
Streamlined behavioral testing workflow for Polish przybywać / przybyć.

This version unifies all experiments around a local OpenAI-compatible
instruction model ("LlamaAnnotator") and replaces sequence-classification /
seq2seq-style classification with zero-shot classification via prompted JSON.

Supported modes
---------------
1. binary
   - Full-sentence binary grammaticality / acceptability classification
   - Uses zero-shot JSON annotation from the local instruct model

2. slot-choice
   - Choose the best slot value among all candidate realizations
   - Uses zero-shot forced-choice JSON annotation

3. generate
   - Prompt completion experiment in CheckList style
   - Uses real generation from the local instruct model

4. usage-features
   - Keeps your original construal-sensitive annotation workflow

Install
-------
    pip install checklist openai transformers

Usage
-----
    python przybyc_streamlined_workflow.py --mode binary --run
    python przybyc_streamlined_workflow.py --mode slot-choice --run
    python przybyc_streamlined_workflow.py --mode generate --run
    python przybyc_streamlined_workflow.py --mode usage-features --run

    python przybyc_streamlined_workflow.py --mode binary --json-out binary_results.json
    python przybyc_streamlined_workflow.py --mode slot-choice --slot-json slot_tasks.json
"""

import json
import argparse
import pickle
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Callable

import checklist
from checklist.editor import Editor
from checklist.test_types import MFT
from checklist.test_suite import TestSuite
from checklist.pred_wrapper import PredictorWrapper

try:
    from checklist.expect import Expect
except Exception:
    Expect = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


# ══════════════════════════════════════════════════════════════════════════════
# EDITOR & LEXICONS
# ══════════════════════════════════════════════════════════════════════════════

editor = Editor()

editor.add_lexicon('VERB_IPFV_T1_pass', ['przybywało'])
editor.add_lexicon('VERB_IPFV_T1_fail', ['przybywały', 'przybywali'])
editor.add_lexicon('VERB_PFV_T1_pass',  ['przybyło'])
editor.add_lexicon('VERB_PFV_T1_fail',  ['przybyły', 'przybyli'])

editor.add_lexicon('VERB_IPFV_T2_pass', ['przybywali'])
editor.add_lexicon('VERB_IPFV_T2_fail', ['przybywało'])
editor.add_lexicon('VERB_PFV_T2_pass',  ['przybyli'])
editor.add_lexicon('VERB_PFV_T2_fail',  ['przybyło', 'przybyły'])

editor.add_lexicon('SUBJ_GEN_pass', ['faktów', 'dowodów', 'danych'])
editor.add_lexicon('SUBJ_NOM_fail', ['nowe fakty', 'nowe dowody'])

editor.add_lexicon('SUBJ_NOM_T2_pass',  ['świadkowie', 'nowi uczestnicy', 'nowi mieszkańcy'])
editor.add_lexicon('SUBJ_GEN_T2_fail',  ['świadków', 'nowych uczestników', 'nowych mieszkańców'])

editor.add_lexicon('QUANT_INAN_pass', ['wiele'])
editor.add_lexicon('QUANT_INAN_fail', ['wielu'])

editor.add_lexicon('PP_NA_ACC_pass', ['na konferencję', 'na miejsce'])
editor.add_lexicon('PP_NA_ACC_fail', ['na konferencji'])
editor.add_lexicon('PP_DO_GEN_pass', ['do miasta', 'do kraju'])
editor.add_lexicon('PP_W_LOC_pass',  ['w sprawie', 'w aktach'])

SLOT_OPTIONS: Dict[str, List[str]] = {
    "VERB_T1_IPFV": ["przybywało", "przybywały", "przybywali"],
    "VERB_T1_PFV":  ["przybyło", "przybyły", "przybyli"],
    "VERB_T2_IPFV": ["przybywali", "przybywało"],
    "VERB_T2_PFV":  ["przybyli", "przybyło", "przybyły"],
    "NP_T1_FACTS": ["faktów", "nowe fakty"],
    "NP_T1_EVID":  ["nowych dowodów", "nowe dowody"],
    "NP_T2_WIT":   ["świadkowie", "świadków"],
    "NP_T2_PART":  ["nowi uczestnicy", "nowych uczestników"],
    "NP_T2_RES":   ["nowi mieszkańcy", "nowych mieszkańców"],
    "QUANT_INAN":  ["wiele", "wielu"],
    "PP_T2_NA":    ["na konferencję", "na konferencji"],
    "PP_T2_PLACE": ["na miejsce"],
    "PP_T1_W":     ["w sprawie", "w aktach"],
    "PP_T2_DO":    ["do miasta", "do kraju"],
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def t(tmpl, label, **slots):
    return editor.template(tmpl, labels=label, meta=True, save=True, **slots)


def flatten_suite_sentences(suite: TestSuite) -> List[str]:
    out = []
    for test in suite.tests.values():
        out.extend(list(test.data))
    return out


def flatten_suite_records(suite: TestSuite) -> List[Dict[str, Any]]:
    rows = []
    idx = 1
    for test_name, test in suite.tests.items():
        cap = getattr(test, "capability", "")
        desc = getattr(test, "description", "")
        metas = getattr(test, "meta", None)
        for i, sent in enumerate(test.data):
            rows.append({
                "id": idx,
                "test_name": test_name,
                "capability": cap,
                "description": desc,
                "sentence": sent,
                "label": int(test.labels[i]) if test.labels is not None else None,
                "meta": metas[i] if metas is not None and i < len(metas) else None,
            })
            idx += 1
    return rows


def print_json_preview(items: List[Dict[str, Any]], n: int = 5):
    for row in items[:n]:
        print(json.dumps(row, ensure_ascii=False, indent=2))


# ══════════════════════════════════════════════════════════════════════════════
# BINARY SUITE
# ══════════════════════════════════════════════════════════════════════════════

def build_binary_suite() -> TestSuite:
    suite = TestSuite()

    ret  = t('{VERBa} wiele {NPa}, ale nie {VERBb} {NPb}.',  1,
             VERBa=['przybywało'],  NPa=['faktów'],
             VERBb=['przybyło'],    NPb=['nowych dowodów'])
    ret += t('{VERBa} wiele {NPa}, ale nie {VERBb} {NPb}.',  0,
             VERBa=['przybywały'],  NPa=['faktów'],
             VERBb=['przybyło'],    NPb=['nowych dowodów'])
    ret += t('{VERBa} wiele {NPa}, ale nie {VERBb} {NPb}.',  0,
             VERBa=['przybywało'],  NPa=['faktów'],
             VERBb=['przybyły'],    NPb=['nowe dowody'])
    ret += t('{VERBa} wiele {NPa}, ale nie {VERBb} {NPb}.',  0,
             VERBa=['przybywało'],  NPa=['nowe fakty'],
             VERBb=['przybyło'],    NPb=['nowych dowodów'])
    suite.add(MFT(**ret,
        name='W1-T1: Telicity — T1 ipfv/pfv contrast',
        capability='Telicity',
        description='T1 telicity contrast with agreement/case perturbations.',
    ))

    ret  = t('{SUBJ} {VERBa} na miejsce, ale nie {VERBb} wszyscy.',  1,
             SUBJ=['Świadkowie'],   VERBa=['przybywali'],  VERBb=['przybyli'])
    ret += t('{SUBJ} {VERBa} na miejsce, ale nie {VERBb} wszyscy.',  0,
             SUBJ=['Świadkowie'],   VERBa=['przybywało'],  VERBb=['przybyli'])
    ret += t('{SUBJ} {VERBa} na miejsce, ale nie {VERBb} wszyscy.',  0,
             SUBJ=['Świadkowie'],   VERBa=['przybywali'],  VERBb=['przybyło'])
    ret += t('{SUBJ} {VERBa} na miejsce, ale nie {VERBb} wszyscy.',  0,
             SUBJ=['Świadków'],     VERBa=['przybywali'],  VERBb=['przybyli'])
    suite.add(MFT(**ret,
        name='W1-T2: Telicity — T2 bounded-event',
        capability='Telicity',
        description='T2 telicity contrast with subject-case and agreement perturbations.',
    ))

    ret  = t('{VERB} wiele {NP}.',  1, VERB=['Przybywało'], NP=['faktów'])
    ret += t('{VERB} wiele {NP}.',  0, VERB=['Przybywały'], NP=['faktów'])
    ret += t('{VERB} wiele {NP}.',  0, VERB=['Przybywało'], NP=['nowe fakty'])
    ret += t('{VERB} wiele {NP}.',  0, VERB=['Przybyły'],   NP=['faktów'])
    suite.add(MFT(**ret,
        name='W2-T1: Existential Construction — neuter default (T1 bare)',
        capability='Existential Construction',
        description='Existential T1 with neuter default and quantified genitive.',
    ))

    ret  = t('Na konferencję {VERB} wielu {NP}.',  1, VERB=['przybyło'], NP=['uczestników'])
    ret += t('Na konferencję {VERB} wielu {NP}.',  0, VERB=['przybyli'], NP=['uczestników'])
    ret += t('Na konferencję {VERB} wielu {NP}.',  0, VERB=['przybyło'], NP=['uczestnicy'])
    ret += t('Na konferencję {VERB} wielu {NP}.',  0, VERB=['przybyły'], NP=['uczestników'])
    suite.add(MFT(**ret,
        name='W2-T2: Existential Construction — neuter default (T2 with PP)',
        capability='Existential Construction',
        description='Existential construction with directional PP and quantified NP.',
    ))

    ret  = t('{VERBa} wielu {NPa}, ale nie {VERBb} {NPb}.',  1,
             VERBa=['Przybywało'], NPa=['faktów'], VERBb=['przybyły'], NPb=['nowe dowody'])
    ret += t('{VERBa} wielu {NPa}, ale nie {VERBb} {NPb}.',  0,
             VERBa=['Przybywało'], NPa=['faktów'], VERBb=['przybyły'], NPb=['nowych dowodów'])
    ret += t('{VERBa} wielu {NPa}, ale nie {VERBb} {NPb}.',  0,
             VERBa=['Przybywali'], NPa=['faktów'], VERBb=['przybyły'], NPb=['nowe dowody'])
    ret += t('{VERBa} wielu {NPa}, ale nie {VERBb} {NPb}.',  0,
             VERBa=['Przybywało'], NPa=['fakty'],  VERBb=['przybyły'], NPb=['nowe dowody'])
    suite.add(MFT(**ret,
        name='W3-T1: Perspective Switch — T1→T2 contrast',
        capability='Perspective Switch',
        description='Switch from T1 genitive existential to T2 nominative entity reading.',
    ))

    ret  = t('Na miejsce {VERB} {SUBJ}.',  1, VERB=['przybyli'], SUBJ=['świadkowie'])
    ret += t('Na miejsce {VERB} {SUBJ}.',  0, VERB=['przybyło'], SUBJ=['świadkowie'])
    ret += t('Na miejsce {VERB} {SUBJ}.',  0, VERB=['przybyli'], SUBJ=['świadków'])
    ret += t('Na miejsce {VERB} {SUBJ}.',  0, VERB=['przybyło'], SUBJ=['świadków'])
    suite.add(MFT(**ret,
        name='W3-T2: Perspective Switch — T2 entity subject',
        capability='Perspective Switch',
        description='T2 entity subject with nominative plural and plural agreement.',
    ))

    ret  = t('{VERB} {QUANT} {NP}.',  1,
             VERB=['Przybywało'], QUANT=['wiele'], NP=['nowych faktów'])
    ret += t('{VERB} {QUANT} {NP}.',  0,
             VERB=['Przybywały'], QUANT=['wiele'], NP=['nowych faktów'])
    ret += t('{VERB} {QUANT} {NP}.',  0,
             VERB=['Przybywało'], QUANT=['wiele'], NP=['nowe fakty'])
    ret += t('{VERB} {QUANT} {NP}.',  0,
             VERB=['Przybywało'], QUANT=['wielu'], NP=['nowe fakty'])
    suite.add(MFT(**ret,
        name='W4-T1: Quantifier Governance — wiele + inanimate NP (T1)',
        capability='Quantifier Governance',
        description='Quantifier governance in T1 with wiele + genitive inanimate NP.',
    ))

    ret  = t('Na miejsce {VERB} wielu {SUBJ}.',  1, VERB=['przybyło'], SUBJ=['świadków'])
    ret += t('Na miejsce {VERB} wielu {SUBJ}.',  0, VERB=['przybyli'], SUBJ=['świadków'])
    ret += t('Na miejsce {VERB} wielu {SUBJ}.',  0, VERB=['przybyło'], SUBJ=['świadkowie'])
    ret += t('Na miejsce {VERB} wielu {SUBJ}.',  0, VERB=['przybyły'], SUBJ=['świadków'])
    suite.add(MFT(**ret,
        name='W4-T2: Quantifier Governance — wielu + animate NP (T2)',
        capability='Quantifier Governance',
        description='Quantifier governance with wielu and existential agreement.',
    ))

    ret  = t('{VERB} {NP} w sprawie.',  1, VERB=['Przybyło'], NP=['nowych faktów'])
    ret += t('{VERB} {NP} w sprawie.',  0, VERB=['Przybyły'], NP=['nowych faktów'])
    ret += t('{VERB} {NP} w sprawie.',  0, VERB=['Przybyło'], NP=['nowe fakty'])
    ret += t('{VERB} {NP} w sprawie.',  0, VERB=['Przybyli'], NP=['nowych faktów'])
    suite.add(MFT(**ret,
        name='W5-T1: Appearance / Emergence — appearance frame (T1)',
        capability='Appearance / Emergence',
        description='Appearance frame with genitive theme and neuter verb.',
    ))

    ret  = t('Do miasta {VERB} {NP}.',  1, VERB=['przybyli'], NP=['nowi mieszkańcy'])
    ret += t('Do miasta {VERB} {NP}.',  0, VERB=['przybyło'], NP=['nowi mieszkańcy'])
    ret += t('Do miasta {VERB} {NP}.',  0, VERB=['przybyli'], NP=['nowych mieszkańców'])
    ret += t('Do miasta {VERB} {NP}.',  0, VERB=['przybyło'], NP=['nowych mieszkańców'])
    suite.add(MFT(**ret,
        name='W5-T2: Appearance / Emergence — motion frame (T2)',
        capability='Appearance / Emergence',
        description='Motion frame with directional destination and nominative plural subject.',
    ))

    ret  = t('{VERB} {NP} w sprawie.',  1, VERB=['Przybyło'], NP=['nowych dowodów'])
    ret += t('{VERB} {NP} w sprawie.',  0, VERB=['Przybyły'], NP=['nowych dowodów'])
    ret += t('{VERB} {NP} w sprawie.',  0, VERB=['Przybyło'], NP=['nowe dowody'])
    ret += t('{VERB} {NP} w sprawie.',  0, VERB=['Przybyli'], NP=['nowych dowodów'])
    suite.add(MFT(**ret,
        name='W6-T1: Argument Structure — qty-existential + LOC PP (T1)',
        capability='Argument Structure',
        description='Full T1 argument structure with genitive NP and static PP.',
    ))

    ret  = t('{SUBJ} {VERB} {PP}.',  1,
             SUBJ=['Nowi uczestnicy'], VERB=['przybyli'], PP=['na konferencję'])
    ret += t('{SUBJ} {VERB} {PP}.',  0,
             SUBJ=['Nowi uczestnicy'], VERB=['przybyło'], PP=['na konferencję'])
    ret += t('{SUBJ} {VERB} {PP}.',  0,
             SUBJ=['Nowi uczestnicy'], VERB=['przybyli'], PP=['na konferencji'])
    ret += t('{SUBJ} {VERB} {PP}.',  0,
             SUBJ=['Nowych uczestników'], VERB=['przybyli'], PP=['na konferencję'])
    suite.add(MFT(**ret,
        name='W6-T2: Argument Structure — entity-motion + directional ACC PP (T2)',
        capability='Argument Structure',
        description='Full T2 motion frame with nominative subject and directional PP.',
    ))

    return suite


# ══════════════════════════════════════════════════════════════════════════════
# SLOT-CHOICE TASKS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SlotChoiceExample:
    prompt: str
    slot_name: str
    options: List[str]
    gold: str
    capability: str
    name: str
    meta: Dict[str, Any]


def build_slot_choice_examples() -> List[SlotChoiceExample]:
    return [
        SlotChoiceExample(
            prompt="Na miejsce {VERB} świadkowie.",
            slot_name="VERB",
            options=SLOT_OPTIONS["VERB_T2_PFV"],
            gold="przybyli",
            capability="Perspective Switch",
            name="SC-VERB-T2-PFV-AGR",
            meta={"frame": "T2", "reason": "nom-pl-agr"}
        ),
        SlotChoiceExample(
            prompt="Przybywało wiele {NP}.",
            slot_name="NP",
            options=SLOT_OPTIONS["NP_T1_FACTS"],
            gold="faktów",
            capability="Quantifier Governance",
            name="SC-NP-T1-GENITIVE",
            meta={"frame": "T1", "reason": "quantifier-genitive"}
        ),
        SlotChoiceExample(
            prompt="Nowi uczestnicy przybyli {PP}.",
            slot_name="PP",
            options=SLOT_OPTIONS["PP_T2_NA"],
            gold="na konferencję",
            capability="Argument Structure",
            name="SC-PP-T2-DIRECTIONAL",
            meta={"frame": "T2", "reason": "directional-case"}
        ),
        SlotChoiceExample(
            prompt="{VERB} nowych dowodów w sprawie.",
            slot_name="VERB",
            options=SLOT_OPTIONS["VERB_T1_PFV"],
            gold="przybyło",
            capability="Argument Structure",
            name="SC-VERB-T1-PFV-AGR",
            meta={"frame": "T1", "reason": "neuter-default"}
        ),
        SlotChoiceExample(
            prompt="Do miasta {VERB} nowi mieszkańcy.",
            slot_name="VERB",
            options=SLOT_OPTIONS["VERB_T2_PFV"],
            gold="przybyli",
            capability="Appearance / Emergence",
            name="SC-VERB-T2-MOTION",
            meta={"frame": "T2", "reason": "nom-pl-agr"}
        ),
        SlotChoiceExample(
            prompt="{VERB} wiele nowych faktów.",
            slot_name="VERB",
            options=SLOT_OPTIONS["VERB_T1_IPFV"],
            gold="przybywało",
            capability="Existential Construction",
            name="SC-VERB-T1-IPFV-EXISTENTIAL",
            meta={"frame": "T1", "reason": "neuter-default"}
        ),
        SlotChoiceExample(
            prompt="{QUANT} nowych faktów przybywało.",
            slot_name="QUANT",
            options=SLOT_OPTIONS["QUANT_INAN"],
            gold="wiele",
            capability="Quantifier Governance",
            name="SC-QUANT-INAN",
            meta={"frame": "T1", "reason": "inanimate-quantifier-selection"}
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# LOCAL INSTRUCT MODEL / LLAMA ANNOTATOR
# ══════════════════════════════════════════════════════════════════════════════

def repair_streamed_json_array(text: str) -> str:
    start = text.find("[")
    if start == -1:
        raise ValueError("No JSON array start found")
    text = text[start:]
    last_obj = text.rfind("}")
    if last_obj != -1:
        text = text[: last_obj + 1]
    text = text.rstrip().rstrip(",")
    if not text.endswith("]"):
        text += "\n]"
    return text


class LlamaAnnotator:
    def __init__(
        self,
        tokenizer=None,
        base_url: str = "http://localhost:9001/v1",
        api_key: str = "not-needed",
        model_name: str = "local-model",
    ):
        if OpenAI is None:
            raise ImportError("openai package is not installed")
        self.tokenizer = tokenizer
        self.model_name = model_name
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def _stream_text(self, messages, temperature=0.0, max_tokens=2048) -> str:
        completion = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        buffer = ""
        for chunk in completion:
            delta = chunk.choices[0].delta.content
            if delta:
                buffer += delta
        return buffer.strip()

    def _stream_json_array(self, messages, temperature=0.0, max_tokens=4096) -> List[Dict[str, Any]]:
        completion = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        buffer = ""
        started = False

        for chunk in completion:
            delta = chunk.choices[0].delta.content
            if not delta:
                continue
            if not started and "[" in delta:
                started = True
            if started:
                buffer += delta

        try:
            try:
                data = json.loads(buffer)
            except json.JSONDecodeError:
                repaired = repair_streamed_json_array(buffer)
                data = json.loads(repaired)

            if not isinstance(data, list):
                raise ValueError("Response is not a JSON array")

            cleaned = []
            for item in data:
                if isinstance(item, dict):
                    cleaned.append(item)
            return cleaned
        except Exception as e:
            print("Final JSON parsing failed:", e)
            print("Preview:", buffer[:800])
            return []

    def classify_usage_features(self, sentences: List[str]) -> List[Dict[str, Any]]:
        history = [{
            "role": "user",
            "content": (
                "You are given Polish sentences describing filling and stuffing events.\n\n"
                "Analyze each sentence using a typed feature–transition system:\n\n"
                "event\n"
                "  Agent → entity\n"
                "  Theme → entity\n"
                "  Goal → entity\n\n"
                "FEATURE SYSTEM:\n"
                "AgentType: sentient | non-sentient\n"
                "ThemeType: solid | unbound | abstract\n"
                "GoalDimensionality: 3D | 2D | non-material\n"
                "GoalBoundedness: bounded | unbounded\n"
                "GoalPlasticity: hard | soft | undefined\n\n"
                "Analyze the following sentences:\n\n"
                + "\n".join(f"- {s}" for s in sentences) +
                "\n\nReturn ONLY a JSON array.\n"
                "Each element must have this format:\n"
                "{\n"
                "  \"Sentence\": \"<original sentence>\",\n"
                "  \"Roles\": {\"Agent\": \"...\", \"Theme\": \"...\", \"Goal\": \"...\"},\n"
                "  \"UsageFeatures\": {\n"
                "    \"AgentType\": \"...\",\n"
                "    \"ThemeType\": \"...\",\n"
                "    \"GoalDimensionality\": \"...\",\n"
                "    \"GoalBoundedness\": \"...\",\n"
                "    \"GoalPlasticity\": \"...\"\n"
                "  }\n"
                "}\n"
            )
        }]
        data = self._stream_json_array(history, temperature=0.0, max_tokens=4096)
        return [
            item for item in data
            if isinstance(item, dict)
            and "Sentence" in item
            and "Roles" in item
            and "UsageFeatures" in item
        ]

    def zero_shot_binary_classify(self, sentences: List[str]) -> List[Dict[str, Any]]:
        history = [{
            "role": "user",
            "content": (
                "You are evaluating Polish sentences for grammatical acceptability in the intended "
                "przybywać / przybyć existential-motion frame.\n\n"
                "For each sentence, assign exactly one label:\n"
                "1 = acceptable / grammatical in the intended reading\n"
                "0 = unacceptable / ungrammatical in the intended reading\n\n"
                "Base the judgment on agreement, case assignment, quantifier governance, "
                "existential vs entity reading, aspectual frame, and directional/static PP selection.\n\n"
                "Return ONLY a JSON array.\n"
                "Each item must have this format:\n"
                "{\n"
                "  \"Sentence\": \"<original sentence>\",\n"
                "  \"Prediction\": 0 or 1,\n"
                "  \"Confidence\": <float between 0 and 1>,\n"
                "  \"Reason\": \"<short linguistic reason>\"\n"
                "}\n\n"
                "Sentences:\n"
                + "\n".join(f"- {s}" for s in sentences)
            )
        }]
        data = self._stream_json_array(history, temperature=0.0, max_tokens=4096)
        cleaned = []
        for item in data:
            if isinstance(item, dict) and "Sentence" in item and "Prediction" in item:
                cleaned.append(item)
        return cleaned

    def zero_shot_slot_choice(self, examples: List[SlotChoiceExample]) -> List[Dict[str, Any]]:
        tasks = []
        for ex in examples:
            tasks.append(json.dumps({
                "name": ex.name,
                "capability": ex.capability,
                "prompt": ex.prompt,
                "slot_name": ex.slot_name,
                "options": ex.options,
                "meta": ex.meta
            }, ensure_ascii=False))

        history = [{
            "role": "user",
            "content": (
                "You are evaluating Polish morphosyntax using forced-choice zero-shot classification.\n\n"
                "For each task, choose exactly one option that best fills the target slot.\n"
                "Use agreement, case, quantifier governance, existential/entity construal, and PP selection.\n\n"
                "Return ONLY a JSON array.\n"
                "Each item must have this format:\n"
                "{\n"
                "  \"name\": \"...\",\n"
                "  \"prompt\": \"...\",\n"
                "  \"slot_name\": \"...\",\n"
                "  \"options\": [\"...\"],\n"
                "  \"Prediction\": \"<one option exactly>\",\n"
                "  \"Confidence\": <float between 0 and 1>,\n"
                "  \"Reason\": \"<short linguistic reason>\"\n"
                "}\n\n"
                "Tasks:\n"
                + "\n".join(f"- {x}" for x in tasks)
            )
        }]
        data = self._stream_json_array(history, temperature=0.0, max_tokens=4096)
        cleaned = []
        for item in data:
            if (
                isinstance(item, dict)
                and "name" in item
                and "prompt" in item
                and "slot_name" in item
                and "Prediction" in item
            ):
                cleaned.append(item)
        return cleaned

    def generate_completions(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        max_tokens: int = 96,
        temperature: float = 0.0,
    ) -> List[str]:
        outputs = []
        for prompt in prompts:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({
                "role": "user",
                "content": (
                    f"Complete the following Polish prompt naturally and briefly.\n\n"
                    f"Prompt: {prompt}\n\n"
                    f"Return only the continuation, not the prompt."
                )
            })
            out = self._stream_text(messages, temperature=temperature, max_tokens=max_tokens)
            outputs.append(out.strip())
        return outputs

    def sentences_to_checklist_json(
        self,
        sentences: List[str],
        task_description: str = "Polish grammaticality evaluation"
    ) -> List[Dict[str, Any]]:
        history = [{
            "role": "user",
            "content": (
                f"You are preparing a JSON dataset for {task_description}.\n\n"
                "Return ONLY a JSON array with entries:\n"
                "{\n"
                "  \"content\": \"<sentence>\",\n"
                "  \"id\": <integer>,\n"
                "  \"notes\": \"<short note>\"\n"
                "}\n\n"
                "Sentences:\n"
                + "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
            )
        }]
        return self._stream_json_array(history, temperature=0.0, max_tokens=4096)


# ══════════════════════════════════════════════════════════════════════════════
# ZERO-SHOT CLASSIFICATION ADAPTERS
# ══════════════════════════════════════════════════════════════════════════════

_ANNOTATOR_RUNTIME = {
    "annotator": None,
    "binary_cache": {},
}


def init_annotator_runtime(annotator: LlamaAnnotator):
    _ANNOTATOR_RUNTIME["annotator"] = annotator
    _ANNOTATOR_RUNTIME["binary_cache"] = {}


def predict_proba_zero_shot_binary(sentences: List[str]):
    annotator = _ANNOTATOR_RUNTIME["annotator"]
    if annotator is None:
        raise RuntimeError("Annotator runtime not initialized.")

    rows = annotator.zero_shot_binary_classify(sentences)
    by_sent = {row["Sentence"]: row for row in rows if "Sentence" in row}

    probs = []
    for sent in sentences:
        row = by_sent.get(sent, {})
        pred = int(row.get("Prediction", 0))
        conf = float(row.get("Confidence", 0.5))
        conf = min(max(conf, 0.0), 1.0)

        if pred == 1:
            p_good = conf
        else:
            p_good = 1.0 - conf

        p_bad = 1.0 - p_good
        probs.append([p_bad, p_good])

        _ANNOTATOR_RUNTIME["binary_cache"][sent] = {
            "pred": pred,
            "conf": conf,
            "reason": row.get("Reason", "")
        }

    return probs


def predict_label_zero_shot_binary(sentences: List[str]) -> List[int]:
    probs = predict_proba_zero_shot_binary(sentences)
    return [1 if row[1] >= row[0] else 0 for row in probs]


def choose_best_option_zero_shot(example: SlotChoiceExample) -> Dict[str, Any]:
    annotator = _ANNOTATOR_RUNTIME["annotator"]
    if annotator is None:
        raise RuntimeError("Annotator runtime not initialized.")

    rows = annotator.zero_shot_slot_choice([example])
    if not rows:
        pred = example.options[0]
        reason = "No output"
        conf = 0.0
    else:
        row = rows[0]
        pred = str(row.get("Prediction", example.options[0]))
        reason = row.get("Reason", "")
        conf = float(row.get("Confidence", 0.0))
        if pred not in example.options:
            pred = example.options[0]

    confs = []
    for opt in example.options:
        if opt == pred:
            confs.append(conf if conf > 0 else 1.0)
        else:
            if len(example.options) > 1:
                confs.append((1.0 - conf) / (len(example.options) - 1))
            else:
                confs.append(1.0)

    completed = [
        example.prompt.replace("{" + example.slot_name + "}", opt)
        for opt in example.options
    ]

    return {
        "pred": pred,
        "conf": confs,
        "completed": completed,
        "pass": pred == example.gold,
        "reason": reason,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SLOT-CHOICE EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_slot_choices(
    examples: List[SlotChoiceExample],
    chooser_fn: Callable[[SlotChoiceExample], Dict[str, Any]]
) -> List[Dict[str, Any]]:
    results = []
    for ex in examples:
        out = chooser_fn(ex)
        results.append({
            "name": ex.name,
            "capability": ex.capability,
            "prompt": ex.prompt,
            "slot_name": ex.slot_name,
            "options": ex.options,
            "gold": ex.gold,
            "pred": out["pred"],
            "conf": out["conf"],
            "completed": out["completed"],
            "pass": bool(out["pass"]),
            "reason": out.get("reason", ""),
            "meta": ex.meta,
        })
    return results


def print_slot_choice_results(results: List[Dict[str, Any]]) -> None:
    total = len(results)
    correct = sum(int(r["pass"]) for r in results)

    print("\n" + "═" * 80)
    print("SLOT-CHOICE RESULTS")
    print("═" * 80)

    for r in results:
        print(f"\n[{r['capability']}] {r['name']}")
        print(f"Prompt:   {r['prompt']}")
        print(f"Options:  {r['options']}")
        print(f"Gold:     {r['gold']}")
        print(f"Pred:     {r['pred']}")
        print(f"Pass:     {'✓' if r['pass'] else '✗'}")
        if r.get("reason"):
            print(f"Reason:   {r['reason']}")
        for cand, score in zip(r["completed"], r["conf"]):
            print(f"  - {score:.4f} :: {cand}")

    print("\n" + "═" * 80)
    print(f"Accuracy: {correct}/{total} = {correct / max(total, 1):.4f}")
    print("═" * 80)


def print_slot_choice_capability_summary(results: List[Dict[str, Any]]) -> None:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        grouped.setdefault(r["capability"], []).append(r)

    print("\n" + "─" * 80)
    print("BY-CAPABILITY SUMMARY")
    print("─" * 80)
    for cap, rows in grouped.items():
        k = sum(int(r["pass"]) for r in rows)
        n = len(rows)
        print(f"{cap:30s}  {k:>3d}/{n:<3d}  acc={k/max(n,1):.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# CHECKLIST GENERATION EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════════

def build_generation_suite() -> TestSuite:
    suite = TestSuite()

    ret = editor.template(
        "{PROMPT}",
        labels=[1, 1, 1, 1],
        PROMPT=[
            "Na miejsce",
            "Do miasta",
            "Na konferencję",
            "Przybywało wiele"
        ],
        meta=True,
        save=True
    )

    expect_fn = None
    if Expect is not None:
        def completion_prefers_target_frame(x, pred, conf, label=None, meta=None):
            pred_l = str(pred).lower()
            good_signals = [
                "przybyli",
                "świadkowie",
                "uczestnicy",
                "mieszkańcy",
                "faktów",
                "dowodów",
            ]
            return any(sig in pred_l for sig in good_signals)
        expect_fn = Expect.single(completion_prefers_target_frame)

    suite.add(MFT(
        **ret,
        name='GEN: prompted completion for target frame continuation',
        capability='Generation',
        description='Prompt completion experiment using the local instruct model.',
        expect=expect_fn
    ))
    return suite


def make_generation_predictor(annotator: LlamaAnnotator, max_tokens: int, temperature: float):
    def _predict(prompts: List[str]) -> List[str]:
        return annotator.generate_completions(
            prompts,
            system_prompt="You are a careful Polish language model.",
            max_tokens=max_tokens,
            temperature=temperature,
        )
    return PredictorWrapper.wrap_predict(_predict)


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ══════════════════════════════════════════════════════════════════════════════

def suite_to_json_file(suite: TestSuite, filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump({"examples": flatten_suite_records(suite)}, f, ensure_ascii=False, indent=2)


def slot_examples_to_json_file(examples: List[SlotChoiceExample], filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump({"examples": [asdict(x) for x in examples]}, f, ensure_ascii=False, indent=2)


def slot_results_to_json_file(results: List[Dict[str, Any]], filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)


def _print_suite(suite: TestSuite):
    total = pass_n = fail_n = 0
    for name, test in suite.tests.items():
        cap = getattr(test, 'capability', '')
        print(f'\n{"─"*72}')
        print(f'  {name}')
        print(f'  Capability: {cap}')
        print(f'{"─"*72}')
        for sent, lbl in zip(test.data, test.labels):
            mark = '✓' if lbl == 1 else '✗'
            print(f'  {mark}  {sent}')
            total += 1
            if lbl == 1:
                pass_n += 1
            else:
                fail_n += 1
    print(f'\n{"═"*72}')
    print(f'  Total: {total} cases  ({pass_n} PASS, {fail_n} FAIL)')
    print(f'{"═"*72}')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def load_tokenizer_if_requested(model_name: str):
    if AutoTokenizer is None:
        return None
    return AutoTokenizer.from_pretrained(model_name)


def main():
    ap = argparse.ArgumentParser(description="Streamlined CheckList workflow with LlamaAnnotator zero-shot classification")
    ap.add_argument('--mode', choices=['binary', 'slot-choice', 'generate', 'usage-features'], default='binary')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--save', action='store_true')
    ap.add_argument('--raw', metavar='FILE')
    ap.add_argument('--json-out', metavar='FILE')
    ap.add_argument('--slot-json', metavar='FILE')

    ap.add_argument('--annotator-model', default='NousResearch/Meta-Llama-3-8B-Instruct')
    ap.add_argument('--annotator-base-url', default='http://localhost:9001/v1')
    ap.add_argument('--annotator-api-key', default='not-needed')
    ap.add_argument('--annotator-engine', default='local-model')

    ap.add_argument('--gen-max-tokens', type=int, default=96)
    ap.add_argument('--gen-temperature', type=float, default=0.0)

    args = ap.parse_args()

    tokenizer = load_tokenizer_if_requested(args.annotator_model)
    annotator = LlamaAnnotator(
        tokenizer=tokenizer,
        base_url=args.annotator_base_url,
        api_key=args.annotator_api_key,
        model_name=args.annotator_engine,
    )
    init_annotator_runtime(annotator)

    if args.mode == 'binary':
        suite = build_binary_suite()
        _print_suite(suite)

        if args.raw:
            suite.to_raw_file(args.raw)
            print(f"\nRaw sentences written to {args.raw}")

        if args.json_out and not args.run:
            suite_to_json_file(suite, args.json_out)
            print(f"\nBinary suite exported → {args.json_out}")

        if args.run:
            predictor = PredictorWrapper.wrap_predict(predict_label_zero_shot_binary)
            suite.run(predictor, overwrite=True)
            suite.summary()

            rows = annotator.zero_shot_binary_classify(flatten_suite_sentences(suite))
            print("\nZero-shot annotation preview:")
            print_json_preview(rows, n=8)

            if args.json_out:
                with open(args.json_out, "w", encoding="utf-8") as f:
                    json.dump({"annotations": rows}, f, ensure_ascii=False, indent=2)
                print(f"\nBinary zero-shot results exported → {args.json_out}")

        if args.save:
            with open('przybyc_mft.pkl', 'wb') as f:
                pickle.dump(suite, f)
            print('\nSuite saved → przybyc_mft.pkl')

    elif args.mode == 'slot-choice':
        examples = build_slot_choice_examples()

        if args.slot_json:
            slot_examples_to_json_file(examples, args.slot_json)
            print(f"\nSlot-choice tasks exported → {args.slot_json}")

        if args.run:
            results = evaluate_slot_choices(examples, choose_best_option_zero_shot)
            print_slot_choice_results(results)
            print_slot_choice_capability_summary(results)

            if args.json_out:
                slot_results_to_json_file(results, args.json_out)
                print(f"\nSlot-choice results exported → {args.json_out}")

    elif args.mode == 'generate':
        suite = build_generation_suite()

        if args.raw:
            suite.to_raw_file(args.raw)
            print(f"\nGeneration prompts written to {args.raw}")

        if args.run:
            predictor = make_generation_predictor(
                annotator=annotator,
                max_tokens=args.gen_max_tokens,
                temperature=args.gen_temperature,
            )
            suite.run(predictor, overwrite=True)
            suite.summary()

            if args.json_out:
                prompts = flatten_suite_sentences(suite)
                completions = annotator.generate_completions(
                    prompts,
                    system_prompt="You are a careful Polish language model.",
                    max_tokens=args.gen_max_tokens,
                    temperature=args.gen_temperature,
                )
                rows = [{"prompt": p, "completion": c} for p, c in zip(prompts, completions)]
                with open(args.json_out, "w", encoding="utf-8") as f:
                    json.dump({"generations": rows}, f, ensure_ascii=False, indent=2)
                print(f"\nGeneration outputs exported → {args.json_out}")

    elif args.mode == 'usage-features':
        sentences = [
            "Piotr wypełnił pokój książkami.",
            "Piotr wypełniał wyrobisko odpadami pogórniczymi.",
            "Piotr zapełnił kosz banknotami.",
            "Piotr zapełniał pokój meblami.",
            "Piotr napełnił karmnik okruchami.",
            "Piotr napełniał woreczki bryłkami.",
            "Piotr wypchał portfele banknotami.",
            "Piotr wypychał kieszenie cukierkami.",
            "Piotr zapchał usta plackiem.",
            "Piotr zapychał szczeliny styropianem.",
            "Piotr napchał kieszenie złotem.",
            "Piotr napychał jabłka bakaliami."
        ]

        if args.run:
            annotations = annotator.classify_usage_features(sentences)
            print_json_preview(annotations, n=len(annotations))

            if args.json_out:
                with open(args.json_out, "w", encoding="utf-8") as f:
                    json.dump({"annotations": annotations}, f, ensure_ascii=False, indent=2)
                print(f"\nUsage-feature annotations exported → {args.json_out}")


if __name__ == '__main__':
    main()