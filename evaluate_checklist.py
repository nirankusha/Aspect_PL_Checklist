#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_checklist.py
=====================
Compatible with BOTH:
  - przybywac_checklist_workflow.py  (suite + slot-choice logic, types 1 & 2)
  - annotator.py                     (LlamaAnnotator / GPTAnnotator, type 3)

The two LlamaAnnotator classes have different interfaces:
  workflow  → _stream_text / _stream_json_array / zero_shot_binary_classify
  annotator → .annotate(sentences) → List[Dict]  (with UsageFeatures key)

Type 3 always uses annotator.py's interface (.annotate).
Types 1 & 2 always use the workflow's interface (zero_shot_binary_classify /
zero_shot_slot_choice).
"""

import os, re, json
from collections import defaultdict
from typing import List, Dict

import pandas as pd

# ── Suite / slot-choice logic from workflow ───────────────────────────────────
from przybywac_checklist_llama import (
    build_binary_suite,
    build_slot_choice_examples,
    LlamaAnnotator       as WorkflowLlama,
    init_annotator_runtime,
    choose_best_option_zero_shot,
    evaluate_slot_choices,
)

# ── Type-3 annotators from annotator.py ──────────────────────────────────────
from annotator import (
    LlamaAnnotator as AnnotatorLlama,
    GPTAnnotator,
    print_annotations,
    SYSTEM_PROMPT,
)

# =============================================================================
# CONFIG
# =============================================================================

LLAMA_BASE_URL   = "http://localhost:9001/v1"
LLAMA_MODEL_NAME = "local-model"
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "sk-...")
GPT_MODEL        = "gpt-4o"

# =============================================================================
# BLOCK → (construction, aspect) — keys = exact name= strings in workflow file
# =============================================================================

BLOCK_META = {
    "W1-T1: Telicity — T1 ipfv/pfv contrast":                              ("existential", "IPFV"),
    "W1-T2: Telicity — T2 bounded-event":                                  ("entity",       "IPFV"),
    "W2-T1: Existential Construction — neuter default (T1 bare)":          ("existential", "IPFV"),
    "W2-T2: Existential Construction — neuter default (T2 with PP)":       ("existential", "PFV"),
    "W3-T1: Perspective Switch — T1\u2192T2 contrast":                     ("existential", "IPFV"),
    "W3-T2: Perspective Switch — T2 entity subject":                       ("entity",       "PFV"),
    "W4-T1: Quantifier Governance — wiele + inanimate NP (T1)":            ("existential", "IPFV"),
    "W4-T2: Quantifier Governance — wielu + animate NP (T2)":              ("existential", "PFV"),
    "W5-T1: Appearance / Emergence — appearance frame (T1)":               ("existential", "PFV"),
    "W5-T2: Appearance / Emergence — motion frame (T2)":                   ("entity",       "PFV"),
    "W6-T1: Argument Structure — qty-existential + LOC PP (T1)":           ("existential", "PFV"),
    "W6-T2: Argument Structure — entity-motion + directional ACC PP (T2)": ("entity",       "PFV"),
}

# =============================================================================
# TYPE 1 — Binary  (workflow annotator: zero_shot_binary_classify)
# =============================================================================

def run_binary_eval(suite, wf_annotator: WorkflowLlama, label: str) -> List[Dict]:
    records = []
    for test_name, test in suite.tests.items():
        capability = getattr(test, "capability", test_name)
        sentences  = [str(s) for s in test.data]
        golds      = list(test.labels)

        rows    = wf_annotator.zero_shot_binary_classify(sentences)
        by_sent = {r["Sentence"]: r for r in rows if "Sentence" in r}

        for sent, gold in zip(sentences, golds):
            row  = by_sent.get(sent, {})
            pred = int(row.get("Prediction", 0))
            records.append({
                "model": label, "block": test_name, "capability": capability,
                "sentence": sent, "gold": gold, "pred": pred,
                "correct": int(gold == pred), "reason": row.get("Reason", ""),
            })
    return records

# =============================================================================
# TYPE 2 — Slot-choice  (workflow annotator: zero_shot_slot_choice)
# =============================================================================

def run_slot_choice_eval(wf_annotator: WorkflowLlama, label: str) -> List[Dict]:
    init_annotator_runtime(wf_annotator)
    examples = build_slot_choice_examples()
    results  = evaluate_slot_choices(examples, choose_best_option_zero_shot)
    return [{
        "model": label, "name": r["name"], "capability": r["capability"],
        "prompt": r["prompt"], "gold": r["gold"], "pred": r["pred"],
        "correct": int(r["pass"]), "reason": r.get("reason", ""),
    } for r in results]

# =============================================================================
# TYPE 3 — Few-shot annotation  (annotator.py: .annotate())
# =============================================================================

# One representative PASS sentence per block (drawn from build_binary_suite)
ANNOTATION_TEST_SENTENCES = [
    "Przybywało wiele faktów, ale nie przybyło nowych dowodów.",    # W1-T1
    "Świadkowie przybywali na miejsce, ale nie przybyli wszyscy.",  # W1-T2
    "Przybywało wiele faktów.",                                      # W2-T1
    "Na konferencję przybyło wielu uczestników.",                    # W2-T2
    "Na miejsce przybyli świadkowie.",                               # W3-T2
    "Na miejsce przybyło wielu świadków.",                           # W4-T2
    "Do miasta przybyli nowi mieszkańcy.",                           # W5-T2
    "Nowi uczestnicy przybyli na konferencję.",                      # W6-T2
]

ANNOTATION_GOLD = [
    dict(Construction="existential", Aspect="IPFV", NPCase="Gen",  VerbAgreement="V-Neut", EventPolarity="appearance"),
    dict(Construction="entity",      Aspect="IPFV", NPCase="Nom",  VerbAgreement="V-Plur", EventPolarity="appearance"),
    dict(Construction="existential", Aspect="IPFV", NPCase="Gen",  VerbAgreement="V-Neut", EventPolarity="appearance"),
    dict(Construction="existential", Aspect="PFV",  NPCase="Gen",  VerbAgreement="V-Neut", EventPolarity="appearance"),
    dict(Construction="entity",      Aspect="PFV",  NPCase="Nom",  VerbAgreement="V-Plur", EventPolarity="appearance"),
    dict(Construction="existential", Aspect="PFV",  NPCase="Gen",  VerbAgreement="V-Neut", EventPolarity="appearance"),
    dict(Construction="entity",      Aspect="PFV",  NPCase="Nom",  VerbAgreement="V-Plur", EventPolarity="appearance"),
    dict(Construction="entity",      Aspect="PFV",  NPCase="Nom",  VerbAgreement="V-Plur", EventPolarity="appearance"),
]

ANNOTATION_FIELDS = ["Construction", "Aspect", "NPCase", "VerbAgreement", "EventPolarity"]


def run_annotation_eval(ann_annotator, label: str):
    """
    Uses annotator.py's .annotate() interface.
    Returns (preds, score_records).
    """
    # annotator.py annotates all sentences in one call
    anns = ann_annotator.annotate(ANNOTATION_TEST_SENTENCES)

    # Build a sentence→UsageFeatures lookup
    by_sent = {}
    for item in anns:
        uf = item.get("UsageFeatures", {})
        by_sent[item.get("Sentence", "")] = uf

    # Flatten to list aligned with ANNOTATION_TEST_SENTENCES
    preds = []
    for sent in ANNOTATION_TEST_SENTENCES:
        uf   = by_sent.get(sent, {})
        flat = {"Sentence": sent}
        flat.update(uf)
        preds.append(flat)

    records = []
    for sent, pred, gold in zip(ANNOTATION_TEST_SENTENCES, preds, ANNOTATION_GOLD):
        for field in ANNOTATION_FIELDS:
            records.append({
                "model": label, "sentence": sent, "field": field,
                "gold": gold.get(field, ""), "pred": pred.get(field, ""),
                "correct": int(pred.get(field, "") == gold.get(field, "")),
            })
    return preds, records


def print_annotations_inline(preds: List[Dict]) -> None:
    w = 52
    print(f"\n  {'Sentence':<{w}}  {'Constr':<12} {'Asp':<5} {'Case':<5} {'Agr':<8} {'Polarity'}")
    print(f"  {'-'*w}  {'-'*12} {'-'*5} {'-'*5} {'-'*8} {'-'*10}")
    for a in preds:
        print(f"  {a.get('Sentence','')[:w]:<{w}}  "
              f"{a.get('Construction','?'):<12} {a.get('Aspect','?'):<5} "
              f"{a.get('NPCase','?'):<5} {a.get('VerbAgreement','?'):<8} "
              f"{a.get('EventPolarity','?')}")

# =============================================================================
# LATTICE MATRICES
# =============================================================================

def construction_aspect_matrix(binary_records: List[Dict]) -> pd.DataFrame:
    rows: Dict = defaultdict(lambda: defaultdict(list))
    for r in binary_records:
        meta = BLOCK_META.get(r["block"])
        if meta:
            rows[meta[0]][meta[1]].append(r["correct"])
    data = []
    for c in ["existential", "entity"]:
        row = {"Construction": c.capitalize() + " construction"}
        for asp in ["IPFV", "PFV"]:
            v = rows[c].get(asp, [])
            row[asp] = round(sum(v)/len(v), 2) if v else float("nan")
        data.append(row)
    return pd.DataFrame(data).set_index("Construction")


def case_agreement_matrix(binary_records: List[Dict]) -> pd.DataFrame:
    gen = [r["correct"] for r in binary_records
           if BLOCK_META.get(r["block"], ("",""))[0] == "existential"]
    nom = [r["correct"] for r in binary_records
           if BLOCK_META.get(r["block"], ("",""))[0] == "entity"]
    return pd.DataFrame([
        {"NP Case":"Nom","Expected Agreement":"V-Plur",
         "Accuracy": round(sum(nom)/len(nom),2) if nom else float("nan")},
        {"NP Case":"Gen","Expected Agreement":"V-Neut",
         "Accuracy": round(sum(gen)/len(gen),2) if gen else float("nan")},
    ]).set_index("NP Case")

# =============================================================================
# MAIN
# =============================================================================

def run_all():
    suite      = build_binary_suite()
    all_binary = []
    all_rank   = []
    all_annot  = []

    MODELS = [
        # (workflow annotator,             type-3 annotator,           label)
        (WorkflowLlama(base_url=LLAMA_BASE_URL, api_key="not-needed",
                       model_name=LLAMA_MODEL_NAME),
         AnnotatorLlama(base_url=LLAMA_BASE_URL, model_name=LLAMA_MODEL_NAME),
         "LLaMA-3.1-8B"),

        (WorkflowLlama(base_url="https://api.openai.com/v1",
                       api_key=OPENAI_API_KEY, model_name=GPT_MODEL),
         GPTAnnotator(api_key=OPENAI_API_KEY, model=GPT_MODEL),
         "GPT-4o"),
    ]

    for wf_ann, ann_ann, label in MODELS:
        print(f"\n{'='*60}\n  Model: {label}\n{'='*60}")

        # ── Type 1 ──────────────────────────────────────────────────────────
        print("\n▶ Prompt type 1 — Binary classification")
        recs = run_binary_eval(suite, wf_ann, label)
        all_binary.extend(recs)
        df1 = pd.DataFrame(recs)
        print(df1.groupby("capability")["correct"]
                 .agg(["sum","count"])
                 .assign(accuracy=lambda x:(x["sum"]/x["count"]).round(2))
                 .rename(columns={"sum":"correct","count":"total"})
                 .to_string())

        # ── Type 2 ──────────────────────────────────────────────────────────
        print("\n▶ Prompt type 2 — Slot-choice")
        recs2 = run_slot_choice_eval(wf_ann, label)
        all_rank.extend(recs2)
        df2 = pd.DataFrame(recs2)
        print(df2[["name","capability","gold","pred","correct"]].to_string(index=False))
        print(f"  Overall: {df2['correct'].mean():.2f}")

        # ── Type 3 ──────────────────────────────────────────────────────────
        print("\n▶ Prompt type 3 — Few-shot constructional annotation")
        preds, recs3 = run_annotation_eval(ann_ann, label)
        print_annotations_inline(preds)
        # verbose per-sentence view via annotator.py's pretty-printer
        print_annotations(ann_ann.annotate(ANNOTATION_TEST_SENTENCES)
                          if False else [], label)   # set True for full detail
        all_annot.extend(recs3)
        df3 = pd.DataFrame(recs3)
        print("\n  Field accuracy:")
        print(df3.groupby("field")["correct"].mean().round(2).rename("accuracy").to_string())

    print("\n\n── Construction × Aspect matrix ─────────────────────────────────")
    print(construction_aspect_matrix(all_binary).to_string())
    print("\n── NP Case × Agreement matrix ───────────────────────────────────")
    print(case_agreement_matrix(all_binary).to_string())

    for fname, data in [
        ("results_binary.json",  all_binary),
        ("results_ranking.json", all_rank),
        ("results_annot.json",   all_annot),
    ]:
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    print("\nSaved: results_binary.json  results_ranking.json  results_annot.json")


if __name__ == "__main__":
    run_all()
