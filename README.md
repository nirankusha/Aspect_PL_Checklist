# Polish Aspect CheckList Evaluation

Constructional lattice evaluation for Polish existential–motion alternation.

This repository evaluates LLM linguistic competence using CheckList-style behavioral testing for Polish aspectual constructions.

---

# Pipeline

CheckList suite
      ↓
Binary classification (Type 1)
      ↓
Slot-choice ranking (Type 2)
      ↓
Few-shot constructional annotation (Type 3)
      ↓
Capability scoring
      ↓
Feature lattice evaluation

---

# Files

Core:
- annotator.py
- evaluate_checklist.py
- przybywac_checklist_llama.py
- srl_checklist.json

Notebook:
- LTP_Checklist_Eval.ipynb

Paper:
- Aspect_PL_Report.pdf

---

# Installation

pip install checklist openai transformers pandas

---

# Run local LLaMA server

python -m llama_cpp.server   --model Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf   --port 9001

---

# Run evaluation

Full:

python evaluate_checklist.py

Binary only:

python przybywac_checklist_llama.py   --mode binary   --run

Slot-choice:

python przybywac_checklist_llama.py   --mode slot-choice   --run

Annotation:

python przybywac_checklist_llama.py   --mode usage-features   --run

---

# Evaluation Types

Type 1 — Binary classification
Sentence acceptability prediction

Type 2 — Slot-choice
Best grammatical form selection

Type 3 — Constructional annotation
Full feature vector prediction

---

# Feature lattice

W1 — Aspect
W2 — Existential construction
W3 — Perspective switch
W4 — Quantifier governance
W5 — Appearance frame
W6 — Argument structure

---

# Models

LLaMA 3.1 8B (local)
GPT-4o (API)

temperature = 0

---

# Output

Binary accuracy
Slot-choice accuracy
Feature annotation accuracy

---

# Use cases

LLM probing
Polish aspect
Unaccusativity
Construction grammar
SRL diagnostics
