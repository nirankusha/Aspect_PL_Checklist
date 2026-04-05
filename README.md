
# Polish Aspect CheckList Evaluation
Constructional lattice evaluation for Polish existential–motion alternation

This repository evaluates LLM linguistic competence using **CheckList-style behavioral testing**
for Polish aspectual constructions, including:

- existential vs entity alternation
- genitive vs nominative argument realization
- neuter vs plural agreement
- perfective vs imperfective aspect
- quantifier governance
- appearance vs motion event frames

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

# Repository Structure

```
annotator.py
evaluate_checklist.py
przybywac_checklist_llama.py
srl_checklist.json
Aspect_PL_Report.pdf
LTP_Checklist_Eval.ipynb
```

---

# Installation

```bash
pip install checklist openai transformers pandas numpy
```

Optional (local model)

```bash
pip install llama-cpp-python
```

---

# Start local LLaMA server

```bash
python -m llama_cpp.server \
  --model Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf \
  --port 9001 \
  --n_ctx 8192 \
  --n_gpu_layers 35 \
  --chat_format chatml
```

Flags

| flag | description |
|------|-------------|
--model | GGUF model path |
--port | server port |
--n_ctx | context size |
--n_gpu_layers | GPU layers |
--chat_format | chat template |
--temperature | decoding temperature |
--top_p | nucleus sampling |
--repeat_penalty | repetition penalty |

---

# Main evaluation

```bash
python evaluate_checklist.py
```

Environment variables

```bash
OPENAI_API_KEY=...
LLAMA_BASE_URL=http://localhost:9001/v1
LLAMA_MODEL_NAME=local-model
```

---

# Binary classification

```bash
python przybywac_checklist_llama.py \
  --mode binary \
  --run
```

Flags

| flag | description |
|------|-------------|
--mode binary | run binary classification |
--run | execute evaluation |
--json-out | save predictions |
--limit | subset size |
--temperature | decoding temperature |
--max_tokens | generation length |

---

# Slot-choice evaluation

```bash
python przybywac_checklist_llama.py \
  --mode slot-choice \
  --run
```

Flags

| flag | description |
|------|-------------|
--mode slot-choice | ranking evaluation |
--run | execute |
--slot-json | save slot tasks |
--temperature | decoding temperature |
--top_p | sampling |
--max_tokens | generation length |

---

# Few-shot annotation

```bash
python przybywac_checklist_llama.py \
  --mode usage-features \
  --run
```

Flags

| flag | description |
|------|-------------|
--mode usage-features | feature annotation |
--run | execute |
--json-out | save annotations |
--batch_size | annotation batch |
--temperature | decoding temperature |
--max_tokens | output length |

---

# evaluate_checklist.py flags

```bash
python evaluate_checklist.py \
  --model llama \
  --prompt-type all \
  --output results.csv
```

Flags

| flag | description |
|------|-------------|
--model | llama / gpt |
--prompt-type | binary / slot / annotation / all |
--output | csv output |
--device | cpu / cuda |
--limit | subset size |
--seed | random seed |
--verbose | print predictions |

---

# annotator.py flags

LLaMA annotator

```
LlamaAnnotator(
    base_url="http://localhost:9001/v1",
    model_name="local-model",
    temperature=0.0,
    max_tokens=4096
)
```

Parameters

| parameter | description |
|-----------|-------------|
base_url | OpenAI-compatible endpoint |
model_name | model id |
temperature | decoding temperature |
max_tokens | max generation tokens |
tokenizer | optional tokenizer |

---

# Evaluation blocks

W1 — Telicity  
W2 — Existential construction  
W3 — Perspective switch  
W4 — Quantifier governance  
W5 — Appearance / emergence  
W6 — Argument structure  

---

# Feature schema

Construction:
- existential
- entity
- ambiguous

Aspect:
- IPFV
- PFV

NPCase:
- Nom
- Gen
- other

VerbAgreement:
- V-Neut
- V-Plur
- other

EventPolarity:
- appearance
- decrease
- other

---

# Output

Binary

```
accuracy per block
macro average
```

Slot-choice

```
correct / total
ranking accuracy
```

Annotation

```
feature accuracy
per-field accuracy
overall accuracy
```

---

# Example

Binary

```
Przybywało wiele faktów.
→ accept
```

Slot

```
Na miejsce ___ wielu świadków
→ przybyło
```

Annotation

```
Construction: existential
Aspect: PFV
Case: Gen
Agreement: V-Neut
```

---

# Models

Local

- LLaMA‑3.1‑8B Instruct Q4_K_M

API

- GPT‑4o

temperature = 0

---

# Use cases

LLM probing  
Polish aspect evaluation  
construction grammar  
unaccusativity diagnostics  
SRL evaluation  
behavioral testing  

