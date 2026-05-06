# The Illusion of Multi-Agent Advantages

This repository contains (1) the code for generating and evaluating the **Smfrs** benchmark dataset used in our experiments, and (2) the evaluation code for all methods compared in the paper, including automatic MAS frameworks, our expert-designed MAS, and single-agent baselines (CoT & CoT-SC).

## Contents

- Repository Structure
- Dataset Generation & Evaluation
  - Generate the Dataset
  - Evaluate Model Outputs
- Running Experiments
  - CoT & CoT-SC (Single-Agent Baselines)
  - Expert-Designed MAS (Smfrs)
  - Automatic MAS Frameworks
    - ADAS
    - AFlow
    - DyLAN
    - MaAS
    - MAS-Orchestra
    - MAS-Zero

---

## Repository Structure

```
.
├── dataset/
│   ├── src/                          # Dataset generation code
│   │   ├── generate_balanced_dataset.py
│   │   ├── config_schema.py
│   │   ├── problem_composer.py
│   │   ├── task_base.py
│   │   ├── data_sources.py
│   │   └── tasks/
│   └── evaluate/                     # Smfrs evaluation script
│       ├── evaluate_smfr_answer_code.py
│       ├── safe_code_executor.py
│       └── requirements_evaluation.txt
│
└── evaluation/
    ├── SAS/                          # CoT & CoT-SC single-agent baselines
    │   ├── GPQA/
    │   ├── HLEMATH/
    │   ├── BCP/
    │   ├── SMFR/
    │   └── SWE/
    ├── expert_MAS_for_smfr_dataset/ # Expert-designed MAS for Smfrs
    │   ├── agents.py
    │   ├── pipeline.py
    │   ├── run_inference.py
    │   ├── debug_trace.py
    │   └── prompts/
    └── Automatic-MAS/                # Automatic MAS frameworks
        ├── ADAS/
        ├── AFlow/
        ├── DyLAN/
        ├── MaAS/
        ├── MAS-Orchestra/
        └── MAS-Zero/
```

---

## Dataset Generation & Evaluation

### Generate the Dataset

```bash
cd dataset/src

# Install dependencies
pip install yfinance python-dateutil tqdm

# Generate with 2 investors per problem
python generate_balanced_dataset.py 2

# Generate with 3 investors per problem
python generate_balanced_dataset.py 3
```

The complexity parameter `N` controls the number of investors, companies (breadth), transactions per investor (depth), and distractors simultaneously.

### Evaluate Model Outputs

```bash
cd dataset/evaluate

pip install -r requirements_evaluation.txt

# Basic evaluation
python evaluate_smfr_answer_code.py \
    --reference  <path_to>/balanced_dataset_single_2_fixed.jsonl \
    --model-output  your_model_outputs.jsonl

# With validation exclusion (required for n=2 and n=3)
python evaluate_smfr_answer_code.py \
    --reference  <path_to>/balanced_dataset_single_2_fixed.jsonl \
    --model-output  your_model_outputs.jsonl \
    --validation  <path_to>/balanced_dataset_single_validate_fixed.jsonl \
    --timeout 30
```

Model output files should be JSONL with one record per line:

```json
{
  "input":  {"problem": "Here is some data on the smfr prices ..."},
  "output": {"answer": "Alice", "code": "def solve(): ..."}
}
```

The `code` field is optional. If present it must define a `solve()` function returning `{"answer": ...}`.

---

## Running Experiments

### CoT & CoT-SC (Single-Agent Baselines)

Each subdirectory under `evaluation/SAS/` is self-contained. `<data_dir>` should point to the folder containing the `.jsonl` benchmark files.

```bash
export OPENAI_API_KEY="your-api-key"
export MAX_CONCURRENT=50

# ── GPQA ──────────────────────────────────────────────────────────────────
cd evaluation/SAS/GPQA
pip install -r ../requirements.txt

# CoT — run ID 1 (repeat with 2, 3 for 3 runs)
python cot_gpqa.py <data_dir>/gpqa_test.jsonl gpqa_test gpt-4o gpqa_downsampled_gpt-4o "['Assistant']" 1

# CoT-SC — ensemble size 5
python cot_sc_gpqa.py <data_dir>/gpqa_test.jsonl gpqa_test gpt-4o gpqa_downsampled_gpt-4o "['Assistant']" 1 5

# ── HLE-Math ──────────────────────────────────────────────────────────────
cd evaluation/SAS/HLEMATH

python cot_hlemath.py <data_dir>/hlemath_test.jsonl hlemath_test gpt-4o hlemath_downsampled_gpt-4o "['Assistant']" 1
python cot_sc_hlemath.py <data_dir>/hlemath_test.jsonl hlemath_test gpt-4o hlemath_downsampled_gpt-4o "['Assistant']" 1 5

# ── BrowseComp+ ───────────────────────────────────────────────────────────
cd evaluation/SAS/BCP

python cot_bcp.py <data_dir>/bcp_test.jsonl bcp_test gpt-4o bcp_downsampled_gpt-4o "['Assistant']" 1
python cot_sc_bcp.py <data_dir>/bcp_test.jsonl bcp_test gpt-4o bcp_downsampled_gpt-4o "['Assistant']" 1 5

# ── Smfrs ────────────────────────────────────────────────────────────────
cd evaluation/SAS/SMFR

# CODE_EVAL_MODE: 1 = require code output, 0 = direct answer only
python cot_smfrs.py <data_dir>/smfr_test.jsonl smfr_test gpt-4o smfr_gpt-4o "['Assistant']" 1 1
python cot_sc_smfr.py <data_dir>/smfr_test.jsonl smfr_test gpt-4o smfr_gpt-4o "['Assistant']" 1 1 5

# ── SWE-Bench ─────────────────────────────────────────────────────────────
cd evaluation/SAS/SWE

# JUDGE_PATH enables Docker-based evaluation (optional)
python cot_swe.py <data_dir>/swe_test.jsonl swe_test gpt-4o swe_downsampled_gpt-4o "['Assistant']" 1 <judge_path> princeton-nlp/SWE-bench_Lite
python cot_sc_swe.py <data_dir>/swe_test.jsonl swe_test gpt-4o swe_downsampled_gpt-4o "['Assistant']" 1 <judge_path> princeton-nlp/SWE-bench_Lite 5
```

Replace `gpt-4o` with `gpt-5` to switch models. The last numeric argument to `cot_sc_*.py` sets the ensemble size.

---

### Expert-Designed MAS (Smfrs)

A manually designed 5-phase multi-agent pipeline (MetaAgent → ExtractAgent → CalculateAgent → ExtractAgent → Python aggregation) for the Smfrs benchmark.

```bash
cd evaluation/expert_MAS_for_smfr_dataset
pip install -r requirements.txt

export OPENAI_API_KEY="your-api-key"

# Quick test (2 samples)
python run_inference.py --input <data_dir>/smfr_test.jsonl --model gpt-4.1 --test

# Full run — 10 parallel pipelines, 3 repetitions
python run_inference.py \
    --input <data_dir>/smfr_test.jsonl \
    --model gpt-4.1 \
    --concurrency 10 \
    --reps 3
```

Supported model aliases: `gpt-4o`, `gpt-4.1`, `o3`, `o4-mini`, `gpt-5`, `gemini-flash`, `gemini-pro`. For Gemini models also set `GOOGLE_CLOUD_PROJECT` and run `gcloud auth application-default login`.

---

### Automatic MAS Frameworks

Each framework has its own `README.md` with full details. Below are the essential commands to reproduce our experiments.

Place benchmark datasets (`.jsonl` files) in each framework's expected data directory before running. Refer to the individual READMEs for the exact path.

#### ADAS

```bash
cd evaluation/Automatic-MAS/ADAS
pip install -r requirements.txt
export OPENAI_API_KEY="your-api-key"

bash run_gpqa.sh
bash run_hle.sh
bash run_bcp.sh
bash run_smfr.sh
bash run_swe.sh
```

#### AFlow

```bash
cd evaluation/Automatic-MAS/AFlow
pip install -r requirements.txt

# Set API key in config/config2.yaml
# models: { "gpt-4o": { api_key: "your-api-key", ... } }

bash run_gpqa.sh
bash run_hlemath.sh
bash run_bcp.sh
bash run_smfr.sh
bash run_swe.sh
```

#### DyLAN

```bash
cd evaluation/Automatic-MAS/DyLAN/code
pip install -r requirements.txt
export OPENAI_API_KEY="your-api-key"

# GPQA
cd GPQA
python llmlp_listwise_gpqa.py <data_dir>/gpqa_test.jsonl gpqa_test gpt-4o results "['Theoretical Physicist', 'Molecular Chemist', 'Cellular Biologist', 'Assistant']"

# HLE-Math
cd ../HLEMATH
python llmlp_listwise_hlemath.py <data_dir>/hlemath_test.jsonl hlemath_test gpt-4o results "['Mathematician', 'AlgebraExpert', 'GeometryWizard', 'Assistant']"

# BrowseComp+
cd ../BCP
python llmlp_listwise_bcp.py <data_dir>/bcp_test.jsonl bcp_test gpt-4o results "['Knowledge Researcher', 'Cultural Historian', 'Information Analyst', 'Assistant']"

# Smfrs
cd ../SMFR
bash exp_smfr_gpt4o.sh

# SWE-Bench
cd ../SWE
bash exp_swe_gpt4o.sh
```

#### MaAS

```bash
cd evaluation/Automatic-MAS/MaAS
pip install -e .

# Configure ~/.metagpt/config2.yaml with your API key

# Train (search)
python -m examples.maas.optimize --dataset GPQA --round 1 --sample 4

# Test
python -m examples.maas.optimize --dataset GPQA --round 1 --sample 4 --is_test True

# Smfrs (3-run evaluation)
bash run_smfr_test_3x.sh
```

#### MAS-Orchestra

```bash
cd evaluation/Automatic-MAS/MAS-Orchestra
pip install -r requirements.txt
export OPENAI_API_KEY="your-api-key"
export TOGETHER_API_KEY="your-together-key"   # required for gpt-oss-120b

# Step 1: deploy the orchestrator model
bash vllm_deploy.sh

# Step 2: generate MAS for each benchmark
bash generate_mas.sh

# Step 3: run evaluation
bash run_gpt4o.sh    # gpt-4o sub-agents
bash run_gpt5.sh     # gpt-5 sub-agents
bash run_gptoss.sh   # gpt-oss-120b sub-agents
```

#### MAS-Zero

```bash
cd evaluation/Automatic-MAS/MAS-Zero
pip install -r requirements.txt
export OPENAI_API_KEY="your-api-key"

# Search — design MAS for GPQA
python main_question.py \
    --dataset workflow_search/gpqa_diamond \
    --option plan \
    --meta_model gpt-4o_chatgpt \
    --node_model gpt-4o_chatgpt \
    --verifier_model gpt-4o_chatgpt \
    --blocks COT COT_SC Reflexion LLM_debate \
    --n_generation 5

# Verify — select best answer
python main_judge.py \
    --dataset gpqa_diamond \
    --judge_method self \
    --baseline workflow_search
```

Refer to `evaluation/Automatic-MAS/MAS-Zero/README.md` for commands covering other benchmarks (HLE-Math, SWE-Bench).
