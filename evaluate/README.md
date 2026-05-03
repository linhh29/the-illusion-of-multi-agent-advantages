## Evaluation

### Quick Start

```bash
python evaluate_stock_answer_code.py \
    --reference  balanced_dataset_single_2_fixed.jsonl \
    --model-output  my_model_outputs.jsonl \
    --validation  balanced_dataset_single_validate.jsonl \
    --timeout 30
```

Pass `--validation` whenever evaluating the n=2 or n=3 datasets — it excludes the held-out validation split from scoring. The `--timeout` flag controls per-sample code execution time (default: 30s).

---

### Overview

`evaluate_stock_answer_code.py` evaluates model outputs against the fixed reference datasets for the stock trading analysis benchmark.

Ground truth is always sourced from the **fixed** reference files (`balanced_dataset_single_N_fixed.jsonl`), matched to model outputs by **problem text** (not by index or ID). Validation samples are excluded automatically when a validation file is provided.

---

### Answer Matching

Answers are compared using investor-name-aware extraction, which handles all common model output formats robustly:

| Model output | Extracted names |
|---|---|
| `"Alice"` | `{Alice}` |
| `["Alice", "Bob"]` | `{Alice, Bob}` |
| `"Alice and Bob"` | `{Alice, Bob}` |
| `'["Alice", "Bob"]'` (JSON string) | `{Alice, Bob}` |
| `{"answer": ["Alice"]}` | `{Alice}` |
| `"None"` / `null` | no answer |

**False ties are penalised**: if the model returns `"Rachel and Patricia"` but only `"Patricia"` is correct, the extracted set `{Rachel, Patricia}` does not match `{Patricia}` → marked wrong.

---

### Usage

```bash
python evaluate_stock_answer_code.py \
    --reference  balanced_dataset_single_2_fixed.jsonl \
    --model-output  my_model_outputs.jsonl
```

With validation exclusion (required for n=2 and n=3 datasets):

```bash
python evaluate_stock_answer_code.py \
    --reference  balanced_dataset_single_2_fixed.jsonl \
    --model-output  my_model_outputs.jsonl \
    --validation  balanced_dataset_single_validate.jsonl \
    --timeout 30
```

---

### Arguments

| Argument | Required | Description |
|---|---|---|
| `--reference` | yes | Fixed reference JSONL (`balanced_dataset_single_N_fixed.jsonl`) |
| `--model-output` | yes | Model output JSONL (see format below) |
| `--validation` | no | Validation JSONL — matching problems are excluded from scoring |
| `--timeout` | no | Code execution timeout in seconds (default: 30) |

---

### Input Formats

**Reference file** (`balanced_dataset_single_N_fixed.jsonl`) — one sample per line:
```json
{
  "problem": "Here is some data on the stock prices ...",
  "answer": {
    "investor_dates": {"Alice": ["January 5, 2026", ...], "Bob": [...]},
    "comparison": {"Alice": "January 5, 2026", "Bob": "January 8, 2026"},
    "answer": ["Alice"]
  }
}
```

**Model output file** — one sample per line:
```json
{
  "input":  {"problem": "Here is some data on the stock prices ..."},
  "output": {
    "answer": "Alice",
    "code":   "def solve(): ..."
  }
}
```

The `"answer"` field in `"output"` may be a string, list, dict, or null (all handled).  
The `"code"` field is optional. If present, it must define a `solve()` function returning a dict with an `"answer"` key.

---

### Prompt Guidance

Use structured outputs to minimise answer format variation. Example response schema:

```python
from pydantic import BaseModel, Field

class StockAnswer(BaseModel):
    analysis: str = Field(..., description="Step-by-step reasoning")
    answer: str   = Field(..., description="Winning investor name(s), or 'None'")
    code: str     = Field(..., description="Python solve() function with all data embedded")
```

For multi-investor ties, instruct the model to return a list:

```
If two or more investors tie for the earliest/latest date, return all tied names as a
Python list, e.g. ["Alice", "Bob"]. If no investor achieves the target, return None.
```

---

### Example Output

```
Loading reference data from: balanced_dataset_single_2_fixed.jsonl
  96 reference samples loaded
  16 validation problems will be excluded

Loading model outputs from: my_model_outputs.jsonl
  96 records loaded

============================================================
EVALUATION RESULTS
============================================================
Total evaluated : 80
Skipped         : 16  (not in reference / validation set)

Direct Answer:
  Correct : 68/80 (85.0%)
  Wrong   : 12/80 (15.0%)

Code Execution:
  Correct          : 70/80 (87.5%)
  Execution errors : 3/80 (3.8%)
============================================================
```

---

### Safe Code Execution

The `SafeCodeExecutor` runs model-generated code in a sandboxed environment:

- Timeout protection (configurable, default 30s)
- Restricted imports (whitelist: `math`, `datetime`, `collections`, etc.)
- No filesystem or network access
- Isolated namespace

Model-generated `solve()` functions should embed all required data directly in the code and return a dict containing at minimum `{"answer": ...}`.
