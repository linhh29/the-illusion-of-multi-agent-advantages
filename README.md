# Synthetic Multi-Hop Financial Reasoning (SMFR) Dataset

This repository contains the code for generating and evaluating the Synthetic Multi-Hop Financial Reasoning (SMFR) benchmark dataset.

## Overview

This benchmark evaluates multi-step reasoning and computational capabilities of large language models through synthetic stock trading problems. Each problem requires:

1. Parsing historical stock price data from multiple companies
2. Tracking investor transactions across multiple stocks
3. Computing portfolio profit/loss for different scenarios
4. Identifying optimal trading dates to achieve target returns
5. Comparing results across multiple investors

## Repository Structure

```
code/
├── src/                          # Dataset generation code
│   ├── generate_balanced_dataset.py   # Main generation script
│   ├── config_schema.py               # Configuration schemas
│   ├── problem_composer.py            # Problem composition engine
│   ├── task_base.py                   # Base classes for tasks
│   ├── data_sources.py                # Stock data fetching
│   └── tasks/                         # Task implementations
│       ├── stock_task.py              # Stock trading task
│       ├── math_task.py               # Math task (extensible)
│       └── coding_task.py             # Coding task (extensible)
│
└── evaluate/                     # Evaluation scripts
    ├── evaluate_stock_answer_code.py  # Main evaluation script
    ├── safe_code_executor.py          # Sandboxed code execution
    ├── requirements_evaluation.txt    # Evaluation dependencies
    └── README.md                      # Evaluation documentation
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd code

# Install dependencies for generation
pip install yfinance python-dateutil tqdm

# Install dependencies for evaluation
pip install -r evaluate/requirements_evaluation.txt
```

### Generate Dataset

```bash
cd src

# Generate dataset with 2 investors per problem
python generate_balanced_dataset.py 2

# Generate dataset with 3 investors per problem
python generate_balanced_dataset.py 3

# This creates: balanced_dataset_single_2.jsonl, balanced_dataset_single_3.jsonl, etc.
```

### Evaluate Model Outputs

```bash
cd evaluate

# Basic evaluation
python evaluate_stock_answer_code.py \
    --reference ../data/balanced_dataset_single_2_fixed.jsonl \
    --model-output your_model_outputs.jsonl

# With validation exclusion (required for n=2 and n=3)
python evaluate_stock_answer_code.py \
    --reference ../data/balanced_dataset_single_2_fixed.jsonl \
    --model-output your_model_outputs.jsonl \
    --validation ../data/balanced_dataset_single_validate_fixed.jsonl \
    --timeout 30
```

## Dataset Generation

### Generation Pipeline

The dataset generation follows a modular pipeline:

1. **Configuration** (`config_schema.py`): Define problem parameters
   - Number of investors
   - Question type (reverse_target_sell, reverse_target_buy)
   - Aggregation (earliest, latest)
   - Target profit percentage
   - Price type (Open, Close)
   - Breadth (number of companies) and depth (number of transactions)

2. **Data Fetching** (`data_sources.py`): Retrieve historical stock prices
   - Uses Yahoo Finance API (yfinance)
   - Fetches 30 days of OHLCV data
   - Supports multiple tickers

3. **Task Generation** (`tasks/stock_task.py`): Create problem instances
   - Generate haystack (stock price data with distractors)
   - Generate needles (investor transactions)
   - Compute answers (valid dates, optimal dates, final answer)
   - Format problem text and chain-of-thought

4. **Problem Composition** (`problem_composer.py`): Assemble complete problems
   - Coordinate multiple investors
   - Apply aggregation operations
   - Format final output

5. **Balanced Generation** (`generate_balanced_dataset.py`): Create dataset
   - Generate samples for all parameter combinations
   - Ensure balanced coverage
   - Retry logic to minimize "None" answers
   - Export to JSONL format

### Generation Parameters

The generation script accepts a single argument specifying the complexity level:

```bash
python generate_balanced_dataset.py <N>
```

Where `N` controls:
- Number of investors: N
- Breadth (companies): N
- Depth (transactions): N (always even)
- Number of distractors: N

### Customization

To customize generation parameters, edit `generate_balanced_dataset.py`:

```python
# Modify parameter ranges
question_types = ['reverse_target_sell', 'reverse_target_buy']
aggregations = ['earliest', 'latest']
price_types = ['Open', 'Close']
target_percentages = [round(0.1 + i * 0.15, 2) for i in range(13)]  # 10% to 200%

# Adjust samples per combination
samples_per_combination = 1  # Increase for more samples

# Control retry behavior
max_retries = 10  # Maximum retries if answer is None
max_none_samples = 0  # Maximum None answers to keep
```

### Adding New Stock Tickers

Edit `tasks/stock_task.py`:

```python
TICKER_LIST = ['AAPL', 'MSFT', 'GOOG', ...]  # Add your tickers
COMPANY_LIST = ['Apple', 'Microsoft', 'Alphabet', ...]  # Add company names
```

## Evaluation

### Evaluation Metrics

The evaluation script computes two metrics:

1. **Direct Answer Match**: Compares model's direct text answer with ground truth
   - Handles multiple output formats (string, list, dict)
   - Extracts investor names using context-aware parsing
   - Penalizes false ties (e.g., "Alice and Bob" when only "Alice" is correct)

2. **Code Execution Match**: Executes model-generated code and compares output
   - Runs code in sandboxed environment
   - Timeout protection (default 30s)
   - Restricted imports for security
   - Compares code output with ground truth

### Model Output Format

Your model outputs should be in JSONL format with the following structure:

```json
{
  "input": {
    "problem": "Here is some data on the stock prices..."
  },
  "output": {
    "answer": "Alice",
    "code": "def solve():\n    # Your code here\n    return {'answer': ['Alice']}"
  }
}
```

The `answer` field can be:
- A string: `"Alice"` or `"Alice and Bob"`
- A list: `["Alice", "Bob"]`
- A dict: `{"answer": ["Alice"]}`
- null or `"None"` for no valid answer

The `code` field is optional. If provided, it must define a `solve()` function that returns a dict with an `"answer"` key.

### Answer Matching Logic

The evaluator uses investor-name-aware extraction:

| Model Output | Extracted | Matches `["Alice"]` |
|--------------|-----------|---------------------|
| `"Alice"` | `["Alice"]` | ✅ Yes |
| `["Alice"]` | `["Alice"]` | ✅ Yes |
| `"Alice and Bob"` | `["Alice", "Bob"]` | ❌ No (false tie) |
| `"None"` | `None` | Only if GT is `[]` |
| `[]` | `None` | Only if GT is `[]` |

### Safe Code Execution

The `SafeCodeExecutor` provides sandboxed execution:

**Security Features:**
- Timeout limits (configurable, default 30s)
- Restricted imports (whitelist: `math`, `datetime`, `collections`, etc.)
- No file system access
- No network access
- Isolated namespace
- Restricted built-ins (`open`, `eval`, `exec` disabled)

**Allowed Imports:**
```python
# Math and utilities
math, random, itertools, functools, operator, collections

# Date/time
datetime, calendar, time

# Data structures
heapq, bisect, array, copy, enum

# String processing
string, re, json
```

**Disallowed Imports:**
```python
# System access
os, sys, subprocess, socket

# Network
urllib, requests, http, ftplib

# File I/O
pickle, shelve, dbm

# Introspection
inspect, dis, gc, pdb
```

### Evaluation Output

```
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

## Prompt Engineering

### Recommended Response Schema

Use structured outputs to minimize answer format variation:

```python
from pydantic import BaseModel, Field

class StockAnswer(BaseModel):
    analysis: str = Field(..., description="Step-by-step reasoning")
    answer: str = Field(..., description="Winning investor name(s), or 'None'")
    code: str = Field(..., description="Python solve() function with all data embedded")
```

### Handling Ties

Instruct models to return lists for ties:

```
If two or more investors tie for the earliest/latest date, return all tied 
names as a Python list, e.g. ["Alice", "Bob"]. If no investor achieves the 
target, return None.
```

### Code Generation Guidelines

For models that generate code:

```
Generate a Python function named solve() that:
1. Embeds all required data directly in the code (no external inputs)
2. Returns a dictionary with at minimum {"answer": [...]}
3. Uses only standard library imports (math, datetime, collections, etc.)
4. Completes execution within 30 seconds
```

## Extensibility

The codebase is designed for extensibility:

### Adding New Task Types

1. Create a new task class in `src/tasks/`:

```python
from task_base import BaseTask

class MyTask(BaseTask):
    def generate_haystack(self, seed):
        # Generate distractor data
        pass
    
    def generate_needles(self, haystack, seed, count):
        # Generate problem instances
        pass
    
    def compute_answer(self, needle):
        # Compute ground truth answer
        pass
    
    def format_problem(self, haystack, needles, template, extra_vars):
        # Format problem text
        pass
    
    def format_cot(self, needles, answers):
        # Format chain-of-thought
        pass
    
    def get_task_type(self):
        return "my_task"
```

2. Register in `problem_composer.py`:

```python
def _create_task(self, task_config):
    if task_type == 'my_task':
        return MyTask(config_dict)
```

3. Add configuration in `config_schema.py`:

```python
def my_task_config(**kwargs) -> PipelineConfig:
    # Define configuration
    pass
```

### Adding New Data Sources

1. Create a data source class in `src/data_sources.py`:

```python
from task_base import BaseDataSource

class MyDataSource(BaseDataSource):
    def fetch(self, params):
        # Fetch data from external source
        pass
    
    def serialize(self, data):
        # Serialize for storage
        pass
    
    def update(self, serialized_data):
        # Update with latest values
        pass
    
    def get_source_type(self):
        return "my_source"
```

2. Register in the factory function:

```python
def create_data_source(source_type: str):
    sources = {
        'stock': StockDataSource,
        'my_source': MyDataSource,
    }
    return sources[source_type]()
```


## Acknowledgments

This dataset uses historical stock price data from Yahoo Finance via the yfinance library.
