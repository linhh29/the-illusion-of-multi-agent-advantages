# DyLAN: Dynamic LLM-Agent Network

DyLAN constructs dynamic multi-agent architectures that adapt to each task query.

## Installation

```bash
cd code
pip install -r requirements.txt
```

## Configuration

```bash
export OPENAI_API_KEY="your-api-key"
```

## Usage

Place benchmark datasets in the appropriate directories, then run:

```bash
# GPQA
cd code/GPQA
python llmlp_listwise_gpqa.py <data_path>/gpqa_test.jsonl gpqa_test gpt-4o results "['Assistant', 'Scientist']"

# HLE-Math
cd code/HLEMATH
python llmlp_listwise_hlemath.py <data_path>/hlemath_test.jsonl hlemath_test gpt-4o results "['Mathematician', 'AlgebraExpert']"

# BrowseComp+
cd code/BCP
python llmlp_listwise_bcp.py <data_path>/bcp_test.jsonl bcp_test gpt-4o results "['Assistant', 'Scientist']"

# Stocks
cd code/STOCKS
bash exp_stocks_gpt4o.sh

# SWE-Bench
cd code/SWE
bash exp_swe.sh
```
