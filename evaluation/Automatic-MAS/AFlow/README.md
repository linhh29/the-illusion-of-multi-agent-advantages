# AFlow: Automated Agentic Workflow Generation

AFlow automatically generates and optimizes multi-agent workflows using a search-based approach.

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config/config2.yaml` to set your API key:

```yaml
models:
  "gpt-4o":
    api_type: "openai"
    base_url: "https://api.openai.com"
    api_key: "your-api-key"
```

## Usage

Place benchmark datasets (gpqa, hlemath, bcp, swe, smfr) in `data/datasets/`, then run:

```bash
# GPQA
bash run_gpqa.sh

# HLE-Math
bash run_hlemath.sh

# BrowseComp+
bash run_bcp.sh

# Smfrs
bash run_smfr.sh

# SWE-Bench
bash run_swe.sh
```
