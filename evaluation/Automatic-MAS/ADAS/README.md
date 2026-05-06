# ADAS: Automated Design of Agentic Systems

ADAS automatically searches for and designs multi-agent systems. A meta agent iteratively programs new agent designs in code and evaluates them.

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Set your API key:

```bash
export OPENAI_API_KEY="your-api-key"
```

## Usage

Place your benchmark datasets (gpqa, hlemath, bcp, swe, smfr) in the `dataset/` directory, then run:

```bash
# GPQA
bash run_gpqa.sh

# HLE-Math
bash run_hle.sh

# BrowseComp+
bash run_bcp.sh

# Smfrs
bash run_smfr.sh

# SWE-Bench
bash run_swe.sh
```

Each script runs the meta agent search for the corresponding domain.
