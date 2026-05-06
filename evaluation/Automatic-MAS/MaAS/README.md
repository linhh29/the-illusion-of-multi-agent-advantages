# MaAS: Multi-agent Architecture Search

MaAS optimizes a probabilistic distribution of multi-agent architectures (agentic supernet).

## Installation

```bash
pip install -e .
```

## Configuration

Create `~/.metagpt/config2.yaml`:

```yaml
llm:
  api_type: "openai"
  model: "gpt-4o-mini"
  base_url: "https://api.openai.com"
  api_key: "your-api-key"
```

## Usage

Place benchmark datasets (gpqa, hlemath, bcp, swe, smfr) in `maas/ext/maas/data/`, then run:

```bash
# Train
python -m examples.maas.optimize --dataset GPQA --round 1 --sample 4

# Test
python -m examples.maas.optimize --dataset GPQA --round 1 --sample 4 --is_test True

# Smfrs evaluation
bash run_smfr_test_3x.sh
```
