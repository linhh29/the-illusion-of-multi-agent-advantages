# MAS-Orchestra: Multi-Agent Reasoning Orchestration

MAS-Orchestra generates and evaluates multi-agent systems using a trained orchestrator model.

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

```bash
export OPENAI_API_KEY="your-api-key"
export TOGETHER_API_KEY="your-together-key"  # For gpt-oss-120b
```

## Usage

Place benchmark datasets (gpqa, hlemath, bcp, swe, stocks) in `data/datasets/`.

### Step 1: Deploy Orchestrator

```bash
bash vllm_deploy.sh
```

This serves the orchestrator model at `http://localhost:8000`.

### Step 2: Generate MAS

```bash
bash generate_mas.sh
```

This generates multi-agent systems for each benchmark.

### Step 3: Evaluate MAS

```bash
# With gpt-4o sub-agents
bash run_gpt4o.sh

# With gpt-5 sub-agents
bash run_gpt5.sh

# With gpt-oss-120b sub-agents
bash run_gptoss.sh
```

Results are saved to `results_run*/`.
