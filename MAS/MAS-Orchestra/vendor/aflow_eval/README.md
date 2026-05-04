Vendored AFlow evaluation code for standalone MAS-Orchestra runs (parity with `MAS-Eval-Foundation/AFlow`).

- `benchmarks/benchmark.py` is a **minimal** `BaseBenchmark` (scoring-only) without `aiofiles`/`pandas` dependencies required by full upstream `benchmarks/benchmark.py`.
- `benchmarks/swe.py` import of `swe_utils` uses `from benchmarks.swe_utils import` for a single `sys.path` entry.
- `config/config2.yaml` ships without API secrets; set `OPENAI_API_KEY` or edit keys locally (do not commit).
- `scripts/async_llm.LLMsConfig.default()` resolves `config/config2.yaml` from the package directory first, so BCP grading works even when the process cwd is not `vendor/aflow_eval`.
