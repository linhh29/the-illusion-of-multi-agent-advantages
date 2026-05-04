import asyncio
import json
import os
import re
import subprocess
from pathlib import Path

subprocess_semaphore = asyncio.Semaphore(20)


def extract_xml(text: str, tag: str) -> str:
    """Extracts XML tag content; used to pull <code> snippets if present."""
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1) if match else ""


def replace_hunk_headers_with_computed_counts(diff_text, hunk_infos):
    """Replace all hunk headers in the diff text using computed values."""
    lines = diff_text.strip().splitlines()
    hunk_header_re = re.compile(r"^@@ -(\d+),(\d+) \+(\d+),(\d+) @@")

    output_lines = []
    hunk_index = 0

    for line in lines:
        match = hunk_header_re.match(line)
        if match and hunk_index < len(hunk_infos):
            info = hunk_infos[hunk_index]
            hunk_index += 1

            original_start = info["original_start"]
            new_start = info["new_start"]
            computed_Y = info["computed_Y"]
            computed_B = info["computed_B"]

            new_header = f"@@ -{original_start},{computed_Y} +{new_start},{computed_B} @@"
            output_lines.append(new_header)
        else:
            output_lines.append(line)

    return "\n".join(output_lines)


def extract_all_hunks_and_recalculate_headers(diff_text):
    """Parse unified diff and recompute hunk header counts."""
    hunks = re.split(r"(?m)^@@ ", diff_text.strip())
    if not hunks:
        return []
    hunk_infos = []
    for h in hunks:
        if not h.strip():
            continue
        header_match = re.match(r"^-(\d+),(\d+) \+(\d+),(\d+) @@", h)
        if not header_match:
            continue
        original_start = int(header_match.group(1))
        new_start = int(header_match.group(3))
        original_lines = h.split("\n")[1:]
        plus_lines = sum(1 for line in original_lines if line.startswith("+"))
        minus_lines = sum(1 for line in original_lines if line.startswith("-"))
        context_lines = sum(
            1
            for line in original_lines
            if not line.startswith("+") and not line.startswith("-")
        )
        computed_Y = minus_lines + context_lines
        computed_B = plus_lines + context_lines
        hunk_infos.append(
            {
                "original_start": original_start,
                "new_start": new_start,
                "computed_Y": computed_Y,
                "computed_B": computed_B,
            }
        )
    return hunk_infos


def replace_hunk_headers_with_computed_counts_wrapper(diff_text):
    """Convenience: compute hunk info then replace headers."""
    infos = extract_all_hunks_and_recalculate_headers(diff_text)
    return replace_hunk_headers_with_computed_counts(diff_text, infos)


def normalize_diff_string(diff_text: str) -> str:
    """Normalize diff string for SWE-bench harness."""
    if not diff_text:
        return diff_text
    # ensure headers are consistent
    return replace_hunk_headers_with_computed_counts_wrapper(diff_text)


async def run_swebench_evaluation(
    judge_path: str,
    instance_id: str,
    extracted_answer: str,
    technique: str,
    solution_name: str,
    code_snippet=None,
    file_path=None,
):
    """Run SWE-bench harness evaluation for a single instance."""
    computed_header = extract_all_hunks_and_recalculate_headers(extracted_answer)
    extracted_answer = replace_hunk_headers_with_computed_counts(
        extracted_answer, computed_header
    )
    extracted_answer = normalize_diff_string(extracted_answer)

    prediction = {
        "instance_id": instance_id,
        "model_patch": extracted_answer,
        "model_name_or_path": "maas",
    }

    # Use absolute paths for all file operations
    judge_path_abs = Path(judge_path).resolve()
    file_path_abs = Path(file_path).resolve()
    
    os.makedirs(judge_path_abs / "results", exist_ok=True)
    os.makedirs(judge_path_abs / "reports", exist_ok=True)

    path_to_prediction = judge_path_abs / "results" / f"_{instance_id}_{solution_name}.json"
    with open(path_to_prediction, "w", encoding="utf-8") as f:
        json.dump([prediction], f, indent=4)

    num_workers = 1
    run_id = f"{instance_id}_{technique}_{solution_name}"
    run_id = (
        run_id.replace(" ", "_")
        .replace("-", "_")
        .replace("(", "_")
        .replace(")", "_")
    )

    # swebench 4.1.0 uses short args: -d, -p, -id, --report_dir
    cmd = [
        "python",
        "-m",
        "swebench.harness.run_evaluation",
        "-d", str(file_path_abs),
        "-p", str(path_to_prediction),
        "--max_workers", str(num_workers),
        "-id", run_id,
        "--report_dir", str(judge_path_abs / "reports"),
    ]

    TIMEOUT_SECONDS = 600  # 10 minutes per evaluation
    MAX_TIMEOUT_RETRIES = 5  # Max retries for timeout, then give up

    async with subprocess_semaphore:
        attempt = 0
        while True:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(judge_path_abs),  # Run in report directory (absolute path)
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=TIMEOUT_SECONDS
            )

            if process.returncode == 0:
                print(f"[SWE] ✓ {instance_id} evaluation completed")
                break
            else:
                stderr_text = stderr.decode() if stderr else ""
                # Patch error - no point retrying same bad patch
                if "Patch Apply Failed" in stderr_text or "FAILED" in stderr_text:
                    print(f"[SWE] ✗ {instance_id}: Patch failed, skipping")
                    return 0.0
                attempt += 1
                # Only print detailed error on first attempt
                if attempt == 1:
                    # Extract key error info
                    lines = stderr_text.strip().split('\n')
                    key_lines = [l for l in lines[-10:] if l.strip() and not l.startswith(' ')]
                    error_summary = '\n'.join(key_lines[-3:]) if key_lines else "Unknown error"
                    print(f"[SWE] ✗ {instance_id} failed:\n{error_summary}")
                else:
                    print(f"[SWE] {instance_id} retry {attempt}")
                await asyncio.sleep(2)

    # Report file name uses run_id: {instance_id}_{technique}_{solution_name} with all - replaced by _
    report_run_id = f"{instance_id}_{technique}_{solution_name}".replace("-", "_")
    
    # swebench may save report to reports/ or directly to judge_path
    result_path_reports = judge_path_abs / "reports" / f"maas.{report_run_id}.json"
    result_path_direct = judge_path_abs / f"maas.{report_run_id}.json"
    
    result_path = None
    if result_path_reports.exists():
        result_path = result_path_reports
    elif result_path_direct.exists():
        result_path = result_path_direct
    
    # if result_path:
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # resolved_instances can be a count or a list

    
    resolved = data.get("resolved_instances", 0)

    if data.get("error_instances") > 0:
        raise Exception(f"Error instances: {instance_id}: {data.get('error_instances')}")
        
    if isinstance(resolved, list):
        score = 1.0 if len(resolved) > 0 else 0.0
    else:
        score = 1.0 if float(resolved) > 0 else 0.0
    # else:
    #     print(f"[SWE] Warning: No result file found for {instance_id}")
    #     score = 0.0

    # clean container if harness created one
    instance_prefix = instance_id.replace("-", "_")
    container_name = f"sweb.eval.{instance_id}.{instance_prefix}__"
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, text=True)

    return score

