import json
import sys
import os
import re

def eval_swe(result_file):
    """Evaluate SWE results from txt file."""
    with open(result_file, 'r') as f:
        content = f.read()
        lines = content.split('\n')
    
    if len(lines) < 1:
        print("Error: Result file is empty or incomplete")
        return
    
    # First line contains API call counts
    if len(lines) >= 1:
        resp_cnt_str = lines[0].strip()
        try:
            parts = resp_cnt_str.split()
            total_calls = int(parts[0])
            avg_calls = float(parts[1])
            print(f"Total API calls: {total_calls}")
            print(f"Average API calls per instance: {avg_calls:.2f}")
        except:
            print("Error parsing API call counts")
    
    # Check for docker evaluation results
    if "Docker Evaluation Results:" in content:
        print("\n" + "="*60)
        print("Docker Evaluation Results Found:")
        print("="*60)
        
        # Extract passed instances
        passed_match = re.search(r'Passed instances:\s*(\d+)/(\d+)\s*\(([\d.]+)%\)', content)
        if passed_match:
            passed = int(passed_match.group(1))
            total = int(passed_match.group(2))
            percentage = float(passed_match.group(3))
            print(f"Passed instances: {passed}/{total} ({percentage:.2f}%)")
        
        # Extract total score
        total_score_match = re.search(r'Total score:\s*([\d.]+)', content)
        if total_score_match:
            total_score = float(total_score_match.group(1))
            print(f"Total score: {total_score:.2f}")
        
        # Extract average score
        avg_score_match = re.search(r'Average score:\s*([\d.]+)', content)
        if avg_score_match:
            avg_score = float(avg_score_match.group(1))
            print(f"Average score: {avg_score:.4f}")
        
        print("="*60)
    else:
        print("\nNote: Docker evaluation results not found in file.")
        print("To enable docker evaluation, provide JUDGE_PATH and FILE_PATH parameters when running the script.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval_swe.py <result_file>")
        sys.exit(1)
    
    eval_swe(sys.argv[1])

