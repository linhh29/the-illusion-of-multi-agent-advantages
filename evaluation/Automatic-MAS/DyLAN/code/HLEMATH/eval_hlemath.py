import json
import sys
import os

def eval_hlemath(result_file):
    """Evaluate HLEMATH results from txt file."""
    with open(result_file, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 1:
        print("Error: Result file is empty or incomplete")
        return
    
    # First line contains accuracy list
    accs_str = lines[0].strip()
    try:
        # Parse the accuracy list
        accs = eval(accs_str.split()[0])  # Get the list part before the accuracy value
        accuracy = float(accs_str.split()[-1])
        print(f"Accuracy: {accuracy:.4f} ({sum(accs)}/{len(accs)})")
    except:
        print("Error parsing accuracy")
    
    if len(lines) >= 2:
        resp_cnt_str = lines[1].strip()
        try:
            parts = resp_cnt_str.split()
            total_calls = int(parts[0])
            avg_calls = float(parts[1])
            print(f"Total API calls: {total_calls}")
            print(f"Average API calls per question: {avg_calls:.2f}")
        except:
            print("Error parsing API call counts")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval_hlemath.py <result_file>")
        sys.exit(1)
    
    eval_hlemath(sys.argv[1])

