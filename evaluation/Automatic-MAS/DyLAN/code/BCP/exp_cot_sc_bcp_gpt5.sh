#!/bin/bash
export OPENAI_API_KEY=
export MAX_CONCURRENT=50
number=5
MODEL=gpt-5
# MODEL=gpt-5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAS_EVAL_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
dir="$MAS_EVAL_ROOT/AFlow/data/datasets"
exp_name=bcp_downsampled_${MODEL}

ROLES="['Knowledge Researcher', 'Cultural Historian', 'Information Analyst', 'Assistant']"

# Run the evaluation 3 times with different run IDs
for run_num in 2
do
    echo "=========================================="
    echo "Starting CoT-SC Run $run_num of 3"
    echo "=========================================="
    
    # Create run-specific output directory
    OUTPUT_DIR="${exp_name}_KnowledgeResearcher_CulturalHistorian_InformationAnalyst_Assistant_run${run_num}"
    mkdir -p "$OUTPUT_DIR"
    
    for file in "$dir"/bcp_test.jsonl
    do
        # Check if file exists
        if [ ! -f "$file" ]; then
            echo "File not found: $file"
            continue
        fi
        
        filename=$(basename -- "$file")
        extension="${filename##*.}"
        filename="${filename%.*}"

        RES_NAME="$OUTPUT_DIR/${filename}_cot_sc.txt"
        LOG_NAME="$OUTPUT_DIR/${filename}_cot_sc.log"

        # check if RES_NAME exists and has content (check for final evaluation section)
        # if [ -f "$RES_NAME" ]; then
        #     if grep -q "FINAL EVALUATION RESULTS" "$RES_NAME"; then
        #         echo "Skipping $filename run $run_num (already completed)"
        #         continue
        #     fi
        # fi

        echo "Processing $filename (CoT-SC Run $run_num)..."
        # run python script with run_id parameter
        python cot_sc_bcp.py "$file" "$filename" "$MODEL" "$exp_name" "$ROLES" "$run_num" "$number" > "$LOG_NAME" 2>&1
    done

    wait
    echo "Run $run_num completed!"
    echo ""
done

echo "All 3 CoT-SC runs completed!"

