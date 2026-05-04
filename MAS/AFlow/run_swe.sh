
pip install -r requirements.txt
# newgrp docker
# conda activate mas_eval

DATASET=SWE
MODEL=gpt-4o

mkdir ./logs
rm ./logs/${DATASET}_${MODEL}.txt
rm -rf ./logs/run_evaluation

python run.py --initial_round 1 --dataset ${DATASET} --opt_model_name ${MODEL} --exec_model_name ${MODEL} >> ./logs/${DATASET}_${MODEL}.txt