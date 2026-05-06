
pip install -r requirements.txt


DATASET=BCP
MODEL=gpt-4o

mkdir ./logs
rm ./logs/${DATASET}_${MODEL}.txt

python run.py --initial_round 1 --dataset ${DATASET} --opt_model_name ${MODEL} --exec_model_name ${MODEL} >> ./logs/${DATASET}_${MODEL}.txt