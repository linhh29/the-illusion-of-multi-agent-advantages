
# pip install -r requirements.txt


DATASET=STOCKS
MODEL=gpt-4o

rm ./logs/${DATASET}_${MODEL}.txt

python run.py --initial_round 1 --dataset ${DATASET} --opt_model_name ${MODEL} --exec_model_name ${MODEL} >> ./logs/${DATASET}_${MODEL}.txt