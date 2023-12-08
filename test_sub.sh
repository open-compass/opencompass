export HF_EVALUATE_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_CACHE=/mnt/petrelfs/share_data/basemodel/checkpoints/llm/hf_hub
cd ~/eval_PR/opencompass
conda activate opencompass
python run.py configs/subjective.py --debug --mode infer --reuse latest

#python run.py configs/subjective_infer.py --slurm -p llmeval -q auto --max-num-workers 32 --debug --mode infer
#python run.py configs/infer_sub.py --slurm -p llmeval -q auto --max-num-workers 32 --debug --mode infer #--reuse latest
#python run.py configs/eval_sub.py --slurm -p llmeval -q auto --max-num-workers 32 --debug #--mode eval --reuse latest


####踩个坑，需要去ds1000安装对应的requirements才行