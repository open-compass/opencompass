# noqa: F401, F403
export HF_EVALUATE_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_CACHE=./ # your cache
cd ./ # your opencompass path
conda activate opencompass
python run.py configs/subjective/eval_subjective_alignbench.py --mode infer --reuse latest  &
python run.py configs/subjective/eval_subjective_compassarena.py --mode all --reuse latest  &
python run.py configs/subjective/eval_subjective_mtbench.py --mode all --reuse latest  &
python run.py configs/subjective/eval_subjective_alpacaeval.py --mode all --reuse latest
