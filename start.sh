cd ../
git lfs install
git clone https://huggingface.co/datasets/LLaMAX/BenchMAX_Function_Completion
mkdir -p ~/.cache/opencompass/data/xhumaneval_plus
mv BenchMAX_Function_Completion/* ~/.cache/opencompass/data/xhumaneval_plus/
cd opencompass