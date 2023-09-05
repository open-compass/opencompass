# Llama Adapter V2

### Prepare the environment

```sh
cd opencompass/multimodal/models/llama_adapter_v2_multimodal
git clone https://github.com/OpenGVLab/LLaMA-Adapter.git
```

### Start evaluation

#### Slurm

```sh
cd $root
python run.py configs/multimodal/tasks.py --mm-eval --slurm -p $PARTITION
```

#### PyTorch

```sh
cd $root
python run.py configs/multimodal/tasks.py --mm-eval
```