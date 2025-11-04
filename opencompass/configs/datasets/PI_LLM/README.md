# 🧠 PI-LLM: Context (Proactive) Interference in Large Language Models

**PI-LLM** is a benchmark dataset for evaluating context (proactive) interference effects in LLMs through simple key-value tracking tasks. Tests and shows LLM working memory limitations beyond context length.

**📄 ICML 2025 Long-Context Foundation Models Workshop - Accepted**

## 🎯 Core Challenge

PI-LLM reveals a **fundamental limitation**: ALL tested SOTA LLMs (GPT-5, Grok-4, Gemini, DeepSeek, etc.) consistently fail at a task that is **super easy for humans** (99%+ accuracy). This pinpoints the core challenge of **Multi-Round Co-Reference (MRCR)**.

### Task Example:
```
Key1: Value_1
Key1: Value_2  
Key1: Value_N

Question: What is the current value (last value) for Key1?
Expected: Value_N
```

**Result**: As N increases, LLMs increasingly answer with earlier values (Value_1, Value_2, etc.) instead of the correct Value_N.

## 📊 Four Experiment Types

### 1. **exp_updates** - Main Updates Test
- **Focus**: Memory decay over increasing update sequences
- **Variable**: Number of updates per key (n_updates: 2-400)
- **Mode**: Single difficulty gradient with randomized order
- **Finding**: Log-linear decline in accuracy

### 2. **exp_keys** - Concurrent Keys Test  
- **Focus**: Parallel interference from concurrent keys
- **Variable**: Number of concurrent keys (n_keys)
- **Modes**: Easy (n_updates=125) + Hard (n_updates=350)
- **Finding**: Performance degrades with more concurrent keys

### 3. **exp_valuelength** - Value Complexity Test
- **Focus**: Value complexity effects on memory retrieval
- **Variable**: Length of values (1-40 characters)  
- **Modes**: Easy (n_updates=4) + Hard (n_updates=20)
- **Finding**: Longer values cause rapid performance decline

### 4. **exp_sequential** - Sequential Order Test
- **Focus**: Sequential vs randomized interference patterns
- **Variable**: Number of updates (2-400)
- **Mode**: Non-randomized, strictly sequential blocks
- **Finding**: Even simpler patterns cause step-like failures

## 🏆 Evaluation Metrics

### Primary Metric: **AUC Score (log base 1.5)**
- Weighs harder tasks (more updates) more heavily
- Better model differentiation for high-performing models
- **Use this for model ranking and leaderboards**

### Reference Metric: **Average Accuracy**
- Simple interpretability
- Treats all difficulties equally

### Expected Output Formats:

**Single-Mode Experiments** (exp_updates, exp_sequential):
```python
{
    'avg_accuracy': 0.600,     # Reference
    'auc_log1.5': 0.412,      # 🏆 PRIMARY METRIC
    'total_samples': 100
}
```

**Two-Mode Experiments** (exp_keys, exp_valuelength):
```python
{
    'avg_accuracy': 0.600,        # Reference  
    'auc_log1.5': 0.576,         # 🏆 PRIMARY combined metric
    'auc_log1.5_easy': 0.850,    # Easy mode performance
    'auc_log1.5_hard': 0.350,    # Hard mode performance
    'total_samples': 150
}
```

## 🚀 Quick Usage

### Evaluate All Experiments:
```bash
python run.py --datasets pi_llm_exp_updates pi_llm_exp_keys pi_llm_exp_valuelength pi_llm_exp_sequential --models hf_llama2_7b
```

### Evaluate Single Experiment:
```bash
python run.py --datasets pi_llm_exp_updates --models hf_llama2_7b
```

### Custom Sample Limits:
```bash
python run.py --datasets pi_llm_exp_updates --max-samples 20
```

## 🔬 Research Impact

### Key Findings:
- **Universal Pattern**: All SOTA models show similar log-linear decline
- **Human Superiority**: Humans achieve 99%+ accuracy on same tasks  
- **Context Independence**: Failures occur within models' context windows
- **Interference > Length**: Memory limits due to interference, not context length
- **Cross-Model Consistency**: Same patterns from GPT-2 to GPT-5

### Industry Adoption:
> **Adoption Note**: This dataset is integrated into a **top-5 open-weight model company's internal benchmarking framework** for assessing tracking capacity and context interference in agents.

## 🔗 Resources

- **Homepage**: https://sites.google.com/view/cog4llm
- **Paper**: https://arxiv.org/abs/2506.08184
- **HuggingFace Dataset**: https://huggingface.co/datasets/giantfish-fly/pi-llm
- **OpenAI MRCR Dataset**: https://huggingface.co/datasets/openai/mrcr
- **DeepMind MRCR Paper**: https://arxiv.org/pdf/2409.12640v2

## 👥 Authors

**Chupei Wang** - University of Virginia, Physics Department  
📫 cw4bb@virginia.edu

**Jiaqiu Vince Sun** - PhD Candidate, NYU Center for Neuroscience  
📫 vince.sun@nyu.edu

## 📚 Citation

```bibtex
@misc{wang2025unableforgetproactiveinterference,
  title={Unable to Forget: Proactive Interference Reveals Working Memory Limits in LLMs Beyond Context Length}, 
  author={Chupei Wang and Jiaqiu Vince Sun},
  year={2025},
  eprint={2506.08184},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2506.08184}, 
}
```

## 🧪 Dataset Details

- **Total Samples**: 740 (580 core + 160 sequential_additional)
- **Context Length**: 5-25k tokens (well within model limits)
- **Languages**: English
- **License**: MIT
- **Format**: Direct HuggingFace loading (Parquet)
- **Categories**: Memory, Retrieval, Context Interference