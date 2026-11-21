# 智能体与工具调用评测

OpenCompass 支持通过 [Lagent](https://github.com/InternLM/lagent) 与
[LangChain](https://github.com/langchain-ai/langchain) 封装器评测会调用外部工具
（例如 Python 解释器）的智能体。本页说明涉及的组件以及如何组装一份最小配置，
回应 issue #430 提出的文档诉求。

## 组件

| 层次 | 组件 | 源码 |
| --- | --- | --- |
| 模型封装 | `LagentAgent` | `opencompass/models/lagent.py` |
| 模型封装 | `LangchainAgent` | `opencompass/models/langchain.py` |
| 推理器 | `AgentInferencer` | `opencompass/openicl/icl_inferencer/icl_agent_inferencer.py` |
| 动作 | `PythonInterpreter` | `opencompass/lagent/actions/python_interpreter.py` |
| 智能体评测器 | `Gsm8kAgentEvaluator`、`MATHAgentEvaluator` | `opencompass/datasets/` |
| 示例配置 | GSM8K / MATH 智能体数据集 | `opencompass/configs/datasets/gsm8k/gsm8k_agent_gen_c3dff3.py`、`opencompass/configs/datasets/math/math_agent_gen_0c1b4e.py` |
| 可运行示例 | `eval_chat_agent.py` | `examples/eval_chat_agent.py` |

`LagentAgent` 依据 LLM 配置（`REGISTRY.build(llm)`）、可选的动作执行器
（`lagent.actions.ActionExecutor`）与可选的 `ReActProtocol` 构建 Lagent 智能体；
`LangchainAgent` 则调用 LangChain 的 `initialize_agent` 与 `load_tools`，并把中间
步骤以结构化 action 返回。两者都会让推理器把每一步记录为 `steps`，与最终
`prediction` 一起供智能体评测器打分。

## 最小配置

可运行参考是 `examples/eval_chat_agent.py`：它用 `ReAct` 智能体、`OpenAI` 模型、
`PythonInterpreter` 动作与 `ReActProtocol` 构建 `LagentAgent`，再配合 GSM8K、MATH
与 MathBench 智能体数据集使用 `AgentInferencer`。精简后的模型配置如下：

```python
from lagent import ReAct
from lagent.agents.react import ReActProtocol

from opencompass.lagent.actions.python_interpreter import PythonInterpreter
from opencompass.models.lagent import LagentAgent
from opencompass.models.openai_api import OpenAI

protocol = dict(
    type=ReActProtocol,
    action=dict(role='ACTION', begin='Tool:', end='\n'),
    action_input=dict(role='ARGS', begin='Tool Input:', end='\n'),
    finish=dict(role='FINISH', begin='FinalAnswer:', end='\n'),
)

models = [
    dict(
        abbr='gpt-react',
        type=LagentAgent,
        agent_type=ReAct,
        max_turn=3,
        llm=dict(type=OpenAI, path='<model>', key='ENV',
                 query_per_second=1, max_seq_len=4096),
        actions=[dict(type=PythonInterpreter)],
        protocol=protocol,
        batch_size=1,
    ),
]
```

数据集配置已绑定对应的推理器与评测器。例如 `gsm8k_agent_gen_c3dff3.py` 设置
`inferencer=dict(type=AgentInferencer)` 与
`evaluator=dict(type=Gsm8kAgentEvaluator)`；MATH 对应配置为
`math_agent_gen_0c1b4e.py`。

## 运行

按[快速开始](../get_started/quick_start.md)配置模型 API Key，然后在仓库根目录运行示例：

```bash
opencompass examples/eval_chat_agent.py --dry-run   # 查看解析后的配置
opencompass examples/eval_chat_agent.py             # 运行评测
```

也可以按[基于配置的评测](../user_guides/config_based_evaluation.md)流程，用配置名直接指定模型与数据集，无需编写脚本。

## 注意事项

- 智能体评测会通过所配置的动作执行模型生成的代码（例如 `PythonInterpreter`）。
  请在隔离环境中运行；基于 Docker 的执行方案参见[代码评测服务](code_eval.md#代码执行服务)。
- 部分历史智能体数据集配置带 `deprecated_` 前缀（例如
  `deprecated_gsm8k_agent_gen_be1606.py`）。编写新配置时请使用当前版本
  （`gsm8k_agent_gen_c3dff3.py`、`math_agent_gen_0c1b4e.py`）。
- `LangchainAgent` 需要安装 LangChain，它是可选依赖，仅在使用该封装器时安装。
