# Agent and Tool-Use Evaluation

OpenCompass can evaluate agents that call external tools (for example a Python
interpreter) through the [Lagent](https://github.com/InternLM/lagent) and
[LangChain](https://github.com/langchain-ai/langchain) wrappers. This page
explains which components are involved and how to assemble a minimal
configuration. It addresses the documentation request in issue #430.

## Components

| Layer | Component | Source |
| --- | --- | --- |
| Model wrapper | `LagentAgent` | `opencompass/models/lagent.py` |
| Model wrapper | `LangchainAgent` | `opencompass/models/langchain.py` |
| Inferencer | `AgentInferencer` | `opencompass/openicl/icl_inferencer/icl_agent_inferencer.py` |
| Action | `PythonInterpreter` | `opencompass/lagent/actions/python_interpreter.py` |
| Agent evaluator | `Gsm8kAgentEvaluator`, `MATHAgentEvaluator` | `opencompass/datasets/` |
| Example config | GSM8K / MATH agent datasets | `opencompass/configs/datasets/gsm8k/gsm8k_agent_gen_c3dff3.py`, `opencompass/configs/datasets/math/math_agent_gen_0c1b4e.py` |
| Runnable example | `eval_chat_agent.py` | `examples/eval_chat_agent.py` |

`LagentAgent` builds a Lagent agent from an LLM config (`REGISTRY.build(llm)`),
an optional action executor (`lagent.actions.ActionExecutor`) and an optional
`ReActProtocol`. `LangchainAgent` instead calls LangChain's `initialize_agent`
with `load_tools` and returns the intermediate steps as structured actions. In
both cases the inferencer records each step as a `steps` entry alongside the
final `prediction`, which is what the agent evaluators score.

## Minimal configuration

The runnable reference is `examples/eval_chat_agent.py`. It builds a
`LagentAgent` with a `ReAct` agent, an `OpenAI` LLM, a `PythonInterpreter`
action and a `ReActProtocol`, then uses `AgentInferencer` with the GSM8K, MATH
and MathBench agent datasets. A trimmed-down model block looks like this:

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

The dataset configs already bind the matching inferencer and evaluator. For
example `gsm8k_agent_gen_c3dff3.py` sets `inferencer=dict(type=AgentInferencer)`
and `evaluator=dict(type=Gsm8kAgentEvaluator)`; the MATH equivalent is
`math_agent_gen_0c1b4e.py`.

## Running

Configure the model API key as described in the
[quick start](../get_started/quick_start.md), then run the example from the
repository root:

```bash
opencompass examples/eval_chat_agent.py --dry-run   # inspect the resolved config
opencompass examples/eval_chat_agent.py             # run the evaluation
```

You can also select a model and dataset by config name without writing a
script, following the [configuration-based evaluation](../user_guides/config_based_evaluation.md)
workflow.

## Notes

- Agent evaluation executes model-generated code through the configured
  action (for example `PythonInterpreter`). Run it in an isolated environment;
  see the [code evaluation service](code_eval.md#code-execution-service) for
  the Docker-based execution option.
- Some historical agent dataset configs are prefixed `deprecated_` (for
  example `deprecated_gsm8k_agent_gen_be1606.py`). Prefer the current ones
  (`gsm8k_agent_gen_c3dff3.py`, `math_agent_gen_0c1b4e.py`) when writing new
  configurations.
- `LangchainAgent` requires LangChain to be installed; it is an optional
  dependency, so install it only when you use that wrapper.
