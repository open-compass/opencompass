# Notifications and Task Monitoring

OpenCompass can report task status through a Lark bot. A webhook is a credential and must not be committed to the repository.

Declare it in a private configuration file:

```python
lark_bot_url = 'YOUR_WEBHOOK_URL'
```

Import it with `read_base()` in the experiment configuration:

```python
from mmengine.config import read_base

with read_base():
    from .secrets import lark_bot_url
```

Explicitly enable notifications when starting the run:

```bash
opencompass my_eval.py --lark
```

Notifications reflect scheduling state only and do not replace result acceptance checks. After a task finishes, inspect the logs, prediction count, evaluation results, and summary files. When using concurrent evaluation watching, also monitor the heartbeat and inference-status files in the work directory.
