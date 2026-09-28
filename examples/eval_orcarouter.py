# Evaluate models through the OrcaRouter AI gateway.
#
# OrcaRouter is an OpenAI-compatible AI gateway for models and agents. It
# exposes a provider/model namespace (e.g. `orcarouter/free`,
# `orcarouter/fusion`) alongside many third-party models, with adaptive
# routing, automatic failover, zero-markup inference, observability,
# guardrails, and agent-tool governance on the same endpoint.
#
# Before running, set your key:
#   export ORCAROUTER_API_KEY=your_key
#
# The full model list is available at:
#   https://api.orcarouter.ai/v1/models
# and at https://www.orcarouter.ai

from opencompass.models import OrcaRouterAPI

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
], )

models = [
    dict(
        abbr='OrcaRouter-Free',
        type=OrcaRouterAPI,
        path='orcarouter/free',  # Any model id exposed by the gateway
        key='ENV',  # Read from $ORCAROUTER_API_KEY
        meta_template=api_meta_template,
        query_per_second=1,
        max_out_len=512,
        max_seq_len=16384,
        batch_size=1,
    ),
]
