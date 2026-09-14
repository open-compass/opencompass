# Prompt Preview and Debugging

Do not immediately run a full benchmark after changing a prompt. First preview the input actually received by the model, then validate generation and answer extraction on a small sample.

## Prompt Viewer

```bash
python tools/prompt_viewer.py my_eval.py -n -c 3
```

- `-n`: non-interactive mode; select the first model and dataset.
- `-c`: number of samples to print.
- `-a`: inspect every model-dataset combination.
- `-p PATTERN`: select only Datasets whose abbreviation matches the pattern.

When given a complete experiment configuration, the tool builds the tokenizer, displays input after model-template processing, and reports token counts. When given only a dataset configuration, it normally inspects only the Dataset-side prompt.

## Exporting Messages Only

If the evaluation backend or template is unsuitable for Prompt Viewer, make the inference task export messages only:

```bash
opencompass my_eval.py \
    --dump-only-message-path /tmp/opencompass-messages \
    --debug
```

This mode is for inspecting construction results and must not be treated as formal predictions. Use a dedicated export directory to avoid mixing it with experiment results.

## Checklist

- Every field placeholder has been replaced.
- System/user/assistant order is as intended.
- Dataset and model templates do not duplicate instructions.
- Few-shot examples do not leak test answers.
- Final input fits the model context, and truncation does not remove the question.
- Generation prompt, stop words, and answer format agree.
- API and local models receive the same semantics.

See [Prompt Templates](raw_prompt_template.md) for RawPromptTemplate and [Model-Side Conversation Template Protocol](meta_template.md) for the model protocol.
