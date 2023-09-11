# FAQ

## Network

**Q1**: My tasks failed with error: `('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer'))` or `urllib3.exceptions.MaxRetryError: HTTPSConnectionPool(host='cdn-lfs.huggingface.co', port=443)`

A: Because of HuggingFace's implementation, OpenCompass requires network (especially the connection to HuggingFace) for the first time it loads some datasets and models. Additionally, it connects to HuggingFace each time it is launched. For a successful run, you may:

- Work behind a proxy by specifying the environment variables `http_proxy` and `https_proxy`;
- Use the cache files from other machines. You may first run the experiment on a machine that has access to the Internet, and then copy the cached files to the offline one. The cached files are located at `~/.cache/huggingface/` by default ([doc](https://huggingface.co/docs/datasets/cache#cache-directory)). When the cached files are ready, you can start the evaluation in offline mode:
  ```python
  HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run.py ...
  ```
  With which no more network connection is needed for the evaluation. However, error will still be raised if the files any dataset or model is missing from the cache.

**Q2**: My server cannot connect to the Internet, how can I use OpenCompass?

Use the cache files from other machines, as suggested in the answer to **Q1**.

**Q3**: In evaluation phase, I'm running into an error saying that `FileNotFoundError: Couldn't find a module script at opencompass/accuracy.py. Module 'accuracy' doesn't exist on the Hugging Face Hub either.`

A: HuggingFace tries to load the metric (e.g. `accuracy`) as an module online, and it could fail if the network is unreachable. Please refer to **Q1** for guidelines to fix your network issue.
