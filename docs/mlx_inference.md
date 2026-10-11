# MLX inference profiling

PR4 extends endpoint profiling for oMLX, MLX-LM and MLX-Audio. Its acceptance
criteria are in [the PR4 plan](mlx-inference-pr4-plan.md). Implementation and
protocol fixture validation are in progress; live server qualification is
required before declaring the integration complete.

Text profiling retains the existing chat defaults. Select `--engine omlx`
or `--engine mlx-lm` and `--task completion` for raw text completions:

```bash
stormlog infer profile --engine mlx-lm --task completion \
  --base-url http://127.0.0.1:8080/v1 --model MODEL_ID \
  --tokenizer estimate --input-tokens 32 --output-tokens 16 \
  --requests 10 --concurrency 1,2 --output /tmp/mlx-completion.jsonl
```

Model IDs remain opaque. `--endpoint` is a full URL; `--base-url` gains only
the task route. JSON and SSE share the existing no-redirect, no-proxy,
no-retry transport. `--timeout` bounds the total request, including trickling
responses; `--max-response-bytes` bounds cumulative response data. Sent
request IDs and server-generated completion IDs are recorded independently.
Unknown server versions and execution attribution remain unavailable.

Text and audio scheduling share arrival indices, in-flight admission, drops,
phase windows and bounded drain. Calls still running after an asyncio drain
remain tracked; the next phase waits for them before starting. The text
extraction passed 148 existing scheduling, profiler, trace and vLLM tests.
