# LLM Lab

Reproducible local inference experiments. Each module has its own build, runtime configuration, tests, and upstream attribution. No model weights or API keys are stored here.

| Module | What it runs |
| --- | --- |
| [Inkling 2 Mini on Ampere](modules/inkling2-mini-ampere/) | Patched vLLM on two SM86 GPUs: mixed INT8/BF16 KV, images, and optional DSpark. |
