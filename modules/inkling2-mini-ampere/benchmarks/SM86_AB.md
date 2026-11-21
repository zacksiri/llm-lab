# SM86 serving A/B: what to turn on

Full-model A/B on two NVLink RTX A4500s (200 W cap), all with the 250,000-token limit, two slots, INT8 KV, vision, and DSpark with decode CUDA graphs. The GPUs were idle and each variant was a fresh server start. Every variant passed text/image/tool/DSpark smoke tests before timing. `ab_bench.py` runs: two timed 384-token completions per prompt after a warm-up (code and prose, median), two simultaneous completions (aggregate tok/s), and one cold ~64K chat prompt followed by decode. Each number is one server run; treat differences under ~5% as noise.

| Variant | Code | Prose | Two sessions | 64K decode | Starts at 250K? | KV capacity (est.) |
|---|---:|---:|---:|---:|---|---:|
| Baseline | 209 | 111 | 180 | 114 | yes | 285,517 |
| Async scheduling, 2,048 chunks | | | | | **no** (needs 6.09 GiB KV, 4.93 available) | |
| Async + 1,024-token chunks | 230 | 121 | 194 | 127 | yes | 302,391 |
| 48-split verify + 1-stage tile | 209 | 111 | 183 | 129 | yes | 285,517 |
| DSpark 3 draft tokens | 199 | 123 | 184 | 133 | yes | 287,545 |
| DSpark 2 draft tokens | 166 | 120 | 178 | 125 | yes | 287,328 |
| Custom all-reduce | | | | | **no**: no SM86 kernel in this vLLM image | |
| 4,096-token chunks | | | | | **no**: a Triton tile needs 144 KiB shared memory; SM86 has 99 KiB | |
| **Combined, 4 drafts** | 230 | 121 | 194 | **136** | yes | 302,391 |
| **Combined, 3 drafts** (recommended) | 221 | **135** | **204** | 134 | yes | 304,198 |

"Combined" = async scheduling + 1,024-token chunks + 48-split verify. Values are tok/s. The baseline's 64K first-token time was inflated in that run, so first-token time is not compared above; the variants measured ~13.9–15 s.

Cold retrieval just below the limit (249,273 prompt tokens, exact phrase retrieved by both combined builds): **88.2 s** (4 drafts) and **88.4 s** (3 drafts) to first token, vs **103.0 s** for the earlier prefill-tuned build and **118.7 s** before prefill tuning. Those older numbers came from separate sessions on the same hardware.

The recommended profile is three drafts: it gives up ~4% on code and ~1.5% at 64K for +12% prose and +5% two-session throughput. Use four drafts for code-dominated workloads. Enable with the commented block in `.env.example`; the module's defaults are unchanged. Async scheduling plus smaller chunks fit 250K because they reserve less transient state, not because the KV pool grew. Two simultaneous ~250K requests still queue.

Raw results: [sm86-ab-results/](sm86-ab-results/).
