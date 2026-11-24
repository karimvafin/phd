# Reproducing Controlled Decoding (Mudgal et al., ICML 2024)

Paper: [arXiv 2310.17022](https://arxiv.org/abs/2310.17022). We reproduce Experiment 2: Anthropic HH, **blockwise CD-FUDGE vs best-of-K**.

**Question.** Can selection over prefixes reach the reward of best-of-K with a large K? The selection works as follows: generate K blocks of M tokens, keep the block with the best prefix scorer score, and continue.

## Setup

| | Paper | Here |
|---|---|---|
| Base model π_ref | PaLM 2-XXS | Qwen2.5-0.5B-Instruct, sampling at T=1 without filtering, up to 256 tokens |
| Reward model | PaLM 2-XXS | Skywork-Reward-V2-Qwen3-0.6B |
| Prefix scorer | CD-FUDGE | Qwen2.5-0.5B + linear head, CD-FUDGE: MSE of every prefix to the reward of the full response |
| Scorer training data | HH | 10,000 HH prompts × 2 π_ref responses |
| Evaluation | — | 500 held-out HH test prompts |
| Hardware | — | Kaggle, 2×T4 |

## Metrics

- **Reward**: mean RM reward of the response on test prompts.
- **Cost**: measured three ways.
  - Number of generated tokens, including discarded candidates.
  - Wall-clock time per prompt: generation, scorer and RM, batch of 50 prompts.
  - Upper bound on KL(π‖π_ref). For best-of-K it is log K − (K−1)/K; for CD it is the same multiplied by the number of blocks.
- **Equivalent K (K_eq)**: the K at which best-of-K reaches the same reward as CD.
- **Significance**: difference CD − best-of-K, 95% CI from a paired bootstrap over prompts (both methods on the same prompts).

## Main results (T=1)

### Prefix scorer quality

Pairwise accuracy: given the first t tokens of two responses to the same prompt, predict which one gets the higher RM reward. Computed only on pairs where both responses are longer than t. Test: 6,800 pairs on 500 held-out prompts, error ±0.012.

| t, tokens | 8 | 16 | 32 | 64 | 128 | full response |
|---|---|---|---|---|---|---|
| Accuracy (test) | 0.61 | 0.65 | 0.69 | 0.75 | 0.79 | 0.84 |

*Accuracy*: share of test pairs (both responses longer than t) where the scorer's ranking of the two t-token prefixes matches the RM ranking of the full responses; chance is 0.5.

Already at 16–32 tokens the scorer is clearly better than chance, so it can guide block selection. No overfitting: 0.83 on the full response on validation.

### Best-of-K

| K | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| Reward | −4.43 | −3.09 | −2.01 | −1.12 | −0.37 | 0.29 |
| KL (bound) | 0 | 0.19 | 0.64 | 1.20 | 1.84 | 2.50 |
| Time, s/prompt | — | 0.59 | 1.17 | 2.31 | 4.10 | 7.47 |

*Reward*: mean Skywork 0.6B reward of the selected response over 500 test prompts. *KL (bound)*: log K − (K−1)/K. *Time*: wall-clock seconds per prompt for generation + RM, batch of 50 prompts.

### Blockwise CD

| M | K | Reward | Tokens | Time, s/prompt | KL (bound) | K_eq [95% CI] | Δ vs best-of-K at equal tokens [95% CI] | Token efficiency vs best-of-K |
|---|---|---|---|---|---|---|---|---|
| 16 | 2 | −2.33 | 263 | 0.93 | 1.66 | 3.3 [2.9; 3.7] | **+1.03** [0.84; 1.23] | 1.8× |
| 16 | 4 | −0.98 | 505 | 1.62 | 5.37 | 9.1 [7.6; 11.0] | **+1.27** [1.06; 1.48] | 2.6× |
| 16 | 8 | **−0.16** | 967 | 3.41 | 9.98 | **19.9** [16.2; 24.6] | **+1.19** [0.99; 1.40] | 3.0× |
| 32 | 2 | −2.61 | 265 | 0.65 | 0.88 | 2.7 [2.4; 3.1] | **+0.73** [0.50; 0.95] | 1.5× |
| 32 | 4 | −1.33 | 500 | 1.19 | 2.79 | 6.8 [5.9; 7.8] | **+0.93** [0.75; 1.11] | 2.0× |
| 32 | 8 | −0.46 | 974 | 1.96 | 5.34 | 14.8 [12.5; 17.8] | **+0.89** [0.71; 1.08] | 2.2× |
| 64 | 2 | −2.90 | 268 | 0.52 | 0.48 | 2.3 [2.0; 2.6] | **+0.41** [0.21; 0.61] | 1.25× |
| 64 | 4 | −1.84 | 511 | 0.90 | 1.55 | 4.6 [4.0; 5.3] | **+0.38** [0.21; 0.56] | 1.3× |
| 64 | 8 | −1.06 | 977 | 1.59 | 2.84 | 8.5 [7.4; 9.8] | **+0.29** [0.12; 0.45] | 1.3× |

*M*: block length in tokens; *K*: candidates per block. *Tokens*: mean generated tokens per prompt, including discarded candidates. *Time*: wall-clock seconds per prompt for generation + scorer + RM, batch of 50. *KL (bound)*: best-of-K bound log K − (K−1)/K times the number of blocks. *K_eq*: the K at which best-of-K reaches CD's mean reward (log-interpolated), CI from a paired bootstrap over prompts. *Δ at equal tokens*: CD reward minus best-of-K reward at the same token budget: best-of-K is computed for every K from 1 to 32 and interpolated to the fractional K with the same number of tokens as CD (`06_equal_tokens`). *Token efficiency*: K_eq divided by that equal-token K. Significant differences are in bold.

**At equal KL bound, best-of-K is better in all 9 configurations:** Δ from −0.45 to −1.96, all CIs below zero.

## Robustness checks

### Independent judges

The responses of all methods are re-scored by two other RMs. Best-of-K still selects the response with Skywork 0.6B, and the judge scores the selected response.

| | Skywork 0.6B (target) | Skywork-V2 4B | GRM-Llama3.2-3B |
|---|---|---|---|
| Rank correlation with target | 1 | 0.78 | 0.48 |
| Win rate vs π_ref: best-of-4 / best-of-32 | 0.82 / 0.97 | 0.75 / 0.88 | 0.65 / 0.74 |
| K_eq: CD M=16, K=4 | 9.1 | 22 | 10.6 [6.4; 21] |
| K_eq: CD M=16, K=8 | 19.9 | ≥32 | 19.1 [9.8; 32] |
| Significant gain at equal tokens (vs the nearest best-of-K with at least as many tokens) | 7 of 9 | 9 of 9 | 6 of 9 |

*Rank correlation*: Spearman correlation between the judge's and the target RM's rewards on all π_ref samples. *Win rate vs π_ref*: share of comparisons in which the selected response gets a higher judge score than each of 4 held-out π_ref samples for the same prompt (tie = 0.5, π_ref itself = 0.5), averaged over prompts. *K_eq*: as above, but with rewards from this judge. *Significant gain*: number of CD configurations whose 95% CI of Δ vs best-of-K lies above zero.

Under another judge the absolute gain of both methods is smaller: part of it is overfitting to Skywork 0.6B. **CD's relative advantage holds**: under GRM, K_eq is almost the same as under the target. Caveat: GRM is also trained on Skywork data, so it cannot be called a fully independent judge.

### Real cost

We compare CD with best-of-K of the same quality (K = K_eq, best-of-K time interpolated).

| CD | CD time, s/prompt | Same quality with best-of-K | Best-of-K time, s/prompt | CD speedup |
|---|---|---|---|---|
| M=16, K=4 | 1.62 | best-of-9 | ~2.6 | 1.6× |
| M=32, K=4 | 1.19 | best-of-7 | ~2.0 | 1.7× |
| M=16, K=8 | 3.41 | best-of-20 | ~4.9 | 1.4× |
| M=32, K=8 | 1.96 | best-of-15 | ~3.8 | 1.9× |

*CD time / best-of-K time*: wall-clock seconds per prompt (generation + scorer + RM), batch of 50 prompts; best-of-K time interpolated to K = K_eq. *Same quality with best-of-K*: best-of-K_eq, rounded. *CD speedup*: best-of-K time divided by CD time.

Including scorer calls, CD at equal quality is 1.4–1.9× cheaper, not 2–2.5× as the token count suggests. The scorer takes 16–30% of CD time. At M=16 there is extra overhead from many small generation calls, and the scorer, lacking a KV cache, re-reads the whole prefix. We measure batched throughput, not single-request latency.

### Response length and truncated responses

| | Length, tokens | Finished | Reward on finished |
|---|---|---|---|
| π_ref | 146 | 72% | −3.94 |
| best-of-4 / best-of-8 | 137 | 80% / 82% | −1.79 / −0.96 |
| CD M=16, K=4 / K=8 | 128 / 126 | 85% / 88% | −1.02 / −0.09 |

*Length*: mean response length in tokens. *Finished*: share of responses that end with EOS before the 256-token limit. *Reward on finished*: mean reward over finished responses only.

The RM penalizes responses truncated at the 256-token limit, and CD produces them less often. **But this does not fully explain CD's gain**: on finished responses only, Δ vs best-of-K stays +0.77 and +0.87 for K=4 and K=8 (vs the nearest best-of-K with at least as many tokens, which understates CD).

### Temperature T=0.7, top-p=0.9

Same scorer, trained on responses at T=1.

| | T=1 | T=0.7 |
|---|---|---|
| π_ref | −4.43 | −2.31 |
| best-of-4 / best-of-8 | −2.01 / −1.12 | −0.48 / 0.18 |
| CD M=16, K=4: reward / K_eq / Δ at equal tokens | −0.98 / 9.1 / **+1.27** | −0.77 / 3.1 / −0.16 |
| CD M=16, K=8: reward / K_eq / Δ at equal tokens | −0.16 / 19.9 / **+1.19** | −0.20 / 5.3 / **−0.27** |

*Reward, K_eq, Δ at equal tokens*: as in the blockwise CD table, with best-of-K computed at the same temperature. Significant Δ in bold.

**With the scorer trained at T=1, CD at T=0.7 is worse than best-of-K in all three configurations**: at equal tokens Δ from −0.16 to −0.28, efficiency 0.75–0.86× of best-of-K. Best-of-K gains 1–1.5 reward from the lower temperature, while CD gains almost nothing.

### Scorer trained at T=0.7

The scorer is trained with the same recipe on 10,000 prompts × 2 responses at T=0.7, top-p=0.9 (`05_temp07`).

Strict pair_acc on test:

| Scorer → responses | t=8 | t=32 | t=64 | full response |
|---|---|---|---|---|
| T=1 → T=1 | 0.61 | 0.69 | 0.75 | 0.84 |
| T=1 → T=0.7 | 0.58 | 0.61 | 0.63 | 0.77 |
| T=0.7 → T=0.7 | 0.60 | 0.64 | 0.67 | 0.80 |

*Scorer → responses*: temperature of the scorer's training data → temperature of the test responses. *Strict pair_acc*: pairwise accuracy as above, only on pairs where both responses are longer than t.

CD at T=0.7:

| CD | Reward: scorer T=1 → T=0.7 | Δ new − old [95% CI] | Δ vs best-of-K at equal tokens: scorer T=1 → T=0.7 | Token efficiency vs best-of-K: scorer T=1 → T=0.7 |
|---|---|---|---|---|
| M=16, K=4 | −0.77 → −0.27 | +0.50 [0.34; 0.68] | **−0.16** → +0.13 [−0.03; 0.28] | 0.86 → 1.13 |
| M=32, K=4 | −0.88 → −0.46 | +0.42 [0.27; 0.57] | **−0.28** → −0.05 [−0.20; 0.10] | 0.78 → 0.95 |
| M=16, K=8 | −0.20 → +0.45 | +0.65 [0.46; 0.83] | **−0.27** → **+0.18** [0.02; 0.35] | 0.75 → 1.24 |

*Reward: scorer T=1 → T=0.7*: CD reward at T=0.7 with the old (T=1) scorer → with the new (T=0.7) scorer. *Δ new − old*: their difference, paired bootstrap CI. *Δ vs best-of-K* and *token efficiency*: as in the blockwise CD table, CI for the new scorer. Significant values in bold.

- **There was a distribution shift.** The new scorer is more accurate on T=0.7 responses, and CD with it is significantly better.
- **But at T=0.7 there is little to choose from.** The reward spread between two responses to the same prompt is 25% smaller (1.46 vs 1.96). Even a matched scorer separates such responses worse than the T=1 scorer separates its own (0.64 vs 0.69 at t=32).
- **Bottom line: at equal tokens, CD with the new scorer is on par with best-of-K** (0.95–1.24×; only M=16, K=8 is significant: +0.18), vs 1.5–3× at T=1. Accounting for scorer time, there is most likely no gain.
- The new scorer picks longer responses: 149 tokens vs 122, 73% finished vs 82%.

## Zero-shot transfer from HH to GSM8K

We evaluated the frozen HH-trained CD-FUDGE scorer on all 1,319 questions in the official GSM8K test split. All methods used Qwen2.5-0.5B-Instruct with the same zero-shot prompt, T=1 sampling without filtering, and a 512-token output cap. Blockwise CD used M=32 tokens and K=4 candidates per block. The random-block baseline generated the same number of candidates per block and kept the first sampled candidate without scoring. Unguided sampling generated one complete response per question.

| Method | Strict exact match | Exact match with boxed fallback | Generated tokens per question | Seconds per question |
|---|---:|---:|---:|---:|
| Unguided sampling | 3.71% (49/1,319) | 25.25% (333/1,319) | 302 | 1.13 |
| Random block selection | 3.34% (44/1,319) | 24.72% (326/1,319) | 1,195 | 2.00 |
| Blockwise CD with HH scorer | 2.96% (39/1,319) | 28.89% (381/1,319) | 1,058 | 2.28 |

Generated tokens include discarded candidates. Times are batched wall-clock seconds per question. Accuracy uses all questions as the denominator and counts unparseable outputs as incorrect.

| CD minus baseline | Strict difference, pp [95% CI] | Boxed fallback difference, pp [95% CI] |
|---|---:|---:|
| Unguided sampling | −0.76 [−2.12, +0.61] | +3.64 [+0.68, +6.60] |
| Random block selection | −0.38 [−1.67, +0.91] | +4.17 [+1.52, +6.82] |

Intervals come from 10,000 paired bootstrap resamples over questions. The original parser required a standalone `#### <number>` line and extracted an answer from 11.3% of unguided, 12.4% of random-block, and 8.3% of CD outputs. The strict result against unguided sampling leaves a loss greater than the 2-percentage-point margin possible.

The post hoc analysis kept every answer extracted by the strict parser and otherwise used the last LaTeX `\boxed{...}` containing a plain signed or unsigned number, allowing decimals and thousands separators. This raised extraction rates to 58.8%, 61.3%, and 62.4%, respectively. Under that rule, CD improved accuracy over both baselines and used about 3.5× the generated tokens and 2.0× the wall-clock time of unguided sampling. The ranking of methods changes with the extraction rule, and 38–41% of outputs still have no extracted answer, so the evidence for improved math accuracy is preliminary.

The remaining compute-efficiency comparison is CD versus best-of-2, 4, and 8, with complete responses ranked by the same frozen HH scorer. Use a common answer extractor validated on a blinded sample of responses, and compare accuracy against actual generated tokens and wall-clock time.

[Notebook](07_gsm8k_transfer.ipynb) · [Full predictions](07_gsm8k_transfer_results/gsm8k_hh_transfer_full_predictions.csv) · [Original summary](07_gsm8k_transfer_results/gsm8k_hh_transfer_full_summary.csv) · [Original paired results](07_gsm8k_transfer_results/gsm8k_hh_transfer_full_paired.csv)

## Conclusions

1. **The main effect reproduces at T=1.** CD with K candidates per block reaches the reward of best-of-K with a 2–2.5× larger K. Example: M=16, K=8 ≈ best-of-20.
2. **At an equal token budget, CD beats best-of-K in all 9 configurations**: 1.25–3× more efficient, Δ from +0.29 to +1.27.
3. **The effect is robust.** It holds under an independent judge, on finished responses only, and in wall-clock time, although the time gain is smaller: 1.4–1.9×. M=32 gives the best cost/quality trade-off.
4. **M acts as a trade-off knob.** Smaller blocks give higher reward and a stronger deviation from π_ref. At M=64 (only 4 blocks per response) the gain is small: 1.3×.
5. **At T=0.7 the effect nearly vanishes.** The scorer must be trained at the same temperature CD runs at: this adds +0.4–0.65 reward. Even then CD is only on par with best-of-K in tokens. The main reason is that at low temperature responses to the same prompt differ less in quality, so prefix selection gains little. Most of CD's gain at T=1 appears to come from high sample diversity.
6. **Best-of-K looks better in KL, but the comparison is unfair.** For CD this is an upper bound, and most likely a loose one: sibling candidates share a common prefix.

## Limitations

- **No LLM judge.** Both extra RMs are models of the same kind as the target. The paper used a separate, stronger model for evaluation.
- **Time** is measured on a batch of 50 prompts on a T4. Single-request latency of CD is higher than best-of-K, because blocks are generated sequentially.
- **Different models.** Instead of PaLM 2 we use open 0.5–0.6B models, so absolute reward values are not comparable with the paper. Only the ordering of methods is comparable.

---

Code:
- `01_rollouts.ipynb` — π_ref responses and their rewards;
- `02_train_scorer.ipynb` — prefix scorer training;
- `03_eval.ipynb` — best-of-K vs CD comparison;
- `04_feedback.ipynb` — robustness checks;
- `05_temp07.ipynb` — scorer trained at T=0.7;
- `06_equal_tokens.ipynb` — comparison with best-of-K at exactly equal token count (CPU).

Kaggle datasets:
- [`cd-hh-rollouts`](https://www.kaggle.com/datasets/karimvafin/cd-hh-rollouts) (`01`): T=1 π_ref responses with Skywork 0.6B rewards for 10,000 HH train prompts × 2 (`train.parquet`), the 500 held-out test prompts (`test_prompts.parquet`), and the generation config (`config.json`).
- [`cd-hh-scorer`](https://www.kaggle.com/datasets/karimvafin/cd-hh-scorer) (`02`): the T=1 prefix scorer (Qwen2.5-0.5B + value head, fp16) with tokenizer, `scorer_config.json` (reward normalization, validation metrics), and the training log `metrics.jsonl`.
- [`cd-hh-eval`](https://www.kaggle.com/datasets/karimvafin/cd-hh-eval) (`03`): per-prompt best-of-K (`bon_*.parquet`) and CD (`cd_M*_K*_*.parquet`) responses with rewards and token counts on the 500 test prompts, plus `run_meta.json`, `summary.csv`, `paired.csv` and `curves.png`.
- [`cd-hh-feedback`](https://www.kaggle.com/datasets/karimvafin/cd-hh-feedback) (`04`): robustness checks: independent-judge scores (`judge_*.parquet`, `judges.csv`), wall-clock timing (`timing.csv`), length/finished stats (`lengths.csv`), and best-of-K and CD runs at T=0.7 with the T=1 scorer (`tbon_*`, `temperature.csv`).
- [`cd-hh-scorer-t07`](https://www.kaggle.com/datasets/karimvafin/cd-hh-scorer-t07) (`05`): the prefix scorer retrained on T=0.7, top-p=0.9 responses, in the same format as `cd-hh-scorer`.
- [`cd-hh-t07`](https://www.kaggle.com/datasets/karimvafin/cd-hh-t07) (`05`): T=0.7 train responses with rewards (`train_t07.parquet`), CD runs with the T=0.7 scorer, pair_acc tables, and `summary_t07.csv`.
