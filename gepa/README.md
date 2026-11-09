# Reproducing GEPA on AIME

Goal: check whether the accuracy gain from GEPA prompt optimization reported in the official
DSPy tutorial ([dspy.ai/tutorials/gepa_aime](https://dspy.ai/tutorials/gepa_aime)) reproduces.

Notebook: gepa_repro.ipynb

The original tutorial notebook is also included: gepa_dspy_tutorial.ipynb (taken from https://github.com/stanfordnlp/dspy/blob/main/docs/docs/tutorials/gepa_aime/index.ipynb)

The optimized prompts are in ./prompts

## Setup

**Code.** The DSPy GEPA AIME tutorial setup with no substantive changes (`dspy==3.4.1`, `gepa==0.1.4`):
a `dspy.ChainOfThought` program with the instruction "Solve the problem and provide the answer in the correct format";
GEPA optimizes the text of this instruction.

**LLMs** (via OpenRouter):
- executor (solves the problems): `gpt-4.1-mini`, temperature = 1, max_tokens = 32000;
- reflector (proposes new prompts): `gpt-5`, temperature = 1.

**Data.**
- train / val: 45 / 45 problems from past AIME years (`AI-MO/aimo-validation-aime`, random 50/50 split);
- test: 30 AIME 2025 problems (`MathArena/aime_2025`).

**Metric.** Exact match of the integer answer. For the reflector, the score comes with
textual feedback: the correct answer and the reference solution of the train problem.

**GEPA hyperparameters.** Budget `auto="light"`, 576 metric calls in practice (executor
rollouts on train/val); `reflection_minibatch_size = 3`; a single seed (seed = 0).

**Test evaluation.** Each problem is solved 5 times independently (30 × 5 = 150 rollouts per program).
95% confidence intervals are computed by bootstrap over problems; two programs are compared
with a paired bootstrap.

**Key difference from the tutorial: the DSPy cache is disabled.** In the tutorial, the test set is built as
`test_split * 5` (30 × 5 = 150 examples), and the LM is created with the default cache. The DSPy cache key
depends on the prompt and request parameters but not on the repeat index, so once the cache is warm,
all 5 repeats of a problem receive the same cached entry. The tutorial's results of 70/150 (46.7%) and 85/150 (56.7%)
therefore correspond to 14 and 17 solved problems out of 30, with one sample per problem. The variance of such an
estimate is understated. Here the 5 repeats are truly independent samples.

The caching behavior was verified with the script `check_dspy_cache.py`: one problem, 5 repeats, tutorial
settings. With sequential evaluation there is a single real API request, and the other 4 responses are
copies of it from the cache. The cache does not deduplicate identical requests sent concurrently from different
threads, so on the very first ("cold") run with `num_threads=32` some of the repeats
may have been independent. This does not affect the published numbers: they were obtained on a re-run
with a warm cache.

## Results

Over 576 rollouts GEPA generated 11 candidate prompts. The best one on val raised accuracy
from 40.0% (seed prompt) to 53.3%.

In addition to our own GEPA run, we evaluated on test **the prompt published in the tutorial** (output of cell 18
of the tutorial notebook), under the same conditions: 30 × 5 samples without cache. All three prompts are in `prompts/`:
`seed_prompt.txt` (seed), `gepa_seed0_prompt.txt` (best on val in the seed 0 run), `tutorial_prompt.txt`.

None of the optimized prompts improves on test:

| Program | Test accuracy, 95% CI | Solved at least once | Solved in all 5 samples |
|---|---|---|---|
| Seed prompt | 48.0% [32.7; 62.7] | 19 / 30 | 8 / 30 |
| GEPA (seed 0) | 46.7% [32.0; 62.0] | 19 / 30 | 8 / 30 |
| Tutorial prompt | 45.3% [30.7; 60.0] | 20 / 30 | 9 / 30 |

Paired differences (bootstrap over problems):
- GEPA (seed 0) − seed prompt: **−1.3 pp, 95% CI [−7.3; +4.7]**;
- tutorial prompt − seed prompt: **−2.7 pp, 95% CI [−8.7; +3.3]**.

**Noise with one sample per problem.** If we take one sample per problem (which is effectively what
the tutorial does because of the cache), the number of solved problems for the same prompt varies
with the sample index: 13–16 out of 30 for the seed prompt and 12–16 for the tutorial prompt.
The 14 → 17 difference reported in the tutorial is within this range.

**Conclusions.**
1. The val gain (+13 pp) did not transfer to test: on AIME 2025 the optimized prompt
   performs the same as the seed prompt. This looks like overfitting to the small val set (45 problems).
2. The tutorial's gain (46.7% → 56.7%) does not reproduce: with independent samples
   the published prompt gives 45.3% vs 48.0% for the seed prompt. A +10 pp effect
   is ruled out by the confidence interval (upper bound of the difference is +3.3 pp).
3. Both optimized prompts consist mostly of worked solutions to specific
   train problems, including their answers. They are more of a "cheat sheet" for train than a general
   solving strategy. This seems to be how feedback with reference solutions works: the reflector
   sees the solution of a minibatch problem and writes it into the prompt.
   - In the best seed 0 prompt: roots of unity, isosceles trapezoids, etc.,
     train answers 51, 242, 315, 321.
   - In the tutorial prompt: 6 of the 45 train problems (#1, #5, #6, #21, #27, #35 in the tutorial's
     split); answers 227, 585, 601 are given as "answer format examples",
     and one more answer (841) follows from 41/800.
   - Neither prompt overlaps with the test problems or answers, so there is no test leakage.
4. The test set is small: the CI for a single program's accuracy is about 30 pp wide, and for the paired
   difference about 12 pp. So a gain of a few pp cannot be reliably detected
   on 30 problems, but an effect of +10 pp or more is unlikely.
