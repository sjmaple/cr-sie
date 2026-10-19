# Where the earlier assets went

There are two independent examples:

- [`request-routing`](../../request-routing/README.md): the original handler
  cascade. Its `direct/` directory makes one native classification call using
  supplied handler labels. Its `cost/` directory is a separately installable,
  historical cheap/strong LLM cost study.
- [`train-a-router`](../README.md): learn and serve a difficulty decision, with
  one environment and a progression from LoRA adaptation of Laya to a learned
  classifier on frozen Qwen embeddings.

The earlier experimental workspace combined several different targets: handler
intent, downstream answer success, global mathematics difficulty and relative
difficulty. Those are not interchangeable. Reranker, teacher, economics and GEPA
experiments are not prerequisites for this example.

The complete earlier source remains in the immutable
[research snapshot](https://github.com/superlinked/sie/tree/23abfac2aa7f600a40a87ca15558e1d3e93c6c97/examples/request-routing/cost).
Its [native-model evidence](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/85c9d93a73d5559a6c69a22145f5f86f6b450fbc/evaluations/2026-10-18/request-routing-course-posttrain-v2)
and other pinned historical datasets remain available. Their original numbers
are historical observations, not results for this corrected protocol.

The audit found equivalent AMC10/AMC12 problems spanning old partitions and a
global label strongly confounded with contest family. LoRA's old confirmation
advantage over full tuning was 5 of 512 cases: +0.98 percentage points, paired
95% interval −1.37 to +3.32. That did not establish a winner. The new example
changes the target, isolates duplicate families and years, separates calibration,
and reports three seeds. See [methodology](methodology.md) for the exact rules.
