# Build a request router that learns from feedback

Start with the **[ODSC West workshop one-pager](./WORKSHOP.md)**. All code and
configuration live in this example; datasets and recorded model outputs download
from a pinned Hugging Face revision. The main lab runs on a CPU laptop.

The task is concrete: route a mathematics problem to routine or advanced handling
using independent, human-performance-derived difficulty labels. With fixed
Qwen3-Embedding-4B vectors, a small learned classifier improves held-out accuracy
from **71.7% to 89.6%**, reducing errors **145 → 53**. Adding feedback from 64 to
512 examples improves the *same classifier* from **79.5% to 89.6%**.

Students train the classifier themselves, freeze it, evaluate it, and route a new
request using SIE Cloud or a local SIE server. Training reuses recorded vectors;
serving needs one embedding call and a local dot product. The exercise measures
routing accuracy, not downstream answer quality or dollar savings.

- [One-page walkthrough](./WORKSHOP.md): install, learn, evaluate, use.
- [Course guide](./cost/README.md): structure, static/improving modes, local serving and extensions.
- [Measured results and dataset](./cost/DIFFICULTY_POLICY.md): methods, confidence intervals and public evidence.
- [Decision-model post-training](./cost/POSTTRAIN.md): native Laya full fine-tuning and encoder LoRA.
- [Original handler cascade](./CASCADE.md), [direct classification](./direct), and
  [earlier cost-routing study](./cost/LEGACY_COST.md) are separate examples retained for reference.

The unchanged embedding policy reaches **91.6%** on a fresh 512-problem confirmation
set. A second learning path fine-tunes a **421M-parameter Laya** decision model:
53.3% → 87.1% with full tuning, or 88.1% with LoRA plus head training. Both
checkpoints serve through public SIE on Apple Silicon and CUDA.
[Public checkpoints, predictions, verification receipts and results figure](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/85c9d93a73d5559a6c69a22145f5f86f6b450fbc/evaluations/2026-10-18/request-routing-course-posttrain-v2).
