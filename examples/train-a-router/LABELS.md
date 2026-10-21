# Regenerate paired labels

This optional adapter starts a **new paid public reproduction**. Saved report replay remains free of model calls. It uses the fixed public MMLU-Pro/BBH/KOR questions, original roles/families/folds, Sol/Luna request settings, prices and grading semantics. New API responses may differ from the published recordings; the output does not replace or claim to recreate their exact bytes.

These commands run from this example folder after downloading `data/`. Use the locked `live` dependency group on Linux or macOS (CPU only). `OPENAI_API_KEY` must already be present in the environment; no key file path or Modal account is required. `--prepared` points to the original public prepared source directory containing `source-plan.json`, `rows.json`, `question-only/inputs.json`, and, for grading only, `gold.json`.

```bash
# Collect the complete 9,111-question training and 3,056-question calibration splits under one all-attempt cap.
uv run --frozen --group live python collect_labels.py collect --prepared data/source --output new-labels --split train calibration --budget-usd 40
```

After the previous process has stopped, repeat the identical command with `--resume`. The filesystem lock rejects two active writers. Completed terminal responses are kept regardless of format or correctness. At most three attempts are allowed for transient transport failures; unresolved execution retains its full cost reservation. Quota, billing, authorization and model-identity errors stop new dispatch. The cap includes physical known costs, unknown reservations and in-flight reservations, and cannot silently change during resume. Incomplete populations remain explicit and cannot be graded as complete subsets.

```bash
# Grade a complete split locally from saved readable responses and exact physical-attempt usage; this makes no API calls.
uv run --frozen --group live python collect_labels.py grade --prepared data/source --collection new-labels --split train --output new-grades-train.json
uv run --frozen --group live python collect_labels.py grade --prepared data/source --collection new-labels --split calibration --output new-grades-calibration.json
```

The advanced workflow below trains on those new grades. The short `router.py fit-*` and `evaluate-refit` walkthrough uses the downloaded original recordings; it is a separate path. Do not replace those recordings or use `reuse-controls` with new acquisition labels. All three learned methods and their shared cost estimates must be refitted under the new acquisition plan.

```bash
# Prepare a new study with the SHA of the new collection plan, then fit the embedding heads from explicit training grades.
NEW_ACQUISITION_SHA=$(python -c 'import hashlib; print(hashlib.sha256(open("new-labels/plan.json", "rb").read()).hexdigest())')
uv run --frozen --group outcomes python paired_training.py prepare --source data/source --pair data/pair.json --run new-run --acquisition-protocol-sha256 "$NEW_ACQUISITION_SHA"
uv run --frozen --group outcomes python paired_training.py train-embedding --run new-run --grades new-grades-train.json --features data/features
```

```bash
# On a CUDA GPU, refit Laya with the same new training grades and the downloaded original base checkpoint; this is additional training.
uv run --frozen --group training --group outcomes python paired_native.py --run new-run --grades new-grades-train.json --initial-checkpoint data/reference/native/base
```

```bash
# Score calibration/final questions without their labels, and fit the training-only text-topic control.
uv run --frozen --group outcomes python paired_training.py score-embedding --run new-run --features data/features --output new-run/embedding-scores.json
uv run --frozen --group training --group outcomes python router.py score --data data --run new-run --method native --device cuda
uv run --frozen --group outcomes python label_workflow.py fit-topic --run new-run --grades new-grades-train.json
```

```bash
# Freeze all 26 policies using only the explicit new calibration grades and the published singleton fee projection.
uv run --frozen --group outcomes python label_workflow.py calibrate --run new-run --calibration-grades new-grades-calibration.json --fee-projection data/fees/projection.json
```

The final split is a separate operation. The public freeze at `new-run/frozen/frozen-final-routes.json` is an explicitly retrospective envelope. **The final collector takes the raw bundle at `new-run/frozen/internal/frozen-final-routes.json` and its exact SHA**, while the outer envelope remains beside it. Previously published outcomes are not described as unseen. The adapter preserves the raw bundle's parent acquisition/grader lineage; new final responses receive their own acquisition hash.

```bash
# Collect only the fixed 3,043 final questions after the raw 26-policy freeze validates; this is another paid phase.
NEW_FROZEN_SHA=$(python -c 'import hashlib; print(hashlib.sha256(open("new-run/frozen/internal/frozen-final-routes.json", "rb").read()).hexdigest())')
uv run --frozen --group live --group outcomes python collect_labels.py collect --prepared data/source --output new-final-labels --split final --budget-usd 20 --frozen-routes new-run/frozen/internal/frozen-final-routes.json --frozen-sha256 "$NEW_FROZEN_SHA"
```

```bash
# Verify and grade the complete final split with its recorded parent/final lineage; no model call occurs here.
uv run --frozen --group live --group outcomes python collect_labels.py grade --prepared data/source --collection new-final-labels --split final --output new-grades-final.json
uv run --frozen --group outcomes python label_workflow.py evaluate --run new-run --final-grades new-grades-final.json --output new-report.json
```

Every run writes the plan, question inputs, physical intents/responses, per-arm first-terminal selections, split collection receipts, and a separate reproduction receipt. Grading checks all declared and actual attempt files, exact model identity, readable usage and exclusive cache accounting. Output tokens include reasoning once. The selected response's deployment cost is separate from the full retry/unknown acquisition ledger; missing usage is never treated as free. Invalid or truncated terminal answers remain in the denominator and charged.

`_collection/collect.py`, `openai_population.py`, and `grade_openai_population.py` preserve the exact original source bytes. `collect.py` is byte-identical to the public `answer_models.py`; its original filename is retained because the unchanged scheduler and attempt verifier import `collect`. No scientific module is monkeypatched or rewritten. The new wrapper owns its distinct plan/grader hashes and reuses the original `body_for`, rolling scheduler, request transport, cost calculation, parser and `checked_attempts` verifier. `_collection/source-manifest.json` records those byte bindings. The adapter's tiny mocked lifecycle tests make no network calls; a complete live run of this new public wrapper has not been claimed.

The final validator is the public example's `paired_calibration.checked_frozen_bundle`; it needs the CPU `outcomes` group. API collection does not read gold. Although the prepared directory may contain gold for later grading, the collector reads only source-plan, reference-free rows and question-only inputs. `label_workflow.py` calls the unchanged calibration and reporting functions, verifies the new acquisition/parent/freeze hashes, and marks the report as a public reproduction. It does not require new responses to match the original reference acquisition. Tiny synthetic metadata tests exercise these lineage checks and the complete 26-policy calibration/report path; no full paid regeneration or new GPU fit of this adapter is claimed.
