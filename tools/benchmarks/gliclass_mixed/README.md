# GLiClass mixed-load reproduction

This retains two explicit comparisons for [issue390](https://github.com/superlinked/sie/issues/390). The default `profile` preparation reproduces the historical [PR656](https://github.com/superlinked/sie/pull/656) contrast: both archives use `2281ba860f3c530d36e6f19d638b624bededef26`, with only the candidate catalog file replaced by `a5e8d7b39be4b8156932d902d75fa46dc3e37e53`. That derived archive does not represent the contributor's entire branch. The completed negative experiment remains pinned to its original harness commit `adfa7cc93ef7496dc7335373978ab752f2421dd1`.

The new `progress` preparation compares complete serving-source snapshots at baseline `f3b34e2dd88df74d4cc62c1446d28982c769f6ea` and candidate `45794f69402d1c334c1238cfc13082cc0596e08f` ([PR798](https://github.com/superlinked/sie/pull/798)). Both use the unchanged adaptive-enabled profile. Preparation and extraction reject every source difference except the batcher and worker documentation; an executable AST check rejects worker code changes. All model profiles, SDK/config code, and dependency files must be identical. The actual imported batcher and worker hashes are checked against each arm's archive before requests. No result from this new comparison is claimed before execution and complete artifact audit.

The fixed checkpoint is `knowledgator/gliclass-large-v1.0@ae8a18644e84cfda53325bd91c7941f2f6fd0ccb`, FP16, default bucketed CUDA graphs, one L4, eight CPU cores, 32 GiB RAM and four Torch threads. The public lock and complete consumed distribution manifests must match. Source hashes are checked in the real SIE server and `SIEClient` processes. The client forbids remote fallback and ignores proxy environment settings.

After two untimed repetitions of each fixture, each fresh server runs C1 for60s, C8 for four60s intervals and C32 for60s, with eight sequential warmup calls per phase. Mixed traffic then uses12 cheap one-item lanes and four24-item lanes (medium7, medium8, long7 and long8 groups) for480s. Each lane issues the next call only after its previous call ends. Calls issued before cutoff retain their full300s deadline; the server remains alive for another60s after the full client drain to observe late work. No automatic experiment retry is permitted.

Every heavy document fits the1024-token compiled window. Medium item costs exceed the4096 adaptive floor but fit the unchanged16384 static budget with the cheap backlog; long item costs exceed16384. The tokenizer-only validator checks both bands. A medium-only improvement does not establish resolution of long-item starvation.

All HTTP attempts, SDK results/errors, nearest-rank latency samples and phase boundaries are retained. Summaries distinguish submission-cohort outcomes from end-timestamp completions and timeouts, including the drain. Minimal symmetric server observers record admission, future completion/cancellation and actual batch/forward membership during correctness and mixed traffic. They are removed, restoring the original methods, during homogeneous timing. Forward timestamps measure host dispatch intervals, not synchronized GPU kernel time. Client timing includes the SDK and scalar response-header hook; evidence serialization is outside each request timer but still consumes inter-request time.

Numerical validity requires exact item count/order/IDs, class/group keys, chosen labels, model identity, local serving and no item errors. Every probability/confidence must be finite in[0,1]. One fixed absolute0.002 bound applies to untimed repeats, candidate-versus-baseline and every successful measured response checked offline. Classifications are compared by label; item order is exact. Any mismatch remains a failed equivalence result.

A reproduction witness requires an admitted, single-attempt heavy300s timeout with no successful server completion before that timeout, while cheap work completes in its final30s. The auditor retains cancellations, late work and unresolved requests. Without baseline reproduction there is no fix claim. One same-device ordered pair is descriptive; intervals and requests are not independent hardware replications. This does not establish generic scheduler fairness, behavior of other model profiles, Rust-sidecar behavior or statistical non-inferiority.

From an initialized public checkout containing the two pinned Git commits:

```console
git fetch origin a5e8d7b39be4b8156932d902d75fa46dc3e37e53
mise exec -- uv run --frozen python -m tools.benchmarks.gliclass_mixed.prepare --output /tmp/gliclass-inputs
mise exec -- uv run --frozen python -m tools.benchmarks.gliclass_mixed.model --output /tmp/gliclass-model
mise exec -- uv run --frozen python -m tools.benchmarks.gliclass_mixed.run --inputs /tmp/gliclass-inputs --output /tmp/gliclass-experiment --deadline-s 3900
mise exec -- uv run --frozen python -m tools.benchmarks.gliclass_mixed.analyze /tmp/gliclass-experiment --output /tmp/gliclass-analysis.json
```

For the whole-source progress comparison, fetch its two pinned commits and replace only the preparation command:

```console
git fetch origin f3b34e2dd88df74d4cc62c1446d28982c769f6ea 45794f69402d1c334c1238cfc13082cc0596e08f
mise exec -- uv run --frozen python -m tools.benchmarks.gliclass_mixed.prepare --comparison progress --output /tmp/gliclass-progress-inputs
```

Pass that inputs directory to the same runner. The workload, numerical bound, observations, timing, and cleanup limits are identical across comparison modes. Fixed legacy source IDs preserve the historical profile reproduction; progress source IDs identify two real Git snapshots.

The runner requires an external owned-process/provider deadline and durable artifact transfer; it does not allocate hardware. Preserve the complete output directory on failure. Model weights are cached separately and are not benchmark evidence. The planned operator has a4500s cap, with3900s runner,45s wait,90s cleanup,300s export and10s margin; it refuses runner admission with4345s or less remaining. Provider/outer caps are4800/5100s and startup is bounded to300s.

The CPU controls use a tiny random local GLiClass checkpoint, actual queued HTTP/SDK calls, the immutable source archives, timeout/late-completion observation and exact process cleanup. They do not qualify pretrained behavior or GPU performance:

```console
mise exec -- uv run --frozen pytest tools/benchmarks/gliclass_mixed/tests -q
```
