# Exact Full-mode matcher: implementation and local measurements

Full matching now evaluates each pair `i < j` once and inserts its score into
both neighborhoods. It preserves safe distance rejection, the final Float32
rounded dot-product threshold, diagonal eligibility, stable largest-first seed
ordering, greedy removal of overlaps, ascending member order, and final stable
largest-first ordering. This is neighborhood extraction, not connected components.

The worker uses zero-copy views over one transferred embedding buffer. No worker
pool or per-core embedding copies were added. The existing packing copy on the
main thread remains. At threshold 0.99 only above-threshold neighbor indices are
retained; there is no similarity matrix or score array for rejected pairs.

As explicitly approved, exact-threshold ties preserve the legacy top-k cutoff
quirk. When the threshold is exactly representable as Float32 (notably 1), sparse
scores that survive early rejection but round just below the threshold are also
retained. They can affect legacy top-k tie selection. A single reusable O(N)
scratch row reconstructs scores only for neighborhoods whose ties are truncated.
The legacy top-k helper can additionally allocate two O(N) typed scratch arrays.
No pair is reevaluated during selection.

Detection progress counts `N*(N-1)/2 + 3*N` checks: unordered pairs, diagonal
checks, neighborhood selections, and greedy assignment steps. The total is known
before worker startup. Time checks occur between bounded 512-pair slices, with
progress/yields targeted about every 32 ms. Full-mode UI counts are labeled
“checks completed”; Smart-mode labels and behavior stay unchanged. Abort still
terminates the worker immediately, and the shared matcher also supports an
AbortSignal at its cooperative checkpoints.

## Reproduce the synthetic benchmark

```sh
node tools/benchmark-full-matcher.mjs
node tools/benchmark-full-matcher.mjs --n=8192
```

Optional flags: `--n`, `--dim`, `--threshold`, `--repeats`. Defaults are 4096,
1024, 0.99, and 3. The runner bundles into an OS temporary directory and cleans
up that directory. It never loads extension state, embeddings/media caches,
Google Photos, or image generation/download code.

Each variant runs in a separate Node process. Each process warms up its matcher
on 160 vectors, creates deterministic signed normalized vectors using seed 42,
and injects 40 independent groups of sizes 2–5. The timed section covers matching
and community extraction, including normal progress callbacks and event-loop
yields. Fixture generation, bundling, and process startup are excluded. Run order
alternates. SHA-256 digests check exact group membership and ordering between
variants and repetitions. Both variants returned 40 groups containing 140 items.

The directed baseline is a retained test-only copy of the previous worker,
including its distance rejection, batched Float32 scores, top-k implementation,
and community extraction. Its off-diagonal counts are the exact loop cardinality;
the triangle matcher also records completed comparisons. Unit tests independently
spy on every pair evaluation to detect repeats or omitted pairs.

## Measurements

Local Node v22.14.0, macOS arm64, Apple M3; 1024 dimensions, threshold 0.99,
three isolated runs per variant. These are synthetic matcher results, not
application-wide or real-library scan measurements.

| Embeddings | Directed off-diagonal comparisons | Triangle comparisons | Directed median | Triangle median | Matcher speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 16,773,120 | 8,386,560 | 409.1 ms | 189.7 ms | 2.16× |
| 8,192 | 67,100,672 | 33,550,336 | 1,625.9 ms | 956.7 ms | 1.70× |

Both variants additionally check the diagonal once per embedding. Individual
8,192-vector times were 1649.1/1587.6/1625.9 ms directed and
1124.4/956.7/948.9 ms triangle. Timing varies with JIT warmup, memory locality,
garbage collection, and machine load; halving comparisons does not guarantee
halving elapsed time.

| Memory data, 8,192 embeddings | Directed | Triangle |
| --- | ---: | ---: |
| Embedding payload | 32 MiB | 32 MiB |
| Squared norms | 64 KiB | 64 KiB |
| One batch similarity buffer | 4 MiB | 0 |
| Stored sparse match entries, including diagonal | — | 8,592 |
| Stored boundary entries / tie scratch | — | 0 / 0 |
| Median process high-water RSS | 137.4 MiB | 102.4 MiB |
| Median observed ArrayBuffer peak | 68.7 MiB | 32.7 MiB |
| Median observed JS heap peak | 7.9 MiB | 11.9 MiB |

RSS is the OS whole-process high-water mark, including runtime, fixture generation,
and warmup. Heap and ArrayBuffer peaks are samples taken at progress callbacks;
they may miss transient peaks and have different sampling intervals between
variants. Sparse JS arrays increase heap usage while eliminating batch score
buffers; their capacity/overhead is runtime-dependent. These numbers are not
exact matcher-only memory peaks. Raw per-run data for both sizes is retained in
[full-matcher-benchmark-results.json](./full-matcher-benchmark-results.json).

## Real-library baseline and remaining bottlenecks

The supplied real-library baseline remains **48,561 cached embeddings, 565 ms
loading, 213,229 ms detection, 214,088 ms total, 58 groups / 122 items**. No live
scan was run, so there is no measured post-change real-library timing and no
claim of a 2× application-wide improvement.

For that embedding count, the new loop has exactly **1,179,061,080 unordered
pairs**, plus 48,561 diagonal checks. Full mode remains quadratic. Distance
checks still read embedding components for every pair, and possible matches
still compute the full dot product. Dense duplicate neighborhoods can require
O(E) storage approaching O(N²), and their greedy extraction remains expensive.
Exact-threshold tie replay retains the old top-k costs. Buffer packing and
embedding loading remain unchanged. No approximate search, threshold changes,
or false-negative tradeoffs were introduced.

## Validation

- Complete repository unit suite: `npm test` — **18 files, 292 tests passed**.
- Production build: `npm run build` — **passed**, including worker regeneration.
- `git diff --check` — **passed**.
- Typecheck excluding the untouched toolkit submodule — **passed**.
- An additional repository-wide `tsc --noEmit --incremental false --project
  tsconfig.test.json` reports three existing errors in the untouched
  `Google-Photos-Toolkit` submodule: two in `getFormData.ts`, one in
  `parseDateFromFilename.test.ts`. They do not prevent the production build.
- Build warnings remain for Browserslist data, a newer Plasmo version, and
  optional `svgo`; dependencies were not changed.

Coverage includes random normalized embeddings, exact duplicates, threshold
boundaries, independent groups, overlapping/transitive neighborhoods, no matches,
large groups, exact-threshold truncation, expanded cutoff state, subthreshold
boundary tie replay, self-membership, pair counts, cancellation before/during
matching and grouping, worker termination, zero-copy worker views, progress
monotonicity/completion, and the Full progress label.

No Google Photos access, live scan, cloud media changes, or cache modifications
were performed. Existing staged changes were preserved; nothing was staged or
committed by this implementation.

## Files changed by this implementation

- `lib/full-matcher.ts` — shared exact upper-triangle matcher and progress totals.
- `lib/cosine-threshold.ts` — reusable scalar predicate with unchanged semantics.
- `lib/duplicate-detector.ts` — initial Full total and testable worker wrapper.
- `workers/embedder.worker.ts` — Full dispatch to the shared matcher.
- `scripts/embedder-worker.js` — regenerated production worker.
- `components/ScanProgress.tsx` — optional count label, old label remains default.
- `tabs/app.tsx` — apply the checks label only to Full detection.
- `tests/components/scan-progress.test.tsx` — Full count-label coverage.
- `tests/helpers/directed-full-reference.ts` — frozen previous Full matcher.
- `tests/helpers/full-match-fixtures.ts` — deterministic synthetic vectors.
- `tests/lib/full-matcher.test.ts` — correctness, ordering, progress, cancellation.
- `tests/workers/full-detection.test.ts` — actual worker dispatch and row views.
- `tests/perf/full-matcher-benchmark.ts` — isolated synthetic benchmark entry.
- `tools/benchmark-full-matcher.mjs` — deterministic offline benchmark runner.
- `tests/perf/full-matcher-benchmark-results.json` — raw local measurements.
- `tests/perf/full-matcher-benchmark.md` — this implementation/measurement report.
