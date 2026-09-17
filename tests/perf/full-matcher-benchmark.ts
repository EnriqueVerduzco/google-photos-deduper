// Local synthetic benchmark entry, bundled/run by tools/benchmark-full-matcher.mjs.
// No extension initialization, cache access, network, or cloud media operations.
import assert from "node:assert/strict"
import { createHash } from "node:crypto"

import {
  fullCommunityDetection,
  type FullMatchStats
} from "../../lib/full-matcher"
import { directedFullReference } from "../helpers/directed-full-reference"
import { seededEmbeddings } from "../helpers/full-match-fixtures"

async function main() {
  const [mode, nArg, dimArg, thresholdArg] = process.argv.slice(2)
  const n = Number(nArg),
    dim = Number(dimArg),
    threshold = Number(thresholdArg)
  assert(mode === "directed" || mode === "triangle")
  const warmup = seededEmbeddings(160, dim)
  if (mode === "directed") await directedFullReference(warmup, threshold)
  else await fullCommunityDetection(warmup, threshold)

  const rows = seededEmbeddings(n, dim)
  // Deterministic independent groups of sizes 2..5 among unrelated vectors.
  let tail = n - 1
  const groupCount = Math.min(40, Math.floor(n / 8))
  for (let group = 0; group < groupCount; group++) {
    for (let copy = 0; copy < 1 + (group % 4); copy++)
      rows[tail--] = rows[group].slice()
  }
  global.gc?.()
  const before = process.memoryUsage()
  const stats = {} as FullMatchStats
  let observedHeapPeak = before.heapUsed
  let observedArrayBufferPeak = before.arrayBuffers
  const sample = () => {
    const memory = process.memoryUsage()
    observedHeapPeak = Math.max(observedHeapPeak, memory.heapUsed)
    observedArrayBufferPeak = Math.max(
      observedArrayBufferPeak,
      memory.arrayBuffers
    )
  }
  const start = performance.now()
  const groups =
    mode === "directed"
      ? await directedFullReference(rows, threshold, undefined, sample)
      : await fullCommunityDetection(rows, threshold, {
          onProgress: sample,
          stats
        })
  const elapsedMs = performance.now() - start
  sample()
  console.log(
    JSON.stringify({
      mode,
      n,
      dim,
      threshold,
      elapsedMs,
      offDiagonalComparisons:
        mode === "directed" ? n * (n - 1) : stats.pairComparisons,
      selfComparisons: mode === "directed" ? n : stats.selfComparisons,
      groups: groups.length,
      groupedItems: groups.reduce((sum, group) => sum + group.length, 0),
      // Parent checks exact membership AND ordering between independent processes.
      groupDigest: createHash("sha256")
        .update(JSON.stringify(groups))
        .digest("hex"),
      embeddingBytes: n * dim * 4,
      normBytes: n * 8,
      legacyBatchScoreBytes: mode === "directed" ? Math.min(128, n) * n * 4 : 0,
      matchEntries: mode === "triangle" ? stats.matchEntries : null,
      boundaryEntries: mode === "triangle" ? stats.boundaryEntries : null,
      tieScratchBytes: mode === "triangle" ? stats.scratchBytes : null,
      heapBeforeBytes: before.heapUsed,
      observedHeapPeakBytes: observedHeapPeak,
      arrayBuffersBeforeBytes: before.arrayBuffers,
      observedArrayBufferPeakBytes: observedArrayBufferPeak,
      // OS high-water RSS includes fixture creation, warmup, runtime, and matcher.
      processMaxRssKiB: process.resourceUsage().maxRSS
    })
  )
}
main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})
