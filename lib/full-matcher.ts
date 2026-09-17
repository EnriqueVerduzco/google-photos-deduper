import { computeSquaredNorms, thresholdedCosine } from "./cosine-threshold"
import { topK } from "./top-k"

export interface FullMatchStats {
  pairComparisons: number
  selfComparisons: number
  matchEntries: number
  boundaryEntries: number
  scratchBytes: number
}

// One unit per unordered pair, diagonal, neighborhood, and greedy assignment.
// Unlike a row counter, this stays proportional to pair work in the triangle.
export function fullDetectionWorkTotal(n: number): number {
  return (n * (n - 1)) / 2 + 3 * n
}

/**
 * Exact Full-mode neighborhoods, followed by legacy largest-first extraction.
 * Row views share the caller's embeddings; only sparse relationships are stored.
 * This is deliberately not union-find: transitive paths need not form a group.
 */
export async function fullCommunityDetection(
  embeddings: Float32Array[],
  threshold: number,
  options: {
    onProgress?: (current: number, total: number) => void
    signal?: AbortSignal
    stats?: FullMatchStats
  } = {}
): Promise<number[][]> {
  const { onProgress, signal, stats } = options
  const n = embeddings.length
  const total = fullDetectionWorkTotal(n)
  let completed = 0
  let lastYield = performance.now()
  signal?.throwIfAborted()
  onProgress?.(0, total)
  if (stats)
    Object.assign(stats, {
      pairComparisons: 0,
      selfComparisons: 0,
      matchEntries: 0,
      boundaryEntries: 0,
      scratchBytes: 0
    })
  if (n < 2) {
    signal?.throwIfAborted()
    onProgress?.(total, total)
    return []
  }

  const pause = async () => {
    signal?.throwIfAborted()
    onProgress?.(completed, total)
    await new Promise<void>((resolve) => setTimeout(resolve, 0))
    signal?.throwIfAborted()
    lastYield = performance.now()
  }
  const dim = embeddings[0].length
  const norms = computeSquaredNorms(embeddings)
  const neighbors: number[][] = Array.from({ length: n }, () => [])

  // Float32 scores cannot equal non-representable thresholds (e.g. 0.99).
  // Only exact ties can trigger the legacy strict-'>' top-k expansion quirk.
  const canTie = Math.fround(threshold) === threshold
  const scores: number[][] | undefined = canTie
    ? Array.from({ length: n }, () => [])
    : undefined
  const boundaryIndices: number[][] | undefined = canTie
    ? Array.from({ length: n }, () => [])
    : undefined
  const boundaryScores: number[][] | undefined = canTie
    ? Array.from({ length: n }, () => [])
    : undefined
  const rejectedScore = Math.fround(threshold - 1)

  const remember = (i: number, j: number, score: number) => {
    if (score >= threshold) {
      neighbors[i].push(j)
      scores?.[i].push(score)
      if (stats) stats.matchEntries++
    } else if (canTie && score !== rejectedScore) {
      // Narrow compatibility exception: surviving scores just below threshold
      // affect legacy topK tie selection. Rejected pairs are never retained.
      boundaryIndices![i].push(j)
      boundaryScores![i].push(score)
      if (stats) stats.boundaryEntries++
    }
  }

  // Keep the hot loop synchronous and small so V8 can optimize it separately
  // from the async orchestration. Bounded slices allow frequent time checks
  // without a clock read, counter update, or await branch on every pair.
  const compareSlice = (i: number, start: number, end: number) => {
    const row = embeddings[i]
    const norm = norms[i]
    for (let j = start; j < end; j++) {
      const score = thresholdedCosine(
        row,
        embeddings[j],
        dim,
        threshold,
        norm,
        norms[j]
      )
      if (score >= threshold || (canTie && score !== rejectedScore)) {
        remember(i, j, score)
        remember(j, i, score)
      }
    }
  }

  for (let i = 0; i < n; i++) {
    signal?.throwIfAborted()
    // Do not assume self=1: Float32 normalization and rejection at threshold 1
    // can exclude the diagonal. Use exactly the same predicate as other pairs.
    remember(
      i,
      i,
      thresholdedCosine(
        embeddings[i],
        embeddings[i],
        dim,
        threshold,
        norms[i],
        norms[i]
      )
    )
    if (stats) stats.selfComparisons++
    completed++
    for (let start = i + 1; start < n; start += 512) {
      const end = Math.min(start + 512, n)
      compareSlice(i, start, end)
      completed += end - start
      if (stats) stats.pairComparisons += end - start
      if (performance.now() - lastYield >= 32) await pause()
    }
  }

  let sortMaxSize = Math.min(50, n)
  let scratch: Float32Array | undefined
  for (let i = 0; i < n; i++) {
    signal?.throwIfAborted()
    const members = neighbors[i]
    if (canTie && members.length >= 2) {
      let strictlyAbove = 0
      for (const score of scores![i]) if (score > threshold) strictlyAbove++
      while (strictlyAbove >= sortMaxSize && sortMaxSize < n) {
        sortMaxSize = Math.min(2 * sortMaxSize, n)
      }
      if (members.length > sortMaxSize) {
        // Replay topK only when its cutoff truncates threshold-equal ties.
        // A single O(n) scratch row, never a dense n-by-n similarity matrix.
        scratch ??= new Float32Array(n)
        scratch.fill(rejectedScore)
        for (let k = 0; k < members.length; k++) {
          scratch[members[k]] = scores![i][k]
        }
        for (let k = 0; k < boundaryIndices![i].length; k++) {
          scratch[boundaryIndices![i][k]] = boundaryScores![i][k]
        }
        neighbors[i] = topK(scratch, sortMaxSize).indices.sort((a, b) => a - b)
        // topK uses at most two n-element 32-bit scratch buffers internally.
        if (stats)
          stats.scratchBytes =
            n * 4 + (sortMaxSize > 50 ? n * 8 : sortMaxSize * 8)
      }
    }
    // Release boundary data as soon as its row has been selected.
    if (scores) {
      scores[i] = []
      boundaryIndices![i] = []
      boundaryScores![i] = []
    }
    completed++
    if (performance.now() - lastYield >= 32) await pause()
  }

  // Stable sorting keeps seed order for equally sized neighborhoods, just as
  // the directed matcher. Neighbor indices were inserted in ascending order.
  neighbors.sort((a, b) => b.length - a.length)
  const assigned = new Uint8Array(n)
  const groups: number[][] = []
  for (const community of neighbors) {
    signal?.throwIfAborted()
    if (community.length >= 2) {
      const remaining = community.filter((index) => !assigned[index])
      if (remaining.length >= 2) {
        groups.push(remaining)
        for (const index of remaining) assigned[index] = 1
      }
    }
    completed++
    if (performance.now() - lastYield >= 32) await pause()
  }
  groups.sort((a, b) => b.length - a.length)
  signal?.throwIfAborted()
  onProgress?.(total, total)
  return groups
}
