// Frozen pre-upper-triangle Full worker matcher. Tests and local benchmarks only.
// Keep selection, Float32 rounding, and strict expansion semantics independent
// of the production implementation so regressions cannot change both sides.

const DISTANCE_EPSILON = 1e-7;
const DISTANCE_BLOCK_SIZE = 16;

function computeSquaredNorms(
  embeddings: Float32Array[],
): Float64Array {
  const norms = new Float64Array(embeddings.length);
  for (let i = 0; i < embeddings.length; i++) {
    let norm = 0;
    const embedding = embeddings[i];
    for (let k = 0; k < embedding.length; k++) {
      norm += embedding[k] * embedding[k];
    }
    norms[i] = norm;
  }
  return norms;
}

/**
 * Compute only the cosine scores that can reach `threshold`.
 *
 * For any vectors a and b:
 *   ||a-b||² = ||a||² + ||b||² - 2(a·b)
 *
 * Squared-distance terms are non-negative, so once a partial distance exceeds
 * the maximum compatible with the cosine threshold, the remaining dimensions
 * cannot rescue that pair. Rejected pairs receive a below-threshold sentinel;
 * possible matches still use the original full dot product for the final
 * decision, preserving threshold behavior.
 */
function thresholdedMatMul(
  a: Float32Array[],
  startA: number,
  endA: number,
  b: Float32Array[],
  startB: number,
  endB: number,
  dim: number,
  threshold: number,
  normsA: Float64Array,
  normsB: Float64Array,
): Float32Array {
  const rowsA = endA - startA;
  const rowsB = endB - startB;
  const result = new Float32Array(rowsA * rowsB);
  const rejectedScore = threshold - 1;

  for (let i = 0; i < rowsA; i++) {
    const aIndex = startA + i;
    const aRow = a[aIndex];
    for (let j = 0; j < rowsB; j++) {
      const bIndex = startB + j;
      const bRow = b[bIndex];
      const maxSquaredDistance =
        normsA[aIndex] + normsB[bIndex] - 2 * threshold;
      let squaredDistance = 0;
      let rejected = maxSquaredDistance < -DISTANCE_EPSILON;

      for (
        let blockStart = 0;
        !rejected && blockStart < dim;
        blockStart += DISTANCE_BLOCK_SIZE
      ) {
        const blockEnd = Math.min(blockStart + DISTANCE_BLOCK_SIZE, dim);
        for (let k = blockStart; k < blockEnd; k++) {
          const difference = aRow[k] - bRow[k];
          squaredDistance += difference * difference;
        }
        if (squaredDistance > maxSquaredDistance + DISTANCE_EPSILON) {
          rejected = true;
        }
      }

      if (rejected) {
        result[i * rowsB + j] = rejectedScore;
        continue;
      }

      // Preserve the original dot-product comparison for possible matches.
      let dot = 0;
      for (let k = 0; k < dim; k++) dot += aRow[k] * bRow[k];
      result[i * rowsB + j] = dot;
    }
  }

  return result;
}

/**
 * Find the k largest values and their indices.
 *
 * Small selections use a bounded min-heap, avoiding the O(n log n) full sort
 * and per-entry object allocation that made large Full scans impractical.
 * Larger selections use quickselect followed by a sort of only the selected
 * partition.
 */
function topK(
  arr: Float32Array,
  k: number,
): { values: number[]; indices: number[] } {
  const n = arr.length;
  k = Math.min(k, n);
  if (k <= 0) return { values: [], indices: [] };

  if (k <= 50) {
    const hVals = new Float32Array(k);
    const hIdxs = new Uint32Array(k);
    let size = 0;

    const siftDown = (pos: number) => {
      while (true) {
        let smallest = pos;
        const left = 2 * pos + 1;
        const right = left + 1;
        if (left < size && hVals[left] < hVals[smallest]) smallest = left;
        if (right < size && hVals[right] < hVals[smallest]) smallest = right;
        if (smallest === pos) break;

        const value = hVals[pos];
        hVals[pos] = hVals[smallest];
        hVals[smallest] = value;
        const index = hIdxs[pos];
        hIdxs[pos] = hIdxs[smallest];
        hIdxs[smallest] = index;
        pos = smallest;
      }
    };

    for (let i = 0; i < n; i++) {
      const value = arr[i];
      if (size < k) {
        hVals[size] = value;
        hIdxs[size] = i;
        size++;
        for (let pos = (size >> 1) - 1; pos >= 0; pos--) siftDown(pos);
      } else if (value > hVals[0]) {
        hVals[0] = value;
        hIdxs[0] = i;
        siftDown(0);
      }
    }

    const values: number[] = new Array(size);
    const indices: number[] = new Array(size);
    for (let i = size - 1; i >= 0; i--) {
      values[i] = hVals[0];
      indices[i] = hIdxs[0];
      hVals[0] = hVals[--size];
      hIdxs[0] = hIdxs[size];
      siftDown(0);
    }
    return { values, indices };
  }

  const valuesBuffer = new Float32Array(n);
  const indicesBuffer = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    valuesBuffer[i] = arr[i];
    indicesBuffer[i] = i;
  }

  let low = 0;
  let high = n - 1;
  while (low < high) {
    const pivot = valuesBuffer[high];
    let partition = low;
    for (let i = low; i < high; i++) {
      if (valuesBuffer[i] >= pivot) {
        let value = valuesBuffer[partition];
        valuesBuffer[partition] = valuesBuffer[i];
        valuesBuffer[i] = value;
        let index = indicesBuffer[partition];
        indicesBuffer[partition] = indicesBuffer[i];
        indicesBuffer[i] = index;
        partition++;
      }
    }

    let value = valuesBuffer[partition];
    valuesBuffer[partition] = valuesBuffer[high];
    valuesBuffer[high] = value;
    let index = indicesBuffer[partition];
    indicesBuffer[partition] = indicesBuffer[high];
    indicesBuffer[high] = index;

    if (partition === k - 1) break;
    if (partition < k - 1) low = partition + 1;
    else high = partition - 1;
  }

  // Sort only the selected partition into descending order.
  for (let i = 1; i < k; i++) {
    const value = valuesBuffer[i];
    const index = indicesBuffer[i];
    let j = i - 1;
    while (j >= 0 && valuesBuffer[j] < value) {
      valuesBuffer[j + 1] = valuesBuffer[j];
      indicesBuffer[j + 1] = indicesBuffer[j];
      j--;
    }
    valuesBuffer[j + 1] = value;
    indicesBuffer[j + 1] = index;
  }

  const values: number[] = new Array(k);
  const indices: number[] = new Array(k);
  for (let i = 0; i < k; i++) {
    values[i] = valuesBuffer[i];
    indices[i] = indicesBuffer[i];
  }
  return { values, indices };
}

export async function directedFullReference(
  embeddings: Float32Array[],
  threshold: number,
  _timestamps?: number[],
  onProgress?: (current: number, total: number) => void,
): Promise<number[][]> {
  const n = embeddings.length;
  if (n < 2) return [];
  const dim = embeddings[0].length;
  const batchSize = 128;
  const minCommunitySize = 2;
  const extractedCommunities: number[][] = [];
  let sortMaxSize = Math.min(Math.max(2 * minCommunitySize, 50), n);
  const squaredNorms = computeSquaredNorms(embeddings);

  for (let startIdx = 0; startIdx < n; startIdx += batchSize) {
    const endIdx = Math.min(startIdx + batchSize, n);
    const batchLen = endIdx - startIdx;

    // Compute cosine similarity: batch x all embeddings
    // Embeddings are L2-normalized so cos_sim = dot product
    const cosScores = thresholdedMatMul(
      embeddings,
      startIdx,
      endIdx,
      embeddings,
      0,
      n,
      dim,
      threshold,
      squaredNorms,
      squaredNorms,
    );

    for (let i = 0; i < batchLen; i++) {
      const row = cosScores.subarray(i * n, (i + 1) * n);

      // Quick check: are there at least minCommunitySize items above threshold?
      const topKMin = topK(row, minCommunitySize);
      if (topKMin.values[topKMin.values.length - 1] < threshold) continue;

      // Find all items above threshold
      let topKResult = topK(row, sortMaxSize);

      // Expand search window if needed
      while (
        topKResult.values[topKResult.values.length - 1] > threshold &&
        sortMaxSize < n
      ) {
        sortMaxSize = Math.min(2 * sortMaxSize, n);
        topKResult = topK(row, sortMaxSize);
      }

      const cluster: number[] = [];
      for (let j = 0; j < topKResult.values.length; j++) {
        if (topKResult.values[j] < threshold) break;
        cluster.push(topKResult.indices[j]);
      }

      if (cluster.length >= minCommunitySize) {
        extractedCommunities.push(cluster);
      }
    }

    onProgress?.(endIdx, n);
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
  }

  // Sort communities by size (largest first)
  extractedCommunities.sort((a, b) => b.length - a.length);

  // Remove overlapping communities (greedy: assign each item to largest community first)
  const uniqueCommunities: number[][] = [];
  const assignedIds = new Set<number>();

  for (const community of extractedCommunities) {
    const nonOverlapping = community
      .slice()
      .sort((a, b) => a - b)
      .filter((idx) => !assignedIds.has(idx));

    if (nonOverlapping.length >= minCommunitySize) {
      uniqueCommunities.push(nonOverlapping);
      for (const idx of nonOverlapping) assignedIds.add(idx);
    }
  }

  uniqueCommunities.sort((a, b) => b.length - a.length);
  return uniqueCommunities;
}
