const DISTANCE_EPSILON = 1e-7;
const DISTANCE_BLOCK_SIZE = 16;

export function computeSquaredNorms(
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
export function thresholdedMatMul(
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

  for (let i = 0; i < rowsA; i++) {
    const aIndex = startA + i;
    const aRow = a[aIndex];
    for (let j = 0; j < rowsB; j++) {
      const bIndex = startB + j;
      const bRow = b[bIndex];
      result[i * rowsB + j] = thresholdedCosine(
        aRow,
        bRow,
        dim,
        threshold,
        normsA[aIndex],
        normsB[bIndex],
      );
    }
  }

  return result;
}

/** Same safe distance rejection and Float32 rounding as the batched matcher. */
export function thresholdedCosine(
  aRow: Float32Array,
  bRow: Float32Array,
  dim: number,
  threshold: number,
  normA: number,
  normB: number,
): number {
  const maxSquaredDistance = normA + normB - 2 * threshold;
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

  if (rejected) return Math.fround(threshold - 1);

  // Preserve the original dot-product comparison for possible matches.
  let dot = 0;
  for (let k = 0; k < dim; k++) dot += aRow[k] * bRow[k];
  return Math.fround(dot);
}
