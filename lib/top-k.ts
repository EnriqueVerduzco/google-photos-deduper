/**
 * Find the k largest values and their indices.
 *
 * Small selections use a bounded min-heap, avoiding the O(n log n) full sort
 * and per-entry object allocation that made large Full scans impractical.
 * Larger selections use quickselect followed by a sort of only the selected
 * partition.
 */
export function topK(
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
