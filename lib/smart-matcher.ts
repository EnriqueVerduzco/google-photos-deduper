/** Exact Smart-mode comparisons: possible matches retain the original double dot product. */
export function smartMatches(
  a: Float32Array,
  b: Float32Array,
  threshold: number,
  normA: number,
  normB: number
): boolean {
  const limit = normA + normB - 2 * threshold
  // Bound accumulation/cancellation error conservatively. Non-finite inputs
  // fall through to the legacy dot product rather than being pruned.
  const guard = Math.max(
    1e-7,
    (normA + normB + Math.abs(2 * threshold)) * Number.EPSILON * a.length * 32
  )
  if (Number.isFinite(limit) && Number.isFinite(guard)) {
    let distance = 0
    for (let start = 0; start < a.length; start += 16) {
      for (let k = start; k < Math.min(start + 16, a.length); k++) {
        const d = a[k] - b[k]
        distance += d * d
      }
      if (distance > limit + guard) return false
    }
  }
  let dot = 0
  for (let k = 0; k < a.length; k++) dot += a[k] * b[k]
  return dot >= threshold
}

export function detectSmartBucket(
  rows: Float32Array[],
  threshold: number
): number[][] {
  const parent = rows.map((_, i) => i)
  const find = (index: number): number => {
    let root = index
    while (parent[root] !== root) root = parent[root]
    while (parent[index] !== index) {
      const next = parent[index]
      parent[index] = root
      index = next
    }
    return root
  }
  const norms = rows.map((row) => {
    let norm = 0
    for (let k = 0; k < row.length; k++) norm += row[k] * row[k]
    return norm
  })
  for (let i = 0; i < rows.length; i++)
    for (let j = i + 1; j < rows.length; j++)
      if (
        find(i) !== find(j) &&
        smartMatches(rows[i], rows[j], threshold, norms[i], norms[j])
      )
        parent[find(i)] = find(j)
  const components = new Map<number, number[]>()
  for (let i = 0; i < rows.length; i++) {
    const root = find(i)
    if (!components.has(root)) components.set(root, [])
    components.get(root)!.push(i)
  }
  return [...components.values()].filter((group) => group.length >= 2)
}
