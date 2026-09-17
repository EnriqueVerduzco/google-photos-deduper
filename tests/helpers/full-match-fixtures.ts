export function normalize(values: number[]): Float32Array {
  const norm = Math.hypot(...values)
  return Float32Array.from(values, (value) => value / norm)
}

export function seededEmbeddings(
  n: number,
  dim: number,
  seed = 42
): Float32Array[] {
  let state = seed >>> 0
  return Array.from({ length: n }, () =>
    normalize(
      Array.from({ length: dim }, () => {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0
        return (state / 0x100000000) * 2 - 1
      })
    )
  )
}

export function axis(index: number, dim = 64): Float32Array {
  const row = new Float32Array(dim)
  row[index] = 1
  return row
}

export function arc(angle: number): Float32Array {
  return new Float32Array([Math.cos(angle), Math.sin(angle)])
}
