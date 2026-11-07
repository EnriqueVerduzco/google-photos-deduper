import { describe, expect, it } from "vitest"

import { detectSmartBucket, smartMatches } from "../../lib/smart-matcher"

function dot(a: Float32Array, b: Float32Array) {
  let score = 0
  for (let i = 0; i < a.length; i++) score += a[i] * b[i]
  return score
}
function legacy(rows: Float32Array[], threshold: number) {
  const parent = rows.map((_, i) => i)
  const find = (i: number): number =>
    parent[i] === i ? i : (parent[i] = find(parent[i]))
  for (let i = 0; i < rows.length; i++)
    for (let j = i + 1; j < rows.length; j++)
      if (dot(rows[i], rows[j]) >= threshold) parent[find(i)] = find(j)
  const groups = new Map<number, number[]>()
  rows.forEach((_, i) => {
    const root = find(i)
    groups.set(root, [...(groups.get(root) ?? []), i])
  })
  return [...groups.values()].filter((group) => group.length >= 2)
}
function normalize(values: number[]) {
  const length = Math.hypot(...values)
  return new Float32Array(values.map((x) => x / length))
}

describe("Smart numeric compatibility", () => {
  it("matches the unrounded legacy dot product at and around exact boundaries", () => {
    for (const dim of [16, 65, 1024])
      for (let i = 0; i < 30; i++) {
        const a = normalize(
          Array.from({ length: dim }, (_, k) => Math.sin(i + k * 1.7))
        )
        const b = normalize(
          Array.from(a, (x, k) => x + Math.cos(i + k) * (i % 3 ? 0.001 : 0.1))
        )
        const score = dot(a, b),
          normA = dot(a, a),
          normB = dot(b, b)
        for (const threshold of [
          0.95,
          0.99,
          1,
          score,
          score - 1e-12,
          score + 1e-12
        ])
          expect(smartMatches(a, b, threshold, normA, normB)).toBe(
            score >= threshold
          )
      }
  })
  it("retains the exact component and member order of the legacy loop", () => {
    const rows = Array.from({ length: 70 }, (_, i) =>
      normalize(
        Array.from({ length: 128 }, (_, k) => Math.sin((i % 13) * 7 + k))
      )
    )
    for (const threshold of [0.95, 0.99, 1])
      expect(detectSmartBucket(rows, threshold)).toEqual(
        legacy(rows, threshold)
      )
  })
  it("handles zero, non-normalized and non-finite inputs conservatively", () => {
    const rows = [
      new Float32Array([0, 0]),
      new Float32Array([1e20, 1e20]),
      new Float32Array([1e20, -1e20]),
      new Float32Array([NaN, 1]),
      new Float32Array([Infinity, 0])
    ]
    for (const threshold of [0, 0.99, 1e39])
      expect(detectSmartBucket(rows, threshold)).toEqual(
        legacy(rows, threshold)
      )
  })
})
