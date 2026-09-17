import { describe, expect, it } from "vitest"

import {
  computeSquaredNorms,
  thresholdedMatMul,
} from "../../lib/cosine-threshold"

function normalize(values: number[]): Float32Array {
  const norm = Math.sqrt(values.reduce((sum, value) => sum + value * value, 0))
  return new Float32Array(values.map((value) => value / norm))
}

function exactDot(a: Float32Array, b: Float32Array): number {
  let dot = 0
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i]
  return dot
}

describe("thresholdedMatMul", () => {
  it.each([0.9, 0.95, 0.99, 1])(
    "preserves exact threshold decisions at threshold %f",
    (threshold) => {
      const embeddings = Array.from({ length: 40 }, (_, row) =>
        normalize(
          Array.from(
            { length: 128 },
            (_, column) =>
              Math.sin((row + 1) * (column + 3)) +
              Math.cos((row + 7) * (column + 1))
          )
        )
      )
      // Include exact and near duplicates as well as unrelated vectors.
      embeddings.push(new Float32Array(embeddings[0]))
      embeddings.push(
        normalize(Array.from(embeddings[0], (value, i) => value + (i % 2 ? 0.001 : -0.001)))
      )

      const norms = computeSquaredNorms(embeddings)
      const scores = thresholdedMatMul(
        embeddings,
        0,
        embeddings.length,
        embeddings,
        0,
        embeddings.length,
        embeddings[0].length,
        threshold,
        norms,
        norms
      )

      for (let i = 0; i < embeddings.length; i++) {
        for (let j = 0; j < embeddings.length; j++) {
          // The legacy worker stores matrix scores in Float32Array before
          // thresholding, so compare against the same rounded representation.
          const exact = Math.fround(exactDot(embeddings[i], embeddings[j]))
          expect(scores[i * embeddings.length + j] >= threshold).toBe(
            exact >= threshold
          )
        }
      }
    }
  )

  it("respects row slices", () => {
    const embeddings = [
      normalize([1, 0, 0]),
      normalize([0, 1, 0]),
      normalize([0, 0, 1]),
    ]
    const norms = computeSquaredNorms(embeddings)
    const scores = thresholdedMatMul(
      embeddings,
      1,
      2,
      embeddings,
      0,
      3,
      3,
      0.99,
      norms,
      norms
    )

    expect(Array.from(scores).map((score) => score >= 0.99)).toEqual([
      false,
      true,
      false,
    ])
  })
})
