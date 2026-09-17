import { afterEach, describe, expect, it, vi } from "vitest"

import * as cosine from "../../lib/cosine-threshold"
import { runCommunityDetectionInWorker } from "../../lib/duplicate-detector"
import {
  fullCommunityDetection,
  fullDetectionWorkTotal,
  type FullMatchStats
} from "../../lib/full-matcher"
import { directedFullReference } from "../helpers/directed-full-reference"
import {
  arc,
  axis,
  normalize,
  seededEmbeddings
} from "../helpers/full-match-fixtures"

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

async function equivalent(embeddings: Float32Array[], threshold: number) {
  const expected = await directedFullReference(embeddings, threshold)
  const actual = await fullCommunityDetection(embeddings, threshold)
  expect(actual).toEqual(expected) // Includes both membership and ordering.
  return actual
}

describe("Full upper-triangle compatibility", () => {
  it.each([0.5, 0.9, 0.95, 0.99, 1])(
    "matches random normalized embeddings at %f",
    async (threshold) => {
      for (let seed = 1; seed <= 12; seed++) {
        const rows = seededEmbeddings(75, seed % 2 ? 8 : 65, seed)
        rows.push(rows[3].slice(), rows[12].slice(), rows[3].slice())
        await equivalent(rows, threshold)
      }
    }
  )

  it("handles empty and singleton input", async () => {
    await equivalent([], 0.99)
    await equivalent([axis(0)], 0.99)
  })

  it("keeps independent groups largest-first with stable ties and ascending members", async () => {
    const rows = [
      axis(0),
      axis(1),
      axis(2),
      axis(1),
      axis(2),
      axis(0),
      axis(2),
      axis(3)
    ]
    expect(await equivalent(rows, 0.99)).toEqual([
      [2, 4, 6],
      [0, 5],
      [1, 3]
    ])
  })

  it("keeps overlapping neighborhoods distinct from transitive connected components", async () => {
    // A--B--C--D--E is connected, but Full greedily selects B's neighborhood.
    expect(await equivalent([0, 0.1, 0.2, 0.3, 0.4].map(arc), 0.99)).toEqual([
      [0, 1, 2],
      [3, 4]
    ])
  })

  it("returns no groups for unrelated embeddings", async () => {
    expect(
      await equivalent(
        Array.from({ length: 64 }, (_, i) => axis(i)),
        0.99
      )
    ).toEqual([])
  })

  it.each([0.5, 0.99, 1])(
    "preserves Float32 decisions around threshold %f",
    async (threshold) => {
      const rows = [
        new Float32Array([1, 0]),
        ...[-2e-7, -3e-8, 0, 3e-8, 2e-7].map((delta) => {
          const x = Math.min(1, threshold + delta)
          return new Float32Array([x, Math.sqrt(1 - x * x)])
        })
      ]
      await equivalent(rows, threshold)
      const spy = vi.spyOn(cosine, "thresholdedCosine")
      await fullCommunityDetection(rows, threshold)
      const norms = cosine.computeSquaredNorms(rows)
      const directed = cosine.thresholdedMatMul(
        rows,
        0,
        rows.length,
        rows,
        0,
        rows.length,
        2,
        threshold,
        norms,
        norms
      )
      // Explicitly check the score predicate rather than only resulting groups.
      for (let i = 0; i < rows.length; i++)
        for (let j = i; j < rows.length; j++) {
          expect(
            Math.fround(
              spy.mock.results[i * rows.length - (i * (i - 1)) / 2 + j - i]
                .value
            ) >= threshold
          ).toBe(directed[i * rows.length + j] >= threshold)
        }
    }
  )

  it("retains the threshold-equal 50-member legacy cutoff", async () => {
    const groups = await equivalent(
      Array.from({ length: 140 }, () => axis(0)),
      1
    )
    expect(groups).toEqual([Array.from({ length: 50 }, (_, i) => i)])
  })

  it("expands large strictly-above-threshold groups", async () => {
    expect(
      await equivalent(
        Array.from({ length: 220 }, () => axis(0)),
        0.99
      )
    ).toEqual([Array.from({ length: 220 }, (_, i) => i)])
  })

  it("preserves the shared expanded top-k cutoff", async () => {
    // A first neighborhood expands the shared window beyond 50, followed by
    // exact-threshold ties, strict matches, and surviving subthreshold scores.
    const highNorm = seededEmbeddings(2000, 2).find(
      (row) => Math.fround(cosine.computeSquaredNorms([row])[0]) > 1
    )!
    expect(highNorm).toBeDefined()
    const rows = Array.from(
      { length: 60 },
      () => new Float32Array([...highNorm, 0, 0])
    )
    rows.push(...Array.from({ length: 160 }, () => axis(2, 4)))
    const groups = await equivalent(rows, 1)
    expect(groups.map((group) => group.length)).toEqual([100, 60])
  })

  it("replays exact tie selection with surviving subthreshold scores", async () => {
    for (let seed = 1; seed <= 8; seed++) {
      const rows = Array.from({ length: 140 }, (_, i) =>
        arc(Math.PI / 4 + Math.sin((i + 1) * seed) * 0.0001)
      )
      const stats = {} as FullMatchStats
      expect(await fullCommunityDetection(rows, 1, { stats })).toEqual(
        await directedFullReference(rows, 1)
      )
      expect(stats.boundaryEntries).toBeGreaterThan(0)
      expect(stats.scratchBytes).toBeGreaterThan(0)
    }
  })

  it("checks self-membership rather than unconditionally inserting the diagonal", async () => {
    const lowNorm = normalize([1, 1])
    expect(Math.fround(cosine.computeSquaredNorms([lowNorm])[0])).toBeLessThan(
      1
    )
    expect(await equivalent([lowNorm, lowNorm.slice()], 1)).toEqual([])
    expect(await equivalent([axis(0), axis(0)], 1)).toEqual([[0, 1]])
  })

  it("evaluates each unordered pair exactly once and keeps sparse storage", async () => {
    const rows = seededEmbeddings(90, 64)
    rows.push(rows[0].slice())
    const spy = vi.spyOn(cosine, "thresholdedCosine")
    const stats = {} as FullMatchStats
    await fullCommunityDetection(rows, 0.99, { stats })
    const seen = new Set<string>()
    for (const [a, b] of spy.mock.calls) {
      const i = rows.indexOf(a),
        j = rows.indexOf(b)
      expect(j).toBeGreaterThanOrEqual(i)
      const key = `${i},${j}`
      expect(seen.has(key)).toBe(false)
      seen.add(key)
    }
    expect(seen.size).toBe((rows.length * (rows.length + 1)) / 2)
    expect(stats.pairComparisons).toBe((rows.length * (rows.length - 1)) / 2)
    expect(stats.selfComparisons).toBe(rows.length)
    expect(stats.matchEntries).toBe(rows.length + 2)
    expect(stats.boundaryEntries).toBe(0)
    expect(stats.scratchBytes).toBe(0)
  })

  it("uses boundary-score storage only where legacy ties can occur", async () => {
    const rows = [normalize([1, 1]), normalize([1, 1]), axis(0, 2)]
    const stats = {} as FullMatchStats
    await fullCommunityDetection(rows, 1, { stats })
    expect(stats.boundaryEntries).toBeGreaterThan(0)
    await fullCommunityDetection(rows, 0.99, { stats })
    expect(stats.boundaryEntries).toBe(0)
  })
})

describe("Full progress and cancellation", () => {
  it.each([0, 1, 250])(
    "reports monotonic progress and an accurate total for %i rows",
    async (n) => {
      const updates: number[] = []
      const total = fullDetectionWorkTotal(n)
      await fullCommunityDetection(seededEmbeddings(n, 32), 0.99, {
        onProgress: (current, reportedTotal) => {
          expect(reportedTotal).toBe(total)
          expect(current).toBeGreaterThanOrEqual(updates.at(-1) ?? 0)
          expect(current).toBeLessThanOrEqual(total)
          updates.push(current)
        }
      })
      expect(updates[0]).toBe(0)
      expect(updates.at(-1)).toBe(total)
    }
  )

  it("reports progress inside a long row, with checkpoints about every 32ms", async () => {
    let now = 0
    vi.spyOn(performance, "now").mockImplementation(() => (now += 40))
    const updates: number[] = []
    await fullCommunityDetection(seededEmbeddings(800, 16), 0.99, {
      onProgress: (current) => updates.push(current)
    })
    expect(updates[1]).toBeLessThan(800)
    expect(updates.length).toBeGreaterThan(10)
    expect(updates).toEqual([...updates].sort((a, b) => a - b))
    expect(updates.at(-1)).toBe(fullDetectionWorkTotal(800))
  })

  it("honors cancellation before any comparisons", async () => {
    const controller = new AbortController()
    controller.abort()
    const spy = vi.spyOn(cosine, "thresholdedCosine")
    await expect(
      fullCommunityDetection([axis(0), axis(0)], 0.99, {
        signal: controller.signal
      })
    ).rejects.toMatchObject({ name: "AbortError" })
    expect(spy).not.toHaveBeenCalled()
  })

  it("yields to timer cancellation promptly while pair matching is active", async () => {
    const controller = new AbortController()
    const rows = seededEmbeddings(4000, 64)
    const stats = {} as FullMatchStats
    const start = performance.now()
    const timer = setTimeout(() => controller.abort(), 0)
    try {
      await expect(
        fullCommunityDetection(rows, 0.99, { signal: controller.signal, stats })
      ).rejects.toMatchObject({ name: "AbortError" })
      expect(performance.now() - start).toBeLessThan(1000)
      expect(stats.pairComparisons).toBeLessThan(
        (rows.length * (rows.length - 1)) / 2
      )
    } finally {
      clearTimeout(timer)
    }
  })

  it.each(["selection", "assignment"])(
    "honors cancellation during %s",
    async (stage) => {
      let now = 0
      vi.spyOn(performance, "now").mockImplementation(() => (now += 40))
      const rows = Array.from({ length: 80 }, () => axis(0))
      const comparisons = (rows.length * (rows.length + 1)) / 2
      const cutoff = comparisons + (stage === "assignment" ? rows.length : 0)
      const controller = new AbortController()
      const updates: number[] = []
      await expect(
        fullCommunityDetection(rows, 1, {
          signal: controller.signal,
          onProgress: (current) => {
            updates.push(current)
            if (current > cutoff) controller.abort()
          }
        })
      ).rejects.toMatchObject({ name: "AbortError" })
      expect(updates.at(-1)).toBeLessThan(fullDetectionWorkTotal(rows.length))
    }
  )

  it("terminates the actual worker wrapper immediately on abort", async () => {
    class TestWorker {
      static instance: TestWorker
      terminate = vi.fn()
      postMessage = vi.fn()
      onmessage?: (event: { data: unknown }) => void
      constructor() {
        TestWorker.instance = this
      }
    }
    vi.stubGlobal("Worker", TestWorker)
    const controller = new AbortController()
    const progress = vi.fn()
    const pending = runCommunityDetectionInWorker(
      [axis(0), axis(0)],
      0.99,
      [],
      "local-worker",
      progress,
      controller.signal
    )
    const worker = TestWorker.instance
    expect(worker.postMessage).toHaveBeenCalledOnce()
    worker.onmessage!({
      data: { type: "detectionProgress", current: 1, total: 7 }
    })
    expect(progress).toHaveBeenCalledWith({
      phase: "detecting_duplicates",
      current: 1,
      total: 7
    })
    controller.abort()
    expect(worker.terminate).toHaveBeenCalledOnce()
    await expect(pending).rejects.toMatchObject({ name: "AbortError" })
  })
})
