import { afterEach, expect, it, vi } from "vitest"

import { detectSmartBucket } from "../../lib/smart-matcher"
import { runSmartWorkerPool } from "../../lib/smart-worker-pool"

afterEach(() => vi.unstubAllGlobals())
it("parallelizes independent jobs and restores source order despite out-of-order completion", async () => {
  let active = 0,
    peak = 0
  const workers: FakeWorker[] = []
  class FakeWorker {
    onmessage: any
    onerror: any
    terminate = vi.fn()
    constructor() {
      workers.push(this)
    }
    postMessage({ data }: any, transfer: ArrayBuffer[]) {
      expect(transfer[0]).toBe(data.flatEmbeddings.buffer)
      active++
      peak = Math.max(peak, active)
      const rows = Array.from({ length: data.n }, (_, i) =>
        data.flatEmbeddings.subarray(i * data.dim, (i + 1) * data.dim)
      )
      const groups = data.buckets.flatMap((bucket: number[]) =>
        detectSmartBucket(
          bucket.map((i) => rows[i]),
          data.threshold
        ).map((group) => group.map((i) => bucket[i]))
      )
      setTimeout(
        () => {
          active--
          this.onmessage({ data: { type: "detectionResults", groups } })
        },
        data.mediaKeys[0] === "k0" ? 10 : 0
      )
    }
  }
  vi.stubGlobal("Worker", FakeWorker)
  vi.stubGlobal("navigator", { hardwareConcurrency: 8 })
  const rows = Array.from({ length: 600 }, () => new Float32Array([1, 0]))
  const buckets = Array.from({ length: 300 }, (_, i) => [i * 2, i * 2 + 1])
  const progress = vi.fn()
  const result = await runSmartWorkerPool(
    rows,
    rows.map((_, i) => `k${i}`),
    0.99,
    buckets,
    "local",
    progress
  )
  expect(result).toEqual(buckets)
  expect(peak).toBeGreaterThan(1)
  expect(
    workers.every((worker) => worker.terminate.mock.calls.length === 1)
  ).toBe(true)
  expect(progress.mock.calls.at(-1)?.[0]).toMatchObject({
    current: 300,
    total: 300
  })
})
it("terminates every matching job on cancellation", async () => {
  const terminate = vi.fn()
  vi.stubGlobal(
    "Worker",
    class {
      postMessage() {}
      terminate = terminate
    }
  )
  const controller = new AbortController()
  const pending = runSmartWorkerPool(
    [new Float32Array([1]), new Float32Array([1])],
    ["a", "b"],
    1,
    [[0, 1]],
    "local",
    undefined,
    controller.signal
  )
  controller.abort()
  await expect(pending).rejects.toMatchObject({ name: "AbortError" })
  expect(terminate).toHaveBeenCalledOnce()
})
