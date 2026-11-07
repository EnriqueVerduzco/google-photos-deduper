import { afterEach, beforeEach, expect, it, vi } from "vitest"

import type { GpdMediaItem } from "../../lib/types"

let cleanup: (() => void) | undefined
const items = (count: number): GpdMediaItem[] =>
  Array.from({ length: count }, (_, i) => ({
    mediaKey: `item-${i}`,
    dedupKey: `d${i}`,
    thumb: `https://local/item-${i}`,
    timestamp: i,
    creationTimestamp: i
  }))
beforeEach(() => {
  vi.resetModules()
  vi.stubGlobal("indexedDB", undefined)
  vi.stubGlobal("navigator", { hardwareConcurrency: 8 })
  vi.stubGlobal("chrome", { runtime: { getURL: (path: string) => path } })
})
afterEach(() => {
  cleanup?.()
  cleanup = undefined
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

it("embeds completed batches before all downloads finish and preserves source ordering", async () => {
  let finishLast!: () => void,
    embeddedBatches = 0,
    workerCount = 0
  const held = new Promise<void>((resolve) => {
    finishLast = resolve
  })
  const fetch = vi.fn(async (url: string) => {
    if (url.includes("tflite"))
      return { ok: true, arrayBuffer: async () => new ArrayBuffer(4) }
    const index = Number(/item-(\d+)/.exec(url)![1])
    if (index === 40) await held
    return { ok: true, blob: async () => new Blob([String(index)]) }
  })
  vi.stubGlobal("fetch", fetch)
  vi.stubGlobal(
    "Worker",
    class {
      onmessage: any
      onerror: any
      constructor() {
        workerCount++
      }
      terminate() {}
      postMessage(message: any, transfer?: ArrayBuffer[]) {
        if (message.type === "init") {
          expect(transfer?.[0]).toBe(message.data.modelBuffer)
          queueMicrotask(() => this.onmessage({ data: { type: "ready" } }))
        } else {
          embeddedBatches++
          void Promise.all(
            message.data.items.map(async ({ localIdx, blob }: any) => ({
              localIdx,
              embedding: new Float32Array([Number(await blob.text()), 1]).buffer
            }))
          ).then((results) =>
            this.onmessage({ data: { type: "results", results } })
          )
        }
      }
    }
  )
  const { prepareEmbeddings } = await import("../../lib/embedding-pipeline")
  cleanup = (await import("../../lib/embedding-worker-pool"))
    .disposeEmbeddingWorkers
  const progress = vi.fn()
  const pending = prepareEmbeddings(items(41), progress)
  await vi.waitFor(() => expect(embeddedBatches).toBeGreaterThan(0))
  finishLast()
  const result = await pending
  expect(result.validIndices).toEqual(Array.from({ length: 41 }, (_, i) => i))
  expect(result.embeddings.map((row) => row[0])).toEqual(result.validIndices)
  expect(workerCount).toBe(3)
  expect(progress.mock.calls.at(-1)?.[0]).toMatchObject({
    phase: "computing_embeddings",
    current: 41,
    total: 41
  })
  const previousWorkers = workerCount
  await prepareEmbeddings(items(41))
  expect(workerCount).toBeLessThanOrEqual(previousWorkers + 1)
  expect(
    fetch.mock.calls.filter(([url]) => url.includes("tflite"))
  ).toHaveLength(1)
})

it("cancels promptly even while a model worker has not initialized", async () => {
  const terminate = vi.fn()
  let constructed = 0
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) =>
      url.includes("tflite")
        ? { ok: true, arrayBuffer: async () => new ArrayBuffer(4) }
        : { ok: true, blob: async () => new Blob(["image"]) }
    )
  )
  vi.stubGlobal(
    "Worker",
    class {
      onmessage: any
      onerror: any
      constructor() {
        constructed++
      }
      postMessage() {}
      terminate = terminate
    }
  )
  const { prepareEmbeddings } = await import("../../lib/embedding-pipeline")
  cleanup = (await import("../../lib/embedding-worker-pool"))
    .disposeEmbeddingWorkers
  const controller = new AbortController()
  const result = prepareEmbeddings(items(2), undefined, controller.signal)
  const assertion = expect(result).rejects.toMatchObject({ name: "AbortError" })
  await vi.waitFor(() => expect(constructed).toBe(1))
  controller.abort()
  await assertion
})
