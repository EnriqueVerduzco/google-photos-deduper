import type { DetectionProgress } from "./duplicate-detector"
import {
  openEmbeddingStore,
  readEmbeddings,
  writeEmbeddings
} from "./embedding-store"
import { EmbeddingWorkerPool } from "./embedding-worker-pool"
import type { EmbeddingInput } from "./embedding-worker-pool"
import type { ScanLogger } from "./scan-log"
import { fetchThumbnails } from "./thumbnail-downloads"
import type { GpdMediaItem } from "./types"

/** Download and embed bounded batches concurrently, preserving candidate order. */
export async function prepareEmbeddings(
  items: GpdMediaItem[],
  onProgress?: (progress: DetectionProgress) => void,
  signal?: AbortSignal,
  logger?: ScanLogger
): Promise<{
  embeddings: Float32Array[]
  validIndices: number[]
  cacheHits: number
  fetchThumbnailsMs: number
  computeEmbeddingsMs: number
}> {
  signal?.throwIfAborted()
  const db = await openEmbeddingStore()
  const controller = new AbortController()
  let pool: EmbeddingWorkerPool | undefined
  const abort = () => {
    controller.abort(signal?.reason)
    pool?.dispose(signal?.reason)
  }
  signal?.addEventListener("abort", abort, { once: true })
  try {
    signal?.throwIfAborted()
    const cacheStarted = performance.now()
    const cached = await readEmbeddings(
      db,
      items.map((item) => item.mediaKey)
    )
    await logger?.phaseComplete(
      "loadCachedEmbeddingsMs",
      Math.round(performance.now() - cacheStarted)
    )
    signal?.throwIfAborted()
    const cacheHits = items.filter((item) => cached.has(item.mediaKey)).length
    logger?.updateInfo({ candidates: items.length, cacheHits })
    const newEmbeddings = new Map<number, Float32Array>()
    if (cacheHits < items.length)
      pool = EmbeddingWorkerPool.acquire(items.length - cacheHits)
    let batch: EmbeddingInput[] = []
    const active = new Set<Promise<void>>()
    let downloadsFinished = false,
      completed = cacheHits,
      computeStarted: number | undefined
    const reportEmbedding = () => {
      if (downloadsFinished)
        onProgress?.({
          phase: "computing_embeddings",
          current: completed,
          total: items.length
        })
    }
    const submit = (inputs: EmbeddingInput[]): Promise<void> => {
      computeStarted ??= performance.now()
      const task = (async () => {
        controller.signal.throwIfAborted()
        const output = await pool!.embed(inputs)
        controller.signal.throwIfAborted()
        const records = output.map(({ localIdx, embedding }) => {
          const vector = new Float32Array(embedding)
          newEmbeddings.set(localIdx, vector)
          return { key: items[localIdx].mediaKey, embedding: vector }
        })
        await writeEmbeddings(db, records)
        controller.signal.throwIfAborted()
        completed += inputs.length
        reportEmbedding()
      })()
      active.add(task)
      void task.then(
        () => active.delete(task),
        () => active.delete(task)
      )
      return task
    }
    const startedAt = performance.now()
    await fetchThumbnails(
      items,
      new Set(cached.keys()),
      onProgress,
      controller.signal,
      logger,
      {
        retainBlobs: false,
        onBlob: async (index, blob) => {
          batch.push({ localIdx: index, blob })
          if (batch.length >= 20) {
            const ready = batch
            batch = []
            await submit(ready)
          }
        }
      }
    )
    const fetchThumbnailsMs = Math.round(performance.now() - startedAt)
    downloadsFinished = true
    reportEmbedding()
    if (batch.length) {
      const ready = batch
      batch = []
      await submit(ready)
    }
    await Promise.all(active)
    signal?.throwIfAborted()
    const computeEmbeddingsMs =
      computeStarted === undefined
        ? 0
        : Math.round(performance.now() - computeStarted)
    // These durations overlap: pipeline throughput is measured independently
    // from its component work, rather than pretending the phases are serial.
    await logger?.phaseComplete("fetchThumbnailsMs", fetchThumbnailsMs)
    await logger?.phaseComplete("computeEmbeddingsMs", computeEmbeddingsMs)
    const embeddings: Float32Array[] = [],
      validIndices: number[] = []
    items.forEach((item, index) => {
      const embedding = cached.get(item.mediaKey) ?? newEmbeddings.get(index)
      if (embedding) {
        embeddings.push(embedding)
        validIndices.push(index)
      }
    })
    pool?.release()
    return {
      embeddings,
      validIndices,
      cacheHits,
      fetchThumbnailsMs,
      computeEmbeddingsMs
    }
  } catch (error) {
    controller.abort(error)
    pool?.dispose(error)
    throw error
  } finally {
    signal?.removeEventListener("abort", abort)
    db?.close()
  }
}
