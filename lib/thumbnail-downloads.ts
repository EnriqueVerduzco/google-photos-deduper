import type { DetectionProgress } from "./duplicate-detector"
import { buildThumbUrl } from "./photo-url"
import type { ScanLogger, ThumbnailDownloadMetrics } from "./scan-log"
import type { GpdMediaItem } from "./types"

type ProgressCallback = (progress: DetectionProgress) => void

const DEFAULT_THUMBNAIL_CONCURRENCY = 16
const THUMBNAIL_FETCH_TIMEOUT_MS = 8000
const THUMBNAIL_MAX_RETRIES = 2
const THUMBNAIL_RETRY_BASE_MS = 100
const THUMB_HEIGHT = 200

function waitForRetry(ms: number, signal?: AbortSignal): Promise<void> {
  signal?.throwIfAborted()
  return new Promise((resolve, reject) => {
    const timeoutId = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort)
      resolve()
    }, ms)
    const onAbort = () => {
      clearTimeout(timeoutId)
      reject(signal?.reason ?? new DOMException("Aborted", "AbortError"))
    }
    signal?.addEventListener("abort", onAbort, { once: true })
  })
}

export interface FetchThumbnailsResult {
  blobs: (Blob | null)[]
  metrics: ThumbnailDownloadMetrics
}

/**
 * Download uncached thumbnails with bounded concurrency and retries.
 * Exported so reliability/cancellation behavior can be unit tested without
 * invoking MediaPipe or accessing Google Photos.
 */
export async function fetchThumbnails(
  items: GpdMediaItem[],
  cachedKeySet: Set<string>,
  onProgress?: ProgressCallback,
  signal?: AbortSignal,
  logger?: ScanLogger,
  options: {
    retainBlobs?: boolean
    onBlob?: (index: number, blob: Blob) => Promise<void>
  } = {}
): Promise<FetchThumbnailsResult> {
  const concurrency = DEFAULT_THUMBNAIL_CONCURRENCY
  const blobs: (Blob | null)[] = new Array(items.length).fill(null)
  let completed = 0
  const startedAt = performance.now()
  const metrics: ThumbnailDownloadMetrics = {
    concurrency,
    attempts: 0,
    successes: 0,
    failures: 0,
    httpStatusFailures: 0,
    throttledResponses: 0,
    timeouts: 0,
    retries: 0,
    elapsedMs: 0,
    effectivePerSecond: 0
  }

  // Only enqueue items that don't have a cached embedding
  const queue = items
    .map((item, i) => ({ item, index: i }))
    .filter(({ item }) => !cachedKeySet.has(item.mediaKey))

  // Report progress only against items that actually need downloading.
  // Counting cached items as "pre-completed" caused the bar to start at e.g. 80%
  // and never visibly move before the phase transitioned.
  const total = queue.length

  const reportProgress = () => {
    if (total === 0) return
    if (completed % 50 === 0 || completed === total) {
      onProgress?.({
        phase: "downloading_thumbnails",
        current: completed,
        total
      })
    }
  }

  const download = async (entry: { item: GpdMediaItem; index: number }) => {
    const url = buildThumbUrl(entry.item.thumb, { height: THUMB_HEIGHT })
    let lastFailure = "unknown error"

    for (let attempt = 0; attempt <= THUMBNAIL_MAX_RETRIES; attempt++) {
      signal?.throwIfAborted()
      metrics.attempts++

      const attemptController = new AbortController()
      let timedOut = false
      const onCancel = () => attemptController.abort(signal?.reason)
      signal?.addEventListener("abort", onCancel, { once: true })
      const timeoutId = setTimeout(() => {
        timedOut = true
        attemptController.abort(
          new DOMException("Thumbnail download timed out", "TimeoutError")
        )
      }, THUMBNAIL_FETCH_TIMEOUT_MS)

      let shouldRetry = false
      try {
        const response = await fetch(url, {
          credentials: "include",
          signal: attemptController.signal
        })
        if (response.ok) {
          const blob = await response.blob()
          metrics.successes++
          return blob
        }

        metrics.httpStatusFailures++
        if (response.status === 429) metrics.throttledResponses++
        lastFailure = `HTTP ${response.status}`
        void response.body?.cancel().catch(() => {})
        shouldRetry = response.status === 429 || response.status >= 500
      } catch (error) {
        // A user cancellation takes precedence over timeout/network handling.
        signal?.throwIfAborted()
        if (timedOut) {
          metrics.timeouts++
          lastFailure = "timeout"
        } else {
          lastFailure = error instanceof Error ? error.message : String(error)
        }
        // Fetch rejections other than user cancellation are transient network
        // failures and are eligible for a bounded retry.
        shouldRetry = true
      } finally {
        clearTimeout(timeoutId)
        signal?.removeEventListener("abort", onCancel)
      }

      if (!shouldRetry || attempt === THUMBNAIL_MAX_RETRIES) break
      metrics.retries++
      await waitForRetry(THUMBNAIL_RETRY_BASE_MS * 2 ** attempt, signal)
    }

    metrics.failures++
    console.warn(
      `[GPD] thumbnail failed: mediaKey=${entry.item.mediaKey}, reason=${lastFailure}`
    )
    return null
  }

  let queuePosition = 0
  const worker = async () => {
    while (queuePosition < queue.length) {
      signal?.throwIfAborted()
      const entry = queue[queuePosition++]
      if (!entry) break

      const blob = await download(entry)
      if (blob) {
        if (options.retainBlobs !== false) blobs[entry.index] = blob
        await options.onBlob?.(entry.index, blob)
      }

      completed++
      reportProgress()
    }
  }

  try {
    const workers = Array.from(
      { length: Math.min(concurrency, queue.length) },
      () => worker()
    )
    await Promise.all(workers)
    return { blobs, metrics }
  } finally {
    metrics.elapsedMs = Math.round(performance.now() - startedAt)
    metrics.effectivePerSecond =
      metrics.elapsedMs > 0
        ? Math.round((metrics.successes / (metrics.elapsedMs / 1000)) * 100) /
          100
        : 0
    console.log(
      `[GPD] thumbnails: concurrency=${metrics.concurrency}, attempts=${metrics.attempts}, ` +
        `successful=${metrics.successes}, failed=${metrics.failures}, ` +
        `httpFailures=${metrics.httpStatusFailures}, throttled=${metrics.throttledResponses}, ` +
        `timeouts=${metrics.timeouts}, ` +
        `retries=${metrics.retries}, elapsed=${metrics.elapsedMs}ms, ` +
        `rate=${metrics.effectivePerSecond}/sec`
    )
    await logger?.recordThumbnailDownloads(metrics)
  }
}
