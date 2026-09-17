import { afterEach, describe, expect, it, vi } from "vitest"

import { fetchThumbnails } from "../../lib/duplicate-detector"
import type { ScanLogger } from "../../lib/scan-log"
import type { GpdMediaItem } from "../../lib/types"

function makeItems(count: number): GpdMediaItem[] {
  return Array.from({ length: count }, (_, i) => ({
    mediaKey: `item-${i}`,
    dedupKey: `item-${i}`,
    thumb: `https://example.com/item-${i}`,
    timestamp: i,
    creationTimestamp: i
  }))
}

function okResponse(): Response {
  return {
    ok: true,
    status: 200,
    blob: async () => new Blob(["thumbnail"]),
    body: null
  } as Response
}

function errorResponse(status: number): Response {
  return {
    ok: false,
    status,
    body: null
  } as Response
}

afterEach(() => {
  vi.restoreAllMocks()
  vi.useRealTimers()
})

describe("fetchThumbnails", () => {
  it("uses concurrency 10 by default", async () => {
    let inFlight = 0
    let maxInFlight = 0
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        inFlight++
        maxInFlight = Math.max(maxInFlight, inFlight)
        await new Promise((resolve) => setTimeout(resolve, 5))
        inFlight--
        return okResponse()
      })
    )

    const result = await fetchThumbnails(makeItems(11), new Set())

    expect(result.metrics.concurrency).toBe(10)
    expect(maxInFlight).toBe(10)
    expect(result.metrics.successes).toBe(11)
  })

  it("honors a selected concurrency", async () => {
    let inFlight = 0
    let maxInFlight = 0
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        inFlight++
        maxInFlight = Math.max(maxInFlight, inFlight)
        await new Promise((resolve) => setTimeout(resolve, 5))
        inFlight--
        return okResponse()
      })
    )

    const result = await fetchThumbnails(
      makeItems(13),
      new Set(),
      undefined,
      undefined,
      12
    )

    expect(result.metrics.concurrency).toBe(12)
    expect(maxInFlight).toBe(12)
  })

  it("retries transient failures and records exhausted and non-retryable failures", async () => {
    const callsByItem = new Map<string, number>()
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request) => {
        const url = String(input)
        const key = url.includes("item-0")
          ? "item-0"
          : url.includes("item-1")
            ? "item-1"
            : "item-2"
        const call = (callsByItem.get(key) ?? 0) + 1
        callsByItem.set(key, call)

        if (key === "item-0") {
          if (call === 1) return errorResponse(429)
          if (call === 2) return errorResponse(503)
          return okResponse()
        }
        if (key === "item-1") return errorResponse(404)
        throw new TypeError("network unavailable")
      })
    )

    const { blobs, metrics } = await fetchThumbnails(makeItems(3), new Set())

    expect(blobs[0]).toBeInstanceOf(Blob)
    expect(blobs[1]).toBeNull()
    expect(blobs[2]).toBeNull()
    expect(metrics).toMatchObject({
      attempts: 7,
      successes: 1,
      failures: 2,
      httpStatusFailures: 3,
      throttledResponses: 1,
      timeouts: 0,
      retries: 4
    })
  })

  it("counts timed-out attempts and stops after the retry bound", async () => {
    vi.useFakeTimers()
    vi.stubGlobal(
      "fetch",
      vi.fn(
        (_input: string | URL | Request, init?: RequestInit) =>
          new Promise<Response>((_resolve, reject) => {
            init?.signal?.addEventListener(
              "abort",
              () => reject(init.signal?.reason),
              { once: true }
            )
          })
      )
    )

    const pending = fetchThumbnails(makeItems(1), new Set())
    await vi.advanceTimersByTimeAsync(8000)
    await vi.advanceTimersByTimeAsync(100)
    await vi.advanceTimersByTimeAsync(8000)
    await vi.advanceTimersByTimeAsync(200)
    await vi.advanceTimersByTimeAsync(8000)

    const { metrics } = await pending
    expect(metrics).toMatchObject({
      attempts: 3,
      successes: 0,
      failures: 1,
      timeouts: 3,
      retries: 2,
    })
  })

  it("aborts in-flight requests promptly without retrying", async () => {
    const controller = new AbortController()
    const recordThumbnailDownloads = vi.fn(async () => {})
    const logger = { recordThumbnailDownloads } as unknown as ScanLogger
    vi.stubGlobal(
      "fetch",
      vi.fn(
        (_input: string | URL | Request, init?: RequestInit) =>
          new Promise<Response>((_resolve, reject) => {
            init?.signal?.addEventListener(
              "abort",
              () => reject(init.signal?.reason),
              { once: true }
            )
          })
      )
    )

    const pending = fetchThumbnails(
      makeItems(20),
      new Set(),
      undefined,
      controller.signal,
      16,
      logger
    )
    controller.abort()

    await expect(pending).rejects.toMatchObject({ name: "AbortError" })
    expect(recordThumbnailDownloads).toHaveBeenCalledWith(
      expect.objectContaining({ concurrency: 16, retries: 0 })
    )
  })
})
