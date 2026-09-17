import { expect, it } from "vitest"

import { MediaStream } from "../../lib/media-stream"
import type {
  GpdMediaItem,
  GptkMediaCompleteMessage,
  GptkMediaPageMessage
} from "../../lib/types"

const page = (chunkIndex: number): GptkMediaPageMessage => ({
  app: "GPD",
  action: "gptkMediaPage",
  command: "getAllMediaItems",
  requestId: "r",
  chunkIndex,
  data: [{ mediaKey: String(chunkIndex) } as GpdMediaItem]
})
const complete = (totalChunks: number): GptkMediaCompleteMessage => ({
  app: "GPD",
  action: "gptkMediaComplete",
  command: "getAllMediaItems",
  requestId: "r",
  totalChunks,
  sentAt: Date.now(),
  metrics: {
    pages: totalChunks,
    itemsReceived: totalChunks,
    itemsEmitted: totalChunks,
    pageItems: [],
    pageRequestMs: [],
    requestMs: 0,
    requestAttempts: totalChunks,
    retries: 0,
    elapsedMs: 0,
    reachedCache: false
  }
})
it("waits for all distinct pages even when completion arrives first", () => {
  const stream = new MediaStream()
  expect(stream.accept(complete(3))).toBeNull()
  expect(stream.accept(page(2))).toBeNull()
  expect(stream.accept(page(2))).toBeNull()
  expect(stream.accept(page(0))).toBeNull()
  expect(stream.accept(page(1))?.items.map((item) => item.mediaKey)).toEqual([
    "0",
    "1",
    "2"
  ])
  expect(stream.accept(complete(3))).toBeNull()
})
it("does not finalize partial pages without a completion marker", () => {
  const stream = new MediaStream()
  expect(stream.accept(page(0))).toBeNull()
  expect(stream.accept(page(1))).toBeNull()
})
it("finishes an empty library and measures only final transfer delay", () => {
  const result = new MediaStream().accept({
    ...complete(0),
    sentAt: Date.now() - 25
  })
  expect(result?.items).toEqual([])
  expect(result?.finalTransferMs).toBeGreaterThanOrEqual(25)
})
it("rejects malformed indices instead of committing an incomplete snapshot", () => {
  const stream = new MediaStream()
  expect(() => stream.accept(page(-1))).toThrow("Invalid")
  stream.accept(page(3))
  expect(() => stream.accept(complete(1))).toThrow("Missing")
})
