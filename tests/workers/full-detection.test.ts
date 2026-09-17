import { afterEach, expect, it, vi } from "vitest"

import * as fullMatcher from "../../lib/full-matcher"
import { fullDetectionWorkTotal } from "../../lib/full-matcher"
import { directedFullReference } from "../helpers/directed-full-reference"
import { axis } from "../helpers/full-match-fixtures"

vi.mock("@mediapipe/tasks-vision", () => ({ ImageEmbedder: {} }))
afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

it("dispatches Full detection through the worker with zero-copy row views and unchanged results", async () => {
  const matcher = vi.spyOn(fullMatcher, "fullCommunityDetection")
  let listener: (event: { data: unknown }) => Promise<void>
  const messages: any[] = []
  vi.stubGlobal("self", {
    addEventListener: (_: string, callback: typeof listener) => {
      listener = callback
    },
    postMessage: (message: unknown) => messages.push(message)
  })
  await import("../../workers/embedder.worker")
  const rows = [axis(0), axis(1), axis(0), axis(1), axis(0)]
  const flat = new Float32Array(rows.length * 64)
  rows.forEach((row, i) => flat.set(row, i * 64))
  const before = flat.slice()
  await listener!({
    data: {
      type: "detect",
      data: {
        flatEmbeddings: flat,
        n: rows.length,
        dim: 64,
        threshold: 0.99,
        timestamps: [5, 4, 3, 2, 1]
      }
    }
  })
  expect(messages.at(-1)).toEqual({
    type: "detectionResults",
    groups: await directedFullReference(rows, 0.99)
  })
  const progress = messages.filter(
    (message) => message.type === "detectionProgress"
  )
  expect(progress[0].current).toBe(0)
  expect(progress.at(-1).current).toBe(fullDetectionWorkTotal(rows.length))
  expect(
    progress.every(
      (message) => message.total === fullDetectionWorkTotal(rows.length)
    )
  ).toBe(true)
  const views = matcher.mock.calls[0][0]
  expect(views.every((row) => row.buffer === flat.buffer)).toBe(true)
  expect(flat).toEqual(before)
})
