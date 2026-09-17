import { beforeEach, expect, it, vi } from "vitest"

import {
  createMediaIndex,
  loadMediaIndex,
  mediaIndexKey,
  mergeMediaItems,
  patchMediaIndex,
  saveMediaIndex
} from "../../lib/media-index"
import type { GpdMediaItem } from "../../lib/types"

let store: Record<string, any>
const storage = {
  get: vi.fn(async (key: string) => ({ [key]: store[key] })),
  set: vi.fn(async (values: Record<string, unknown>) => {
    Object.assign(store, values)
  })
}
const item = (key: string, timestamp = 100): GpdMediaItem => ({
  mediaKey: key,
  dedupKey: key,
  thumb: "local",
  timestamp,
  creationTimestamp: timestamp
})
beforeEach(() => {
  store = {}
  vi.clearAllMocks()
  vi.stubGlobal("chrome", { storage: { local: storage } })
})

it("keeps independent per-account indices after scan results are cleared", async () => {
  await saveMediaIndex(" A@Example.com ", [item("a")])
  await saveMediaIndex("b@example.com", [item("b")])
  delete store.scanResults
  expect(
    Object.keys((await loadMediaIndex("a@example.com"))!.mediaItems)
  ).toEqual(["a"])
  expect(
    Object.keys((await loadMediaIndex("b@example.com"))!.mediaItems)
  ).toEqual(["b"])
})

it("migrates matching legacy results without deleting or rewriting them", async () => {
  const legacy = {
    accountEmail: "a@example.com",
    mediaItems: { a: item("a", 200) },
    groups: [],
    scanDate: 123
  }
  store.scanResults = legacy
  expect(await loadMediaIndex("a@example.com")).toMatchObject({
    newestCreationTimestamp: 200,
    refreshedAt: 123
  })
  expect(store.scanResults).toBe(legacy)
  expect(store[mediaIndexKey("a@example.com")!]).toBeDefined()
  storage.get.mockClear()
  await loadMediaIndex("a@example.com")
  expect(storage.get.mock.calls.map(([key]) => key)).toEqual([
    mediaIndexKey("a@example.com")
  ])
})

it.each([undefined, "", "other@example.com"])(
  "never reuses another account's metadata for %s",
  async (email) => {
    store.scanResults = {
      accountEmail: "a@example.com",
      mediaItems: { a: item("a") }
    }
    expect(await loadMediaIndex(email)).toBeNull()
  }
)

it("never migrates or writes unidentified account metadata", async () => {
  store.scanResults = { mediaItems: { a: item("a") } }
  expect(await loadMediaIndex("a@example.com")).toBeNull()
  await saveMediaIndex(undefined, [item("x")])
  expect(storage.set).not.toHaveBeenCalled()
})

it("merges repeated pages by key, keeps source order, and refreshes fetched fields", () => {
  const fetched = [
    item("new", 200),
    { ...item("old"), thumb: "updated" },
    item("new", 200)
  ]
  expect(
    mergeMediaItems(fetched, { old: item("old"), oldest: item("oldest", 1) })
  ).toEqual([fetched[0], fetched[1], item("oldest", 1)])
})

it("full snapshot replacement removes externally deleted items and accepts an empty library", async () => {
  await saveMediaIndex("a@example.com", [item("old"), item("keep")])
  await saveMediaIndex("a@example.com", [item("keep")])
  expect(
    Object.keys((await loadMediaIndex("a@example.com"))!.mediaItems)
  ).toEqual(["keep"])
  await saveMediaIndex("a@example.com", [])
  expect((await loadMediaIndex("a@example.com"))!.mediaItems).toEqual({})
})

it("serializes successful trash/restore patches after a pending snapshot write", async () => {
  await Promise.all([
    saveMediaIndex("a@example.com", [item("keep"), item("trash", 200)]),
    patchMediaIndex("a@example.com", ["trash"])
  ])
  expect(
    Object.keys((await loadMediaIndex("a@example.com"))!.mediaItems)
  ).toEqual(["keep"])
  await patchMediaIndex("a@example.com", [], [item("trash", 200)])
  expect((await loadMediaIndex("a@example.com"))!.newestCreationTimestamp).toBe(
    200
  )
})

it("retains the prior index if saving fails", async () => {
  await saveMediaIndex("a@example.com", [item("old")])
  storage.set.mockRejectedValueOnce(new Error("disk full"))
  await expect(saveMediaIndex("a@example.com", [item("new")])).rejects.toThrow(
    "disk full"
  )
  expect(
    Object.keys((await loadMediaIndex("a@example.com"))!.mediaItems)
  ).toEqual(["old"])
})

it("computes the watermark from valid timestamps and handles duplicate keys", () => {
  expect(
    createMediaIndex("a@example.com", [
      item("a", 100),
      item("a", 200),
      item("b", NaN)
    ])
  ).toMatchObject({ newestCreationTimestamp: 100 })
})
