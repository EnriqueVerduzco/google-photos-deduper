import { IDBFactory, IDBKeyRange } from "fake-indexeddb"
import { beforeAll, expect, it, vi } from "vitest"

import {
  loadLibrarySnapshot,
  saveLibrarySnapshot
} from "../../lib/library-snapshots"
import {
  loadMediaIndex,
  mediaIndexKey,
  saveMediaIndex
} from "../../lib/media-index"
import {
  clearSavedResults,
  hydrateSavedResults,
  persistSavedResults
} from "../../lib/saved-results"
import type { GpdMediaItem } from "../../lib/types"

const store: Record<string, any> = {}
const item = (key: string): GpdMediaItem => ({
  mediaKey: key,
  dedupKey: key,
  thumb: "local",
  timestamp: 1,
  creationTimestamp: 1
})
beforeAll(() => {
  vi.stubGlobal("indexedDB", new IDBFactory())
  vi.stubGlobal("IDBKeyRange", IDBKeyRange)
  vi.stubGlobal("chrome", {
    storage: {
      local: {
        get: async (keys: string | string[]) =>
          Object.fromEntries(
            (Array.isArray(keys) ? keys : [keys]).map((key) => [
              key,
              store[key]
            ])
          ),
        set: async (values: any) => {
          Object.assign(store, values)
        },
        remove: async (key: string) => {
          delete store[key]
        }
      }
    }
  })
})

it("stores metadata once and hydrates summary-only results from the same snapshot", async () => {
  const account = "snapshots@example.com",
    rows = [item("a"), item("b")]
  await saveMediaIndex(account, rows)
  const pointer = store[mediaIndexKey(account)!]
  expect(pointer.version).toBe(2)
  expect(pointer.mediaItems).toBeUndefined()
  await persistSavedResults({
    accountEmail: account,
    mediaItems: { a: rows[0], b: rows[1] },
    groups: [],
    scanDate: 1,
    totalItems: 2
  })
  expect(store.scanResults.mediaSnapshotId).toBe(pointer.snapshotId)
  expect(store.scanResults.mediaItems).toBeUndefined()
  expect((await hydrateSavedResults(store.scanResults))!.mediaItems).toEqual({
    a: rows[0],
    b: rows[1]
  })
  expect(Object.keys((await loadMediaIndex(account))!.mediaItems)).toEqual([
    "a",
    "b"
  ])
})
it("keeps saved results intact when a newer scan updates the current account snapshot", async () => {
  const old = { ...store.scanResults }
  await saveMediaIndex(old.accountEmail, [item("new-scan")])
  expect(Object.keys((await hydrateSavedResults(old))!.mediaItems)).toEqual([
    "a",
    "b"
  ])
  expect(
    await loadLibrarySnapshot(old.mediaSnapshotId, "other@example.com")
  ).toBeNull()
})
it("preserves legacy inline results and prevents pending saves resurrecting cleared results", async () => {
  const legacy = {
    accountEmail: "legacy@example.com",
    mediaItems: { a: item("a") },
    groups: [],
    scanDate: 1,
    totalItems: 1
  }
  expect((await hydrateSavedResults(legacy))!.mediaItems).toEqual(
    legacy.mediaItems
  )
  const saving = persistSavedResults(legacy)
  await clearSavedResults()
  await saving
  expect(store.scanResults).toBeUndefined()
})
it("creates a separate immutable snapshot when group membership changes", async () => {
  const rows = [item("x"), item("y")]
  const full = await saveLibrarySnapshot("membership@example.com", rows)
  const subset = await saveLibrarySnapshot("membership@example.com", [rows[0]])
  expect(subset).not.toBe(full)
  expect(
    Object.keys((await loadLibrarySnapshot(full!, "membership@example.com"))!)
  ).toEqual(["x", "y"])
})
