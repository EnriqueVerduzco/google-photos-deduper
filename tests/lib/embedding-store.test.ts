import { IDBFactory } from "fake-indexeddb"
import { afterEach, beforeEach, expect, it, vi } from "vitest"

import {
  openEmbeddingStore,
  readEmbeddings,
  writeEmbeddings
} from "../../lib/embedding-store"

beforeEach(() => vi.stubGlobal("indexedDB", new IDBFactory()))
afterEach(() => vi.unstubAllGlobals())

it("retains existing database values, bulk reads and selective subset reads", async () => {
  const db = (await openEmbeddingStore())!
  const records = Array.from({ length: 10 }, (_, i) => ({
    key: String(i),
    embedding: new Float32Array([i, 1])
  }))
  await writeEmbeddings(db, records)
  expect([...(await readEmbeddings(db, ["1", "2"])).entries()]).toEqual([
    ["1", new Float32Array([1, 1])],
    ["2", new Float32Array([2, 1])]
  ])
  expect(
    (
      await readEmbeddings(
        db,
        records.map((r) => r.key)
      )
    ).size
  ).toBe(10)
  db.close()
})
it("does not classify invalid records as cache hits", async () => {
  const db = (await openEmbeddingStore())!
  await new Promise<void>((resolve) => {
    const tx = db.transaction("embeddings", "readwrite")
    tx.objectStore("embeddings").put("broken", "a")
    tx.oncomplete = () => resolve()
  })
  expect((await readEmbeddings(db, ["a"])).size).toBe(0)
  db.close()
})
