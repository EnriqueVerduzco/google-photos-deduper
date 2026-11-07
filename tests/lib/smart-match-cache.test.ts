import { webcrypto } from "node:crypto"
import { IDBFactory } from "fake-indexeddb"
import { afterEach, beforeEach, expect, it, vi } from "vitest"

import { SmartMatchCache } from "../../lib/smart-match-cache"

beforeEach(() => {
  vi.stubGlobal("indexedDB", new IDBFactory())
  vi.stubGlobal("crypto", webcrypto)
})
afterEach(() => vi.unstubAllGlobals())

it("invalidates on exact vector, ordered membership and threshold changes", async () => {
  const cache = (await SmartMatchCache.open())!
  const rows = [new Float32Array([1, 0]), new Float32Array([1, 0])]
  const key = await cache.fingerprint(["a", "b"], rows, 0.99)
  await cache.put(key, 2, [[0, 1]])
  expect(await cache.get(key, 2)).toEqual([[0, 1]])
  expect(await cache.fingerprint(["b", "a"], rows, 0.99)).not.toBe(key)
  expect(await cache.fingerprint(["a", "b"], rows, 1)).not.toBe(key)
  expect(
    await cache.fingerprint(
      ["a", "b"],
      [rows[0], new Float32Array([1, 1e-10])],
      0.99
    )
  ).not.toBe(key)
  cache.close()
})
it("rejects invalid cached memberships and preserves cached empty results", async () => {
  const cache = (await SmartMatchCache.open())!
  await cache.put("corrupt", 2, [[0, 2]])
  expect(await cache.get("corrupt", 2)).toBeNull()
  await cache.put("empty", 2, [])
  expect(await cache.get("empty", 2)).toEqual([])
  expect(await cache.get("empty", 3)).toBeNull()
  cache.close()
})
it("falls back when persistent caching is unavailable", async () => {
  vi.stubGlobal("indexedDB", undefined)
  expect(await SmartMatchCache.open()).toBeNull()
})
