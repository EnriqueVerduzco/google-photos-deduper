import type { GpdMediaItem } from "./types"

interface Snapshot {
  id: string
  account: string
  createdAt: number
  mediaItems: Record<string, GpdMediaItem>
}
interface Stamp {
  id: string
  account: string
  count: number
}
const hot = new WeakMap<GpdMediaItem, Stamp>()
let database: Promise<IDBDatabase | null> | undefined
let lastPrune = 0

export function openSnapshotStore(): Promise<IDBDatabase | null> {
  if (typeof indexedDB === "undefined") return Promise.resolve(null)
  return (database ??= new Promise((resolve) => {
    const request = indexedDB.open("gpd-library-snapshots", 1)
    request.onupgradeneeded = () => {
      request.result
        .createObjectStore("snapshots", { keyPath: "id" })
        .createIndex("createdAt", "createdAt")
      request.result.createObjectStore("accounts")
    }
    request.onsuccess = () => {
      request.result.onversionchange = () => {
        request.result.close()
        database = undefined
      }
      resolve(request.result)
    }
    request.onerror = () => {
      database = undefined
      resolve(null)
    }
  }))
}

export async function saveLibrarySnapshot(
  account: string | undefined,
  items: GpdMediaItem[]
): Promise<string | null> {
  const normalized = account?.trim().toLowerCase()
  if (!normalized) return null
  const db = await openSnapshotStore()
  if (!db) return null
  try {
    // App metadata is immutable. The detection/result map reuses these exact
    // item objects, so its initial persistence can reference the fetched snapshot.
    const stamp = items.length ? hot.get(items[0]) : undefined
    if (
      stamp?.account === normalized &&
      stamp.count === items.length &&
      items.every((item) => hot.get(item) === stamp)
    ) {
      const exists = await new Promise<boolean>((resolve) => {
        const request = db
          .transaction("snapshots")
          .objectStore("snapshots")
          .getKey(stamp.id)
        request.onsuccess = () => resolve(request.result !== undefined)
        request.onerror = () => resolve(false)
      })
      if (exists) return stamp.id
    }
    const id = crypto.randomUUID()
    const mediaItems: Record<string, GpdMediaItem> = Object.create(null)
    for (const item of items)
      if (!Object.hasOwn(mediaItems, item.mediaKey))
        mediaItems[item.mediaKey] = item
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction("snapshots", "readwrite")
      tx.objectStore("snapshots").put({
        id,
        account: normalized,
        createdAt: Date.now(),
        mediaItems
      } satisfies Snapshot)
      tx.oncomplete = () => resolve()
      tx.onerror = tx.onabort = () => reject(tx.error)
    })
    const nextStamp = {
      id,
      account: normalized,
      count: Object.keys(mediaItems).length
    }
    for (const item of Object.values(mediaItems)) hot.set(item, nextStamp)
    return id
  } catch {
    return null
  }
}

export async function loadLibrarySnapshot(
  id: string,
  account: string | undefined
): Promise<Record<string, GpdMediaItem> | null> {
  const db = await openSnapshotStore()
  if (!db || !account) return null
  return new Promise((resolve) => {
    const request = db.transaction("snapshots").objectStore("snapshots").get(id)
    request.onsuccess = () => {
      const snapshot = request.result as Snapshot | undefined
      if (snapshot?.account !== account.trim().toLowerCase()) {
        resolve(null)
        return
      }
      const items = Object.values(snapshot.mediaItems)
      const stamp = { id, account: snapshot.account, count: items.length }
      for (const item of items) hot.set(item, stamp)
      resolve(snapshot.mediaItems)
    }
    request.onerror = () => resolve(null)
  })
}

export async function rememberSnapshotAccount(
  storageKey: string
): Promise<void> {
  const db = await openSnapshotStore()
  if (!db) return
  await new Promise<void>((resolve) => {
    const tx = db.transaction("accounts", "readwrite")
    tx.objectStore("accounts").put(true, storageKey)
    tx.oncomplete = tx.onabort = tx.onerror = () => resolve()
  })
  void pruneLibrarySnapshots().catch(() => {})
}

/** Unreferenced snapshots have a grace period for other tabs' pending writes. */
export async function pruneLibrarySnapshots(): Promise<void> {
  if (Date.now() - lastPrune < 60 * 60 * 1000) return
  lastPrune = Date.now()
  const db = await openSnapshotStore()
  if (!db) return
  const expired = await new Promise<string[]>((resolve) => {
    const keys: string[] = []
    const request = db
      .transaction("snapshots")
      .objectStore("snapshots")
      .index("createdAt")
      .openKeyCursor(IDBKeyRange.upperBound(Date.now() - 60 * 60 * 1000))
    request.onsuccess = () => {
      const cursor = request.result
      if (cursor) {
        keys.push(cursor.primaryKey as string)
        cursor.continue()
      } else resolve(keys)
    }
    request.onerror = () => resolve([])
  })
  if (!expired.length) return
  const keys = await new Promise<string[]>((resolve) => {
    const request = db
      .transaction("accounts")
      .objectStore("accounts")
      .getAllKeys()
    request.onsuccess = () => resolve(request.result as string[])
    request.onerror = () => resolve([])
  })
  const pointers = await chrome.storage.local.get([...keys, "scanResults"])
  const referenced = new Set(
    Object.values(pointers).map(
      (value: any) => value?.mediaSnapshotId ?? value?.snapshotId
    )
  )
  await new Promise<void>((resolve) => {
    const tx = db.transaction("snapshots", "readwrite")
    for (const id of expired)
      if (!referenced.has(id)) tx.objectStore("snapshots").delete(id)
    tx.oncomplete = tx.onabort = tx.onerror = () => resolve()
  })
}
