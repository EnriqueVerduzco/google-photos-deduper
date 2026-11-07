import type { GpdMediaItem, StoredState } from "./types"
import { loadLibrarySnapshot, rememberSnapshotAccount, saveLibrarySnapshot } from "./library-snapshots"

/** Metadata only. Independent of duplicate results and embedding/media caches. */
export interface MediaIndex {
  version: 1
  accountEmail: string
  mediaItems: Record<string, GpdMediaItem>
  newestCreationTimestamp: number
  refreshedAt: number
  snapshotId?: string
}

export function normalizeAccount(email?: string): string {
  return email?.trim().toLowerCase() ?? ""
}

export function mediaIndexKey(email?: string): string | null {
  const account = normalizeAccount(email)
  return account ? `mediaIndex:v1:${encodeURIComponent(account)}` : null
}

export function createMediaIndex(
  email: string,
  items: GpdMediaItem[],
  refreshedAt = Date.now()
): MediaIndex {
  // First occurrence wins, preserving source order across overlapping pages.
  const mediaItems: Record<string, GpdMediaItem> = Object.create(null)
  let newestCreationTimestamp = 0
  for (const item of items) {
    if (Object.hasOwn(mediaItems, item.mediaKey)) continue
    mediaItems[item.mediaKey] = item
    if (Number.isFinite(item.creationTimestamp)) {
      newestCreationTimestamp = Math.max(
        newestCreationTimestamp,
        item.creationTimestamp
      )
    }
  }
  return {
    version: 1,
    accountEmail: normalizeAccount(email),
    mediaItems,
    newestCreationTimestamp,
    refreshedAt
  }
}

export function mergeMediaItems(
  fetched: GpdMediaItem[],
  cached?: Record<string, GpdMediaItem> | null
): GpdMediaItem[] {
  const seen = new Set<string>()
  const items: GpdMediaItem[] = []
  for (const item of fetched) {
    if (seen.has(item.mediaKey)) continue
    seen.add(item.mediaKey)
    items.push(item)
  }
  if (cached)
    for (const item of Object.values(cached)) {
      if (!seen.has(item.mediaKey)) items.push(item)
    }
  return items
}

// Serialize writes/patches per account within the app, including trash/restore.
const writes = new Map<string, Promise<void>>()
function enqueue(key: string, operation: () => Promise<void>): Promise<void> {
  const next = (writes.get(key) ?? Promise.resolve()).then(operation)
  const settled = next.catch(() => {})
  writes.set(key, settled)
  void settled.then(() => {
    if (writes.get(key) === settled) writes.delete(key)
  })
  return next
}

async function persistIndex(key: string, index: MediaIndex): Promise<void> {
  const snapshotId = await saveLibrarySnapshot(index.accountEmail, Object.values(index.mediaItems))
  if (snapshotId) {
    const { mediaItems, ...summary } = index
    await chrome.storage.local.set({ [key]: { ...summary, version: 2, snapshotId } })
    await rememberSnapshotAccount(key)
  } else await chrome.storage.local.set({ [key]: index })
}

export async function saveMediaIndex(email: string | undefined, items: GpdMediaItem[]): Promise<void> {
  const key = mediaIndexKey(email)
  if (!key) return
  const index = createMediaIndex(email!, items)
  await enqueue(key, () => persistIndex(key, index))
}

async function readIndex(key: string, email?: string): Promise<MediaIndex | null> {
  const stored = await chrome.storage.local.get(key)
  const index = stored[key]
  if (normalizeAccount(index?.accountEmail) !== normalizeAccount(email)) return null
  if (index?.version === 1 && index.mediaItems) return index as MediaIndex
  if (index?.version === 2 && typeof index.snapshotId === "string") {
    const mediaItems = await loadLibrarySnapshot(index.snapshotId, email)
    if (mediaItems) return { ...index, version: 1, mediaItems }
  }
  return null
}

export async function loadMediaIndex(
  email?: string
): Promise<MediaIndex | null> {
  const key = mediaIndexKey(email)
  if (!key) return null
  await writes.get(key)
  const existing = await readIndex(key, email)
  if (existing) return existing
  // Migrate lazily without removing/changing the existing scan results.
  const legacy = (
    (await chrome.storage.local.get("scanResults")) as Partial<StoredState>
  ).scanResults
  if (
    !legacy ||
    !normalizeAccount(legacy.accountEmail) ||
    normalizeAccount(legacy.accountEmail) !== normalizeAccount(email)
  )
    return null
  const legacyItems = legacy.mediaItems ?? (legacy.mediaSnapshotId
    ? await loadLibrarySnapshot(legacy.mediaSnapshotId, email) : null)
  if (!legacyItems) return null
  const migrated = createMediaIndex(
    email!,
    Object.values(legacyItems),
    legacy.scanDate
  )
  await enqueue(key, () => persistIndex(key, migrated))
  return migrated
}

/** Apply only successful local media changes; full refresh reconciles external ones. */
export async function patchMediaIndex(
  email: string | undefined,
  removedKeys: string[],
  restoredItems: GpdMediaItem[] = []
): Promise<void> {
  const key = mediaIndexKey(email)
  if (!key) return
  await enqueue(key, async () => {
    const index = await readIndex(key, email)
    if (!index) return
    const removed = new Set(removedKeys)
    const items = mergeMediaItems(restoredItems, index.mediaItems).filter(
      (item) => !removed.has(item.mediaKey)
    )
    await persistIndex(key, createMediaIndex(email!, items, index.refreshedAt))
  })
}
