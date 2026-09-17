import type { GpdMediaItem, StoredState } from "./types"

/** Metadata only. Independent of duplicate results and embedding/media caches. */
export interface MediaIndex {
  version: 1
  accountEmail: string
  mediaItems: Record<string, GpdMediaItem>
  newestCreationTimestamp: number
  refreshedAt: number
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

export async function saveMediaIndex(
  email: string | undefined,
  items: GpdMediaItem[]
): Promise<void> {
  const key = mediaIndexKey(email)
  if (!key) return // Never share an unidentified account's metadata.
  const index = createMediaIndex(email!, items)
  await enqueue(key, () => chrome.storage.local.set({ [key]: index }))
}

export async function loadMediaIndex(
  email?: string
): Promise<MediaIndex | null> {
  const key = mediaIndexKey(email)
  if (!key) return null
  await writes.get(key)
  const stored = await chrome.storage.local.get(key)
  const index = stored[key] as MediaIndex | undefined
  if (
    index?.version === 1 &&
    normalizeAccount(index.accountEmail) === normalizeAccount(email) &&
    index.mediaItems
  ) {
    return index
  }
  // Migrate lazily without removing/changing the existing scan results.
  const legacy = (
    (await chrome.storage.local.get("scanResults")) as Partial<StoredState>
  ).scanResults
  if (
    !legacy?.mediaItems ||
    !normalizeAccount(legacy.accountEmail) ||
    normalizeAccount(legacy.accountEmail) !== normalizeAccount(email)
  )
    return null
  const migrated = createMediaIndex(
    email!,
    Object.values(legacy.mediaItems),
    legacy.scanDate
  )
  await enqueue(key, () => chrome.storage.local.set({ [key]: migrated }))
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
    const stored = await chrome.storage.local.get(key)
    const index = stored[key] as MediaIndex | undefined
    if (
      !index ||
      normalizeAccount(index.accountEmail) !== normalizeAccount(email)
    )
      return
    const removed = new Set(removedKeys)
    const items = mergeMediaItems(restoredItems, index.mediaItems).filter(
      (item) => !removed.has(item.mediaKey)
    )
    await chrome.storage.local.set({
      [key]: createMediaIndex(email!, items, index.refreshedAt)
    })
  })
}
