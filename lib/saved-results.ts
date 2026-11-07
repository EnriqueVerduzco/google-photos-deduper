import { loadLibrarySnapshot, saveLibrarySnapshot } from "./library-snapshots"
import type { GpdMediaItem, StoredState } from "./types"

type SavedResults = NonNullable<StoredState["scanResults"]>
let writes = Promise.resolve()
let revision = 0

export async function hydrateSavedResults(
  saved: SavedResults
): Promise<
  (SavedResults & { mediaItems: Record<string, GpdMediaItem> }) | null
> {
  // Legacy inline records remain readable, including unidentified accounts.
  const mediaItems =
    saved.mediaItems ??
    (saved.mediaSnapshotId
      ? await loadLibrarySnapshot(saved.mediaSnapshotId, saved.accountEmail)
      : null)
  return mediaItems ? { ...saved, mediaItems } : null
}

export function persistSavedResults(
  saved: SavedResults & { mediaItems: Record<string, GpdMediaItem> }
): Promise<void> {
  const current = ++revision
  const task = writes
    .catch(() => {})
    .then(async () => {
      if (current !== revision) return
      const mediaSnapshotId = await saveLibrarySnapshot(
        saved.accountEmail,
        Object.values(saved.mediaItems)
      )
      if (current !== revision) return
      const { mediaItems, ...summary } = saved
      await chrome.storage.local.set({
        scanResults: mediaSnapshotId
          ? { ...summary, mediaSnapshotId }
          : { ...summary, mediaItems }
      })
    })
  writes = task.catch(() => {})
  return task
}

export function clearSavedResults(): Promise<void> {
  ++revision
  const task = writes
    .catch(() => {})
    .then(() => chrome.storage.local.remove("scanResults"))
  writes = task.catch(() => {})
  return task
}
