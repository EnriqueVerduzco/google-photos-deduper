const DB_NAME = "gpd-smart-matches"
const STORE = "buckets"
const ALGORITHM_VERSION = 1

export class SmartMatchCache {
  private constructor(private db: IDBDatabase) {}

  static async open(): Promise<SmartMatchCache | null> {
    if (typeof indexedDB === "undefined" || !globalThis.crypto?.subtle)
      return null
    return new Promise((resolve) => {
      const request = indexedDB.open(DB_NAME, 1)
      request.onupgradeneeded = () =>
        request.result
          .createObjectStore(STORE)
          .createIndex("createdAt", "createdAt")
      request.onsuccess = () => resolve(new SmartMatchCache(request.result))
      request.onerror = () => resolve(null)
    })
  }

  async fingerprint(
    keys: string[],
    rows: Float32Array[],
    threshold: number
  ): Promise<string> {
    // Ordered membership, exact vector bytes and comparison settings all affect
    // the result. Favorite/quality choices are deliberately recalculated later.
    const header = new TextEncoder().encode(
      JSON.stringify([
        ALGORITHM_VERSION,
        String(threshold),
        keys,
        rows.map((row) => row.length)
      ])
    )
    const bytes = new Uint8Array(
      header.length + rows.reduce((n, row) => n + row.byteLength, 0)
    )
    bytes.set(header)
    let offset = header.length
    for (const row of rows) {
      bytes.set(
        new Uint8Array(row.buffer, row.byteOffset, row.byteLength),
        offset
      )
      offset += row.byteLength
    }
    const digest = await crypto.subtle.digest("SHA-256", bytes)
    return Array.from(new Uint8Array(digest), (b) =>
      b.toString(16).padStart(2, "0")
    ).join("")
  }

  async get(key: string, count: number): Promise<number[][] | null> {
    return new Promise((resolve) => {
      try {
        const request = this.db.transaction(STORE).objectStore(STORE).get(key)
        request.onsuccess = () => {
          const record = request.result
          const groups = record?.groups
          const seen = new Set<number>()
          if (
            record?.count !== count ||
            !Array.isArray(groups) ||
            !groups.every(
              (group: unknown) =>
                Array.isArray(group) &&
                group.length >= 2 &&
                group.every((index: unknown) => {
                  if (
                    !Number.isInteger(index) ||
                    (index as number) < 0 ||
                    (index as number) >= count ||
                    seen.has(index as number)
                  )
                    return false
                  seen.add(index as number)
                  return true
                })
            )
          )
            resolve(null)
          else resolve(groups)
        }
        request.onerror = () => resolve(null)
      } catch {
        resolve(null)
      }
    })
  }

  async put(key: string, count: number, groups: number[][]): Promise<void> {
    return new Promise((resolve) => {
      try {
        const tx = this.db.transaction(STORE, "readwrite")
        const store = tx.objectStore(STORE)
        store.put({ count, groups, createdAt: Date.now() }, key)
        const countRequest = store.count()
        countRequest.onsuccess = () => {
          let remaining = countRequest.result - 2000
          if (remaining <= 0) return
          const cursorRequest = store.index("createdAt").openKeyCursor()
          cursorRequest.onsuccess = () => {
            const cursor = cursorRequest.result
            if (cursor && remaining-- > 0) {
              store.delete(cursor.primaryKey)
              cursor.continue()
            }
          }
        }
        tx.oncomplete = tx.onabort = tx.onerror = () => resolve()
      } catch {
        resolve()
      }
    })
  }

  close(): void {
    this.db.close()
  }
}
