const DB_NAME = "gpd-embeddings"
const STORE = "embeddings"

export async function openEmbeddingStore(): Promise<IDBDatabase | null> {
  if (typeof indexedDB === "undefined") return null
  return new Promise((resolve) => {
    const request = indexedDB.open(DB_NAME, 1)
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(STORE))
        request.result.createObjectStore(STORE)
    }
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => resolve(null)
  })
}

function decode(value: unknown): Float32Array | null {
  return value instanceof ArrayBuffer &&
    value.byteLength > 0 &&
    value.byteLength % 4 === 0
    ? new Float32Array(value)
    : null
}

export async function readEmbeddings(
  db: IDBDatabase | null,
  keys: string[]
): Promise<Map<string, Float32Array>> {
  const result = new Map<string, Float32Array>()
  if (!db || !keys.length) return result
  try {
    const allKeys = await new Promise<string[]>((resolve) => {
      const request = db.transaction(STORE).objectStore(STORE).getAllKeys()
      request.onsuccess = () => resolve(request.result as string[])
      request.onerror = () => resolve([])
    })
    const wanted = new Set(keys)
    const hits = allKeys.filter((key) => wanted.has(key))
    if (!hits.length) return result
    if (hits.length >= allKeys.length / 2) {
      // Large scans retain efficient bulk reads. Never depend on two separate
      // transactions having the same ordering while another scan is writing.
      await new Promise<void>((resolve) => {
        const tx = db.transaction(STORE)
        const store = tx.objectStore(STORE)
        const keyRequest = store.getAllKeys(),
          values = store.getAll()
        tx.oncomplete = () => {
          keyRequest.result.forEach((key, i) => {
            if (!wanted.has(key as string)) return
            const embedding = decode(values.result[i])
            if (embedding) result.set(key as string, embedding)
          })
          resolve()
        }
        tx.onerror = tx.onabort = () => resolve()
      })
    } else {
      // Small Smart subsets do not load unrelated vectors into memory.
      for (let start = 0; start < hits.length; start += 512) {
        await new Promise<void>((resolve) => {
          const tx = db.transaction(STORE)
          const store = tx.objectStore(STORE)
          for (const key of hits.slice(start, start + 512)) {
            const request = store.get(key)
            request.onsuccess = () => {
              const embedding = decode(request.result)
              if (embedding) result.set(key, embedding)
            }
          }
          tx.oncomplete = tx.onerror = tx.onabort = () => resolve()
        })
      }
    }
  } catch {
    /* Missing/invalid entries are downloaded rather than silently skipped. */
  }
  return result
}

export async function writeEmbeddings(
  db: IDBDatabase | null,
  records: Array<{ key: string; embedding: Float32Array }>
): Promise<void> {
  if (!db || !records.length) return
  await new Promise<void>((resolve) => {
    try {
      const tx = db.transaction(STORE, "readwrite")
      const store = tx.objectStore(STORE)
      for (const { key, embedding } of records)
        store.put(embedding.slice().buffer, key)
      tx.oncomplete = tx.onerror = tx.onabort = () => resolve()
    } catch {
      resolve()
    }
  })
}
