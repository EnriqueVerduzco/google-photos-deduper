export interface MatchingProgress {
  phase: "detecting_duplicates"
  current: number
  total: number
}

/** Independent jobs keep each bucket's pair order and restore source result order. */
export async function runSmartWorkerPool(
  rows: Float32Array[],
  keys: string[],
  threshold: number,
  buckets: number[][],
  workerUrl: string,
  onProgress?: (progress: MatchingProgress) => void,
  signal?: AbortSignal
): Promise<number[][]> {
  signal?.throwIfAborted()
  if (!buckets.length) return []
  const jobs: number[][][] = []
  let batch: number[][] = [],
    size = 0
  for (const bucket of buckets) {
    if (batch.length && (size + bucket.length > 4096 || batch.length >= 128)) {
      jobs.push(batch)
      batch = []
      size = 0
    }
    batch.push(bucket)
    size += bucket.length
  }
  if (batch.length) jobs.push(batch)
  const count = Math.min(
    jobs.length,
    Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 4) - 1))
  )
  const workers: Worker[] = []
  const results: number[][][] = new Array(jobs.length)
  const done = new Array(jobs.length).fill(0)
  let next = 0,
    completed = 0,
    lastProgress = 0,
    cacheHits = 0,
    stopped = false
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      stopped = true
      workers.forEach((worker) => worker.terminate())
      signal?.removeEventListener("abort", abort)
    }
    const fail = (error: unknown) => {
      if (!stopped) {
        cleanup()
        reject(error)
      }
    }
    const abort = () =>
      fail(signal?.reason ?? new DOMException("Aborted", "AbortError"))
    signal?.addEventListener("abort", abort, { once: true })
    const progress = (force = false) => {
      const now = performance.now()
      if (force || now - lastProgress >= 100) {
        onProgress?.({
          phase: "detecting_duplicates",
          current: done.reduce((a, b) => a + b, 0),
          total: buckets.length
        })
        lastProgress = now
      }
    }
    const dispatch = (worker: Worker) => {
      if (stopped) return
      if (signal?.aborted) {
        abort()
        return
      }
      const job = next++
      if (job >= jobs.length) return
      const mapping: number[] = []
      const localBuckets = jobs[job].map((bucket) =>
        bucket.map((index) => {
          mapping.push(index)
          return mapping.length - 1
        })
      )
      const dim = rows[0].length
      const flat = new Float32Array(mapping.length * dim)
      mapping.forEach((index, i) => flat.set(rows[index], i * dim))
      worker.onmessage = (event) => {
        if (stopped) return
        if (event.data.type === "detectionProgress") {
          done[job] = event.data.current
          progress()
        } else if (event.data.type === "detectionError") {
          fail(new Error(event.data.message))
        } else if (event.data.type === "detectionResults") {
          cacheHits += event.data.cacheHits || 0
          results[job] = event.data.groups.map((group: number[]) =>
            group.map((index) => mapping[index])
          )
          done[job] = jobs[job].length
          if (++completed === jobs.length) {
            console.log(
              `[GPD] Smart matching: ${buckets.length} buckets, ${count} workers, ${cacheHits} cached buckets`
            )
            progress(true)
            cleanup()
            resolve(results.flat())
          } else {
            progress()
            dispatch(worker)
          }
        }
      }
      worker.onerror = (event) =>
        fail(new Error(event.message || "Smart matching worker failed"))
      worker.postMessage(
        {
          type: "detectSmart",
          data: {
            flatEmbeddings: flat,
            n: mapping.length,
            dim,
            threshold,
            buckets: localBuckets,
            mediaKeys: mapping.map((index) => keys[index])
          }
        },
        [flat.buffer]
      )
    }
    try {
      progress(true)
      for (let i = 0; i < count && !stopped; i++) {
        const worker = new Worker(workerUrl)
        workers.push(worker)
        dispatch(worker)
      }
    } catch (error) {
      fail(error)
    }
  })
}
