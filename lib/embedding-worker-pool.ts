const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_large/float32/latest/mobilenet_v3_large.tflite"
export interface EmbeddingInput {
  localIdx: number
  blob: Blob
}
export interface EmbeddingOutput {
  localIdx: number
  embedding: ArrayBuffer
}
interface Slot {
  worker: Worker
  busy: boolean
}
interface Task {
  items: EmbeddingInput[]
  resolve: (rows: EmbeddingOutput[]) => void
  reject: (error: unknown) => void
}
let model: Promise<ArrayBuffer> | undefined
let warmPool: EmbeddingWorkerPool | undefined

export function disposeEmbeddingWorkers(): void {
  warmPool?.dispose()
}

function loadModel(): Promise<ArrayBuffer> {
  if (model) return model
  const controller = new AbortController()
  const timeout = setTimeout(
    () => controller.abort(new Error("Model download timed out")),
    30_000
  )
  return (model = fetch(MODEL_URL, { signal: controller.signal })
    .then((response) => {
      if (!response.ok)
        throw new Error(`Failed to download model: HTTP ${response.status}`)
      return response.arrayBuffer()
    })
    .catch((error) => {
      model = undefined
      throw error
    })
    .finally(() => clearTimeout(timeout)))
}

export class EmbeddingWorkerPool {
  private slots: Slot[] = []
  private queue: Task[] = []
  private active = new Set<Task>()
  private idleTimer: ReturnType<typeof setTimeout> | undefined
  private disposed = false
  private ready: Promise<void>
  private rejectClosed!: (error: unknown) => void
  private closed = new Promise<never>((_resolve, reject) => {
    this.rejectClosed = reject
  })
  private initializing = new Set<(error: unknown) => void>()
  private leased = true
  private initialized = false

  private constructor(count: number) {
    this.ready = this.initialize(count).then(() => {
      this.initialized = true
    })
    // A download can fail before the first batch ever awaits initialization.
    void this.ready.catch(() => {})
    void this.closed.catch(() => {})
  }

  static acquire(itemCount: number): EmbeddingWorkerPool {
    const count = Math.max(
      1,
      Math.min(Math.ceil(itemCount / 20), navigator.hardwareConcurrency || 4, 8)
    )
    if (
      warmPool &&
      !warmPool.disposed &&
      !warmPool.leased &&
      warmPool.initialized
    ) {
      clearTimeout(warmPool.idleTimer)
      warmPool.leased = true
      if (warmPool.slots.length > count)
        warmPool.slots.splice(count).forEach((slot) => slot.worker.terminate())
      if (warmPool.slots.length < count) {
        const pool = warmPool
        pool.initialized = false
        pool.ready = pool.initialize(count - pool.slots.length).then(() => {
          pool.initialized = true
        })
        void pool.ready.catch(() => {})
      }
      return warmPool
    }
    if (warmPool && !warmPool.leased) warmPool.dispose()
    return (warmPool = new EmbeddingWorkerPool(count))
  }

  private async initialize(count: number): Promise<void> {
    try {
      const buffer = await loadModel()
      if (this.disposed) throw new DOMException("Aborted", "AbortError")
      const workerUrl = chrome.runtime.getURL("scripts/embedder-worker.js")
      await Promise.all(
        Array.from({ length: count }, async () => {
          const slot: Slot = { worker: new Worker(workerUrl), busy: false }
          this.slots.push(slot)
          await new Promise<void>((resolve, reject) => {
            const timeout = setTimeout(
              () =>
                reject(new Error("Embedding worker initialization timed out")),
              30_000
            )
            const failed = (error: unknown) => {
              clearTimeout(timeout)
              this.initializing.delete(failed)
              reject(error)
            }
            this.initializing.add(failed)
            slot.worker.onmessage = (event) => {
              if (event.data.type === "ready") {
                clearTimeout(timeout)
                this.initializing.delete(failed)
                resolve()
              } else if (event.data.type === "initError")
                failed(new Error(event.data.message))
            }
            slot.worker.onerror = (event) =>
              failed(
                new Error(
                  event.message || "Embedding worker initialization failed"
                )
              )
            const copy = buffer.slice(0)
            slot.worker.postMessage(
              {
                type: "init",
                data: {
                  wasmLoaderUrl: chrome.runtime.getURL(
                    "scripts/vision_wasm_internal.js"
                  ),
                  wasmBinaryUrl: chrome.runtime.getURL(
                    "scripts/vision_wasm_internal.wasm"
                  ),
                  modelBuffer: copy
                }
              },
              [copy]
            )
          })
        })
      )
    } catch (error) {
      this.dispose(error)
      throw error
    }
  }

  async embed(items: EmbeddingInput[]): Promise<EmbeddingOutput[]> {
    await Promise.race([this.ready, this.closed])
    if (this.disposed) throw new DOMException("Aborted", "AbortError")
    return new Promise((resolve, reject) => {
      this.queue.push({ items, resolve, reject })
      this.dispatch()
    })
  }

  private dispatch(): void {
    for (const slot of this.slots) {
      if (slot.busy || !this.queue.length || this.disposed) continue
      const task = this.queue.shift()!
      this.active.add(task)
      slot.busy = true
      slot.worker.onmessage = (event) => {
        if (event.data.type !== "results") return
        this.active.delete(task)
        slot.busy = false
        task.resolve(event.data.results)
        this.dispatch()
      }
      slot.worker.onerror = (event) =>
        this.dispose(new Error(event.message || "Embedding worker failed"))
      try {
        slot.worker.postMessage({ type: "embed", data: { items: task.items } })
      } catch (error) {
        this.dispose(error)
      }
    }
  }

  release(): void {
    this.leased = false
    if (!this.disposed) {
      // Retain only a small warm pool while large matching matrices are live.
      void this.ready
        .then(() => {
          if (!this.leased && !this.disposed)
            this.slots.splice(2).forEach((slot) => slot.worker.terminate())
        })
        .catch(() => {})
      this.idleTimer = setTimeout(() => this.dispose(), 60_000)
    }
  }

  dispose(error: unknown = new DOMException("Aborted", "AbortError")): void {
    if (this.disposed) return
    this.disposed = true
    this.rejectClosed(error)
    for (const reject of this.initializing) reject(error)
    this.initializing.clear()
    clearTimeout(this.idleTimer)
    this.slots.forEach((slot) => slot.worker.terminate())
    for (const task of [...this.queue, ...this.active]) task.reject(error)
    this.queue = []
    this.active.clear()
    if (warmPool === this) warmPool = undefined
  }
}
