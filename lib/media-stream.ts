import type {
  GpdMediaItem,
  GptkMediaCompleteMessage,
  GptkMediaPageMessage
} from "./types"

/** Order-independent page assembly. Completion alone never commits partial data. */
export class MediaStream {
  private pages = new Map<number, GpdMediaItem[]>()
  private terminal?: GptkMediaCompleteMessage
  private finished = false

  accept(message: GptkMediaPageMessage | GptkMediaCompleteMessage): {
    items: GpdMediaItem[]
    completion: GptkMediaCompleteMessage
    finalTransferMs: number
  } | null {
    if (this.finished) return null
    if (message.action === "gptkMediaPage") {
      if (!Number.isSafeInteger(message.chunkIndex) || message.chunkIndex < 0)
        throw new Error("Invalid media page index")
      if (!this.pages.has(message.chunkIndex)) {
        this.pages.set(message.chunkIndex, message.data)
      }
    } else {
      if (!Number.isSafeInteger(message.totalChunks) || message.totalChunks < 0)
        throw new Error("Invalid media page total")
      this.terminal = message
    }
    const completion = this.terminal
    if (!completion || this.pages.size < completion.totalChunks) return null
    if (this.pages.size > completion.totalChunks)
      throw new Error("Unexpected media page")
    const items: GpdMediaItem[] = []
    for (let i = 0; i < completion.totalChunks; i++) {
      const page = this.pages.get(i)
      if (!page) throw new Error("Missing media page")
      for (const item of page) items.push(item)
    }
    this.finished = true
    this.pages.clear()
    // Shared wall-clock origin across contexts; clamp wall-clock adjustments.
    const finalTransferMs = Math.max(0, Date.now() - completion.sentAt)
    return { items, completion, finalTransferMs }
  }
}
