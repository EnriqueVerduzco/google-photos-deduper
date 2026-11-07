// MAIN world command handler for Google Photos pages.
// This script runs in the page's JS context (injected via <script> tag)
// and has access to GPTK globals: window.gptkApi, window.gptkCore, window.gptkApiUtils.
//
// Communication with the extension happens via window.postMessage.
// The bridge content script (google-photos-bridge.ts) relays these to chrome.runtime.

const GPD_APP_ID = "GPD"

// Number of items per API request for batch operations (trash, restore).
// Matches GPTK's default operationSize. Large single requests cause HTTP 504.
const BATCH_SIZE = 250

function postResult(command, requestId, data) {
  window.postMessage({
    app: GPD_APP_ID,
    action: "gptkResult",
    command,
    requestId,
    success: true,
    data
  })
}

function postError(command, requestId, error, mediaFetchMetrics) {
  window.postMessage({
    app: GPD_APP_ID,
    action: "gptkResult",
    command,
    requestId,
    success: false,
    error: String(error),
    ...(mediaFetchMetrics ? { mediaFetchMetrics } : {})
  })
}

// Number of media items per result chunk.
//
// Chrome caps a single extension message at 64MiB. Real-world media items
// serialize to ~460 bytes each, so a 190k-item library produces ~84MB — the
// oversize chrome.runtime.sendMessage in the bridge throws synchronously, the
// result never reaches the app, and the scan hangs on "Fetching media items"
// forever. 10k items is ~4.6MB, leaving room for items far larger than average.
const RESULT_CHUNK_SIZE = 10000

// Leave a short gap between library pages to reduce bursts against Google's
// private metadata API. Thumbnail download concurrency is independent of this.
const MIN_LIBRARY_PAGE_DELAY_MS = 250

// Sends an array result as a sequence of gptkResultChunk messages.
// Always sends at least one chunk so an empty result still completes.
function postResultChunked(command, requestId, items) {
  const totalChunks = Math.max(1, Math.ceil(items.length / RESULT_CHUNK_SIZE))
  for (let i = 0; i < totalChunks; i++) {
    window.postMessage({
      app: GPD_APP_ID,
      action: "gptkResultChunk",
      command,
      requestId,
      chunkIndex: i,
      totalChunks,
      data: items.slice(i * RESULT_CHUNK_SIZE, (i + 1) * RESULT_CHUNK_SIZE)
    })
  }
  console.log(
    `[GPD] sent ${items.length} items for ${command} in ${totalChunks} chunk(s)`
  )
}

// command is optional; when provided, the app can route progress to the right handler.
function postProgress(requestId, itemsProcessed, message, command, position) {
  window.postMessage({
    app: GPD_APP_ID,
    action: "gptkProgress",
    requestId,
    itemsProcessed,
    message,
    ...(command !== undefined ? { command } : {}),
    ...position
  })
}

function chunkArray(arr, size) {
  const chunks = []
  for (let i = 0; i < arr.length; i += size) {
    chunks.push(arr.slice(i, i + size))
  }
  return chunks
}

// ============================================================
// Command: getAllMediaItems
// Fetches all media items from the library via GPTK pagination.
// ============================================================

async function getAllMediaItems(requestId, args) {
  const gptkApi = window.gptkApi
  if (!gptkApi) {
    postError(
      "getAllMediaItems",
      requestId,
      "GPTK API not available. Reload the Google Photos page."
    )
    return
  }

  const streaming = args?.streamResults === true
  const expectedAccount = args?.accountEmail?.trim().toLowerCase()
  const getAccount = () => (window.WIZ_global_data?.oPEP7c || "").trim().toLowerCase()
  const assertAccount = () => {
    if (expectedAccount && getAccount() !== expectedAccount) {
      throw new Error("The Google Photos account changed. Reconnect before scanning.")
    }
  }
  const startedAt = performance.now()
  let pageDelayMs = 500
  let healthyPages = 0
  const metrics = {
    pages: 0, itemsReceived: 0, itemsEmitted: 0,
    pageItems: [], pageRequestMs: [], requestMs: 0,
    requestAttempts: null, retries: null, elapsedMs: 0, reachedCache: false
  }

  // Each network attempt gets 60s. Retry backoff has its own allowance so a
  // rate-limited page can recover without the old total-page timer cutting it off.
  const PAGE_TIMEOUT_MS = 60_000

  function withTimeout(startRequest, ms, label) {
    let timer
    let resetTimeout
    const controller = new AbortController()
    const timeout = new Promise((_, reject) => {
      resetTimeout = (allowance = ms) => {
        clearTimeout(timer)
        timer = setTimeout(() => {
          reject(
            new Error(
              `${label} timed out after ${Math.round(ms / 1000)}s without a response. Please re-scan.`
            )
          )
          // Stop the fetch or retry sleep instead of leaving requests running
          // after the scan has already failed.
          controller.abort()
        }, allowance)
      }
      resetTimeout()
    })
    const request = Promise.resolve().then(() => startRequest(resetTimeout, controller.signal))
    return Promise.race([request, timeout]).finally(() => clearTimeout(timer))
  }

  try {
    assertAccount()
    let nextPageId = null
    const mediaItems = [] // Only populated for older callers without streaming.
    let chunkIndex = 0
    const seenPageIds = new Set()
    const accountUrlPrefix =
      window.location.pathname.match(/^\/u\/\d+\//)?.[0] || "/"

    // Google rejects EzkLib for some accounts. Taken-date pagination uses
    // lcxiM and must read every page: upload watermarks do not apply here.
    do {
      assertAccount()
      let retried = false
      const pageStart = performance.now()
      let page
      try {
        page = await withTimeout(
          (resetTimeout, signal) => gptkApi.getItemsByTakenDate(
            null,
            null,
            nextPageId,
            500,
            true,
            (attempt) => {
              if (attempt > 1) retried = true
              resetTimeout()
              metrics.requestAttempts = (metrics.requestAttempts ?? 0) + 1
              metrics.retries = (metrics.retries ?? 0) + (attempt > 1 ? 1 : 0)
              if (attempt > 1) {
                postProgress(requestId, metrics.itemsEmitted, `Retrying library page (attempt ${attempt})`)
              }
            },
            {
              signal,
              onRetry: ({ attempt, maxAttempts, delayMs, rateLimited }) => {
                retried = true
                healthyPages = 0
                if (rateLimited) pageDelayMs = Math.min(5000, Math.max(1000, pageDelayMs * 2))
                resetTimeout(PAGE_TIMEOUT_MS + delayMs)
                const reason = rateLimited
                  ? "Google Photos may be limiting requests."
                  : "Google Photos could not return this library page."
                postProgress(requestId, metrics.itemsEmitted,
                  `${reason} Waiting ${Math.ceil(delayMs / 1000)}s before retry ${attempt + 1} of ${maxAttempts}.`)
              }
            }
          ),
          PAGE_TIMEOUT_MS,
          "Fetching page from Google Photos"
        )
      } finally {
        const elapsed = performance.now() - pageStart
        metrics.requestMs += elapsed
        metrics.pageRequestMs.push(elapsed)
      }
      assertAccount()
      if (!page || !Array.isArray(page.items)) {
        throw new Error("Invalid library page; saved library details were not replaced.")
      }
      metrics.pages++
      metrics.itemsReceived += page.items.length
      metrics.pageItems.push(page.items.length)
      const pageItems = []
      for (const item of page.items) {
        pageItems.push({
          mediaKey: item.mediaKey,
          dedupKey: item.dedupKey,
          thumb: item.thumb,
          timestamp: item.timestamp,
          creationTimestamp: item.creationTimestamp,
          resWidth: item.resWidth,
          resHeight: item.resHeight,
          duration: item.duration,
          isOwned: item.isOwned,
          isOriginalQuality: item.isOriginalQuality ?? null,
          isFavorite: item.isFavorite === true,
          fileName: item.descriptionShort || null,
          productUrl: "https://photos.google.com" + accountUrlPrefix + "photo/" + item.mediaKey
        })
      }
      metrics.itemsEmitted += pageItems.length
      if (streaming) {
        // Bound message size, including when an API page itself is very large.
        for (let i = 0; i < pageItems.length; i += RESULT_CHUNK_SIZE) {
          window.postMessage({
            app: GPD_APP_ID, action: "gptkMediaPage", command: "getAllMediaItems",
            requestId, chunkIndex: chunkIndex++, data: pageItems.slice(i, i + RESULT_CHUNK_SIZE)
          })
        }
      } else {
        for (const item of pageItems) mediaItems.push(item)
      }
      nextPageId = page.nextPageId || null
      const oldest = pageItems[pageItems.length - 1]
      const position = Number.isFinite(oldest?.timestamp) && oldest.timestamp > 0
        ? { oldestTakenAt: oldest.timestamp } : undefined
      postProgress(requestId, metrics.itemsEmitted, `Fetched ${metrics.itemsEmitted} items`, undefined, position)
      if (nextPageId && seenPageIds.has(nextPageId)) throw new Error("Library pagination repeated a page token")
      if (nextPageId) {
        seenPageIds.add(nextPageId)
        if (!retried && ++healthyPages >= 4) {
          pageDelayMs = Math.max(MIN_LIBRARY_PAGE_DELAY_MS, Math.round(pageDelayMs * 0.7))
          healthyPages = 0
        }
        await new Promise((resolve) => setTimeout(resolve, pageDelayMs))
      }
    } while (nextPageId)

    metrics.elapsedMs = performance.now() - startedAt
    if (streaming) {
      window.postMessage({
        app: GPD_APP_ID, action: "gptkMediaComplete", command: "getAllMediaItems",
        requestId, totalChunks: chunkIndex, accountEmail: getAccount() || undefined,
        sentAt: Date.now(), metrics
      })
    } else {
      postResultChunked("getAllMediaItems", requestId, mediaItems)
    }
  } catch (error) {
    metrics.elapsedMs = performance.now() - startedAt
    postError("getAllMediaItems", requestId, error, metrics)
  }
}

// ============================================================
// Command: trashItems
// Moves items to trash via GPTK's batch API (no DOM clicking).
// ============================================================

async function trashItems(requestId, args) {
  // Call api.moveItemsToTrash directly instead of gptkApiUtils.moveToTrash,
  // because the latter goes through executeWithConcurrency which checks
  // gptkCore.isProcessRunning (always false when called from extension).
  const api = window.gptkApiUtils?.api
  if (!api) {
    postError(
      "trashItems",
      requestId,
      "GPTK API not available. Reload the Google Photos page."
    )
    return
  }

  try {
    const dedupKeys = args.dedupKeys
    const mediaKeysToTrash = args.mediaKeysToTrash || []
    const total = dedupKeys.length
    const chunks = chunkArray(dedupKeys, BATCH_SIZE)
    console.log(
      `[GPD] trash: ${total} items, ${chunks.length} chunk(s) of ${BATCH_SIZE}`
    )

    // Chunk to avoid HTTP 504 Gateway Timeout on large batches (fixes #107).
    let trashed = 0
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i]
      await api.moveItemsToTrash(chunk)
      trashed += chunk.length
      console.log(
        `[GPD] trash chunk ${i + 1}/${chunks.length}: ${trashed}/${total} done`
      )
      postProgress(
        requestId,
        trashed,
        `Moved ${trashed} of ${total} items to trash`,
        "trashItems"
      )
    }

    console.log(`[GPD] trash complete: ${total} items moved`)
    postResult("trashItems", requestId, {
      trashedCount: total,
      trashedKeys: mediaKeysToTrash
    })
  } catch (error) {
    console.error("[GPD] trash error:", error)
    postError("trashItems", requestId, error)
  }
}

// ============================================================
// Command: restoreItems
// Restores items from trash via GPTK's batch API.
// ============================================================

async function restoreItems(requestId, args) {
  // Call api.restoreFromTrash directly (same reason as trashItems —
  // executeWithConcurrency checks isProcessRunning which is always false here).
  const api = window.gptkApiUtils?.api
  if (!api) {
    postError(
      "restoreItems",
      requestId,
      "GPTK API not available. Reload the Google Photos page."
    )
    return
  }

  try {
    const dedupKeys = args.dedupKeys
    const total = dedupKeys.length
    const chunks = chunkArray(dedupKeys, BATCH_SIZE)
    console.log(
      `[GPD] restore: ${total} items, ${chunks.length} chunk(s) of ${BATCH_SIZE}`
    )

    // Chunk to avoid HTTP 504 Gateway Timeout on large batches.
    let restored = 0
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i]
      await api.restoreFromTrash(chunk)
      restored += chunk.length
      console.log(
        `[GPD] restore chunk ${i + 1}/${chunks.length}: ${restored}/${total} done`
      )
      postProgress(
        requestId,
        restored,
        `Restored ${restored} of ${total} items`,
        "restoreItems"
      )
    }

    console.log(`[GPD] restore complete: ${total} items restored`)
    postResult("restoreItems", requestId, { restoredCount: total })
  } catch (error) {
    console.error("[GPD] restore error:", error)
    postError("restoreItems", requestId, error)
  }
}

// ============================================================
// Command: healthCheck
// Verifies GPTK is loaded and WIZ_global_data is available.
// ============================================================

function healthCheck(requestId) {
  const hasGptk = typeof window.gptkApi !== "undefined"
  const hasWizData = typeof window.WIZ_global_data !== "undefined"
  // oPEP7c is the signed-in account email in WIZ_global_data
  const accountEmail = hasWizData ? window.WIZ_global_data.oPEP7c || "" : ""
  postResult("healthCheck", requestId, { hasGptk, hasWizData, accountEmail })
}

// ============================================================
// Message listener
// ============================================================

window.addEventListener("message", async (event) => {
  if (event.source !== window) return
  const msg = event.data
  if (msg?.app !== GPD_APP_ID || msg?.action !== "gptkCommand") return

  const { command, requestId, args } = msg
  console.log("GPD: Received command", command, requestId)

  switch (command) {
    case "getAllMediaItems":
      await getAllMediaItems(requestId, args)
      break
    case "trashItems":
      await trashItems(requestId, args)
      break
    case "restoreItems":
      await restoreItems(requestId, args)
      break
    case "healthCheck":
      healthCheck(requestId)
      break
    default:
      postError(command, requestId, `Unknown command: ${command}`)
  }
})

console.log("GPD: Command handler loaded")
