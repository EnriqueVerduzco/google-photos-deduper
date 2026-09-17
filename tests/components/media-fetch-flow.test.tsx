import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor
} from "@testing-library/react"
import { afterEach, beforeEach, expect, it, vi } from "vitest"

import { createMediaIndex, mediaIndexKey } from "../../lib/media-index"
import type {
  AppMessage,
  GpdMediaItem,
  GptkCommandMessage
} from "../../lib/types"
import App from "../../tabs/app"

const detection = vi.hoisted(() => ({ smart: vi.fn(), full: vi.fn() }))
vi.mock("../../lib/duplicate-detector", () => ({
  smartDetectDuplicates: detection.smart,
  fullDetectDuplicates: detection.full,
  selectDefaultKeep: (items: GpdMediaItem[]) => items[0].mediaKey
}))
vi.mock("../../components/ScanConfig", () => ({
  ScanConfig: ({ onStartScan, onSettingsChange }: any) => (
    <>
      <button onClick={onStartScan}>Start local test scan</button>
      <button onClick={() => onSettingsChange({ mediaRefreshMode: "full" })}>
        Full refresh
      </button>
    </>
  )
}))
vi.mock("../../components/ScanProgress", () => ({
  ScanProgress: ({ phase, onCancel }: any) => (
    <>
      <div>{phase}</div>
      <button onClick={onCancel}>Cancel local test scan</button>
    </>
  )
}))
vi.mock("../../components/DuplicateGroups", () => ({
  DuplicateGroups: () => null
}))
vi.mock("../../components/ActionBar", () => ({
  ActionBar: ({ onRescan }: any) => <button onClick={onRescan}>Rescan</button>
}))
vi.mock("canvas-confetti", () => ({ default: vi.fn() }))

let store: Record<string, any>
let listeners: Set<(message: AppMessage, sender: object) => void>
let sent: any[]
const account = "a@example.com"
const item = (key: string, date = 100): GpdMediaItem => ({
  mediaKey: key,
  dedupKey: key,
  thumb: "local",
  timestamp: date,
  creationTimestamp: date
})
function emit(message: unknown) {
  for (const listener of listeners) listener(message as AppMessage, {})
}
async function deliver(message: unknown) {
  await act(async () => {
    emit(message)
  })
}
const storage = {
  get: vi.fn((keys: string | string[], callback?: (value: any) => void) => {
    const values = Object.fromEntries(
      (Array.isArray(keys) ? keys : [keys]).map((key) => [key, store[key]])
    )
    if (callback) queueMicrotask(() => callback(values))
    return Promise.resolve(values)
  }),
  set: vi.fn(async (values: any) => {
    Object.assign(store, values)
  }),
  remove: vi.fn(async (keys: string | string[]) => {
    for (const key of Array.isArray(keys) ? keys : [keys]) delete store[key]
  })
}
beforeEach(() => {
  store = {}
  listeners = new Set()
  sent = []
  vi.clearAllMocks()
  detection.smart.mockResolvedValue([])
  detection.full.mockResolvedValue({ groups: [], timing: {} })
  vi.stubGlobal("chrome", {
    storage: { local: storage },
    runtime: {
      getURL: (path: string) => path,
      onMessage: {
        addListener: (fn: any) => listeners.add(fn),
        removeListener: (fn: any) => listeners.delete(fn)
      },
      sendMessage: vi.fn(async (message: any) => {
        sent.push(message)
        if (message.action === "healthCheck")
          queueMicrotask(() =>
            emit({
              app: "GPD",
              action: "healthCheck.result",
              success: true,
              hasGptk: true,
              accountEmail: account
            })
          )
      })
    }
  })
})
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})
async function mount() {
  render(<App />)
  await screen.findByRole("button", { name: "Start local test scan" })
}
async function start(): Promise<GptkCommandMessage> {
  const previous = sent.length
  fireEvent.click(screen.getByRole("button", { name: "Start local test scan" }))
  await waitFor(() =>
    expect(
      sent.slice(previous).some((m) => m.command === "getAllMediaItems")
    ).toBe(true)
  )
  return sent.slice(previous).find((m) => m.command === "getAllMediaItems")
}
const page = (requestId: string, rows: GpdMediaItem[], chunkIndex = 0) => ({
  app: "GPD",
  action: "gptkMediaPage",
  command: "getAllMediaItems",
  requestId,
  chunkIndex,
  data: rows
})
const end = (requestId: string, totalChunks = 1) => ({
  app: "GPD",
  action: "gptkMediaComplete",
  command: "getAllMediaItems",
  requestId,
  totalChunks,
  accountEmail: account,
  sentAt: Date.now(),
  metrics: {
    pages: totalChunks,
    itemsReceived: 1,
    itemsEmitted: 1,
    pageItems: [1],
    pageRequestMs: [10],
    requestMs: 10,
    requestAttempts: totalChunks,
    retries: 0,
    elapsedMs: 12,
    reachedCache: false
  }
})

it("saves the index before detection, retains it after zero matches, and uses it on the next scan", async () => {
  detection.smart.mockImplementation(async () => {
    expect(store[mediaIndexKey(account)!].mediaItems.a).toEqual(item("a"))
    return []
  })
  await mount()
  const first = await start()
  expect(first.args).toMatchObject({
    streamResults: true,
    accountEmail: account
  })
  await deliver(page(first.requestId, [item("a")]))
  expect(detection.smart).not.toHaveBeenCalled()
  await deliver(end(first.requestId))
  await screen.findByText("No duplicates found in your library.")
  expect(store.scanResults).toBeUndefined()
  expect(store[mediaIndexKey(account)!]).toBeDefined()
  expect(store.scanLogs.at(-1)).toMatchObject({
    totalItems: 1,
    groupsFound: 0,
    mediaFetch: { pages: 1, mode: "full" }
  })
  fireEvent.click(screen.getByRole("button", { name: "Back to Scan" }))
  await screen.findByRole("button", { name: "Start local test scan" })
  const second = await start()
  expect(second.args).toMatchObject({ sinceTimestamp: 100 })
  fireEvent.click(
    screen.getByRole("button", { name: "Cancel local test scan" })
  )
})

it("commits only after every distinct page and merges in source order", async () => {
  store[mediaIndexKey(account)!] = createMediaIndex(account, [
    item("cached", 50)
  ])
  await mount()
  const command = await start()
  await deliver(end(command.requestId, 2))
  await deliver(page(command.requestId, [item("second", 100)], 1))
  await deliver(page(command.requestId, [item("second", 100)], 1))
  expect(detection.smart).not.toHaveBeenCalled()
  expect(Object.keys(store[mediaIndexKey(account)!].mediaItems)).toEqual([
    "cached"
  ])
  await deliver(page(command.requestId, [item("first", 200)], 0))
  await waitFor(() => expect(detection.smart).toHaveBeenCalledTimes(1))
  expect(
    detection.smart.mock.calls[0][0].map((row: GpdMediaItem) => row.mediaKey)
  ).toEqual(["first", "second", "cached"])
})

it("keeps the previous index when a partial refresh fails", async () => {
  const prior = createMediaIndex(account, [item("old")])
  store[mediaIndexKey(account)!] = prior
  await mount()
  const command = await start()
  await deliver(page(command.requestId, [item("partial", 200)]))
  await deliver({
    app: "GPD",
    action: "gptkResult",
    command: "getAllMediaItems",
    requestId: command.requestId,
    success: false,
    error: "local test timeout"
  })
  expect(store[mediaIndexKey(account)!]).toBe(prior)
  expect(detection.smart).not.toHaveBeenCalled()
  await waitFor(() => expect(store.scanLogs.at(-1).status).toBe("error"))
})

it("ignores delayed pages and completion after fetch cancellation", async () => {
  const prior = createMediaIndex(account, [item("old")])
  store[mediaIndexKey(account)!] = prior
  await mount()
  const command = await start()
  await deliver(page(command.requestId, [item("partial", 200)]))
  fireEvent.click(
    screen.getByRole("button", { name: "Cancel local test scan" })
  )
  await deliver(end(command.requestId))
  expect(store[mediaIndexKey(account)!]).toBe(prior)
  expect(detection.smart).not.toHaveBeenCalled()
})

it("retains a completed fetch when detection is cancelled", async () => {
  detection.smart.mockImplementation(
    (_items, _threshold, _window, _progress, signal: AbortSignal) =>
      new Promise((_resolve, reject) => {
        signal.addEventListener(
          "abort",
          () => reject(new DOMException("Aborted", "AbortError")),
          { once: true }
        )
      })
  )
  await mount()
  const command = await start()
  await deliver(page(command.requestId, [item("saved")]))
  await deliver(end(command.requestId))
  await waitFor(() => expect(detection.smart).toHaveBeenCalledOnce())
  fireEvent.click(
    screen.getByRole("button", { name: "Cancel local test scan" })
  )
  await waitFor(() => expect(store.scanLogs.at(-1).status).toBe("cancelled"))
  expect(Object.keys(store[mediaIndexKey(account)!].mediaItems)).toEqual([
    "saved"
  ])
})

it("full refresh bypasses the watermark and replaces the index after success", async () => {
  store[mediaIndexKey(account)!] = createMediaIndex(account, [
    item("deleted-externally")
  ])
  await mount()
  fireEvent.click(screen.getByRole("button", { name: "Full refresh" }))
  const command = await start()
  expect((command.args as any).sinceTimestamp).toBeUndefined()
  await deliver(page(command.requestId, [item("current")]))
  await deliver(end(command.requestId))
  await waitFor(() => expect(detection.smart).toHaveBeenCalledOnce())
  expect(Object.keys(store[mediaIndexKey(account)!].mediaItems)).toEqual([
    "current"
  ])
  expect(
    detection.smart.mock.calls[0][0].map((row: GpdMediaItem) => row.mediaKey)
  ).toEqual(["current"])
})

it("rejects a completed stream for a different account without replacing the index", async () => {
  const prior = createMediaIndex(account, [item("old")])
  store[mediaIndexKey(account)!] = prior
  await mount()
  const command = await start()
  await deliver(page(command.requestId, [item("wrong-account", 200)]))
  await deliver({
    ...end(command.requestId),
    accountEmail: "other@example.com"
  })
  expect(store[mediaIndexKey(account)!]).toBe(prior)
  expect(store[mediaIndexKey("other@example.com")!]).toBeUndefined()
  expect(detection.smart).not.toHaveBeenCalled()
})

it("does not send a fetch command if cancelled while loading the index", async () => {
  let resolveCache!: (value: unknown) => void
  const original = storage.get.getMockImplementation()!
  storage.get.mockImplementation((keys, callback) => {
    if (keys === mediaIndexKey(account))
      return new Promise((resolve) => {
        resolveCache = resolve
      }) as any
    return original(keys, callback)
  })
  try {
    await mount()
    fireEvent.click(
      screen.getByRole("button", { name: "Start local test scan" })
    )
    await waitFor(() => expect(resolveCache).toBeDefined())
    fireEvent.click(
      screen.getByRole("button", { name: "Cancel local test scan" })
    )
    await act(async () =>
      resolveCache({
        [mediaIndexKey(account)!]: createMediaIndex(account, [item("old")])
      })
    )
    expect(sent.some((message) => message.command === "getAllMediaItems")).toBe(
      false
    )
  } finally {
    storage.get.mockImplementation(original)
  }
})
