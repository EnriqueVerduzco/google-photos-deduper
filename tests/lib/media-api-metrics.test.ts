import { afterEach, beforeEach, expect, it, vi } from "vitest"

import Api from "../../Google-Photos-Toolkit/src/api/api"

vi.mock("../../Google-Photos-Toolkit/src/windowGlobalData", () => ({
  windowGlobalData: { path: "/", at: "local-test" }
}))

// The exact error frame supplied by the user, including the anti-XSSI prefix
// and the trailing transport/telemetry records.
const status8Body = `)]}'

104
[["wrb.fr","EzkLib",null,null,null,[8],"generic"],["di",28],["af.httprm",27,"8976668271959269130",33]]
25
[["e",4,null,null,140]]
`

const rpcResponse = (payload: unknown) => ({
  ok: true,
  text: async () => JSON.stringify([["wrb.fr", "EzkLib", JSON.stringify(payload)]])
})
const status8Response = () => ({ ok: true, text: async () => status8Body })

beforeEach(() => {
  vi.useFakeTimers()
  vi.spyOn(console, "error").mockImplementation(() => {})
  vi.spyOn(Math, "random").mockReturnValue(0)
})

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})
it("observes actual RPC attempts including retries without adding any requests", async () => {
  const payload = [null, null]
  const fetch = vi
    .fn()
    .mockRejectedValueOnce(new Error("transient"))
    .mockResolvedValueOnce({
      ok: true,
      text: async () =>
        JSON.stringify([["wrb.fr", "EzkLib", JSON.stringify(payload)]])
    })
  vi.stubGlobal("fetch", fetch)
  const observe = vi.fn()
  const pending = new Api().getItemsByUploadedDate(null, false, observe)
  await vi.runAllTimersAsync()
  expect(await pending).toEqual(payload)
  expect(observe.mock.calls).toEqual([[1], [2]])
  expect(fetch).toHaveBeenCalledTimes(2)
})

it("backs off on the supplied status-8 response and retries the same page", async () => {
  const payload = [[], "following-page"]
  const fetch = vi.fn()
    .mockResolvedValueOnce(status8Response())
    .mockResolvedValueOnce(status8Response())
    .mockResolvedValueOnce(rpcResponse(payload))
  vi.stubGlobal("fetch", fetch)
  const observe = vi.fn()
  const onRetry = vi.fn()
  const pending = new Api().getItemsByUploadedDate("current-page", false, observe, { onRetry })

  await vi.advanceTimersByTimeAsync(9999)
  expect(fetch).toHaveBeenCalledTimes(1)
  await vi.advanceTimersByTimeAsync(1)
  expect(fetch).toHaveBeenCalledTimes(2)
  await vi.advanceTimersByTimeAsync(19_999)
  expect(fetch).toHaveBeenCalledTimes(2)
  await vi.advanceTimersByTimeAsync(1)
  expect(await pending).toEqual(payload)
  expect(observe.mock.calls).toEqual([[1], [2], [3]])
  expect(onRetry.mock.calls.map(([info]) => info.delayMs)).toEqual([10_000, 20_000])
  expect(onRetry).toHaveBeenCalledWith(expect.objectContaining({ rateLimited: true, maxAttempts: 5 }))
  expect(new Set(fetch.mock.calls.map(([, init]) => init.body)).size).toBe(1)
})

it("bounds persistent throttling and preserves the RPC status in the final error", async () => {
  const fetch = vi.fn().mockResolvedValue(status8Response())
  vi.stubGlobal("fetch", fetch)
  const onRetry = vi.fn()
  const result = new Api().getItemsByUploadedDate(null, false, undefined, { onRetry })
    .then(() => { throw new Error("Expected the RPC to fail") }, (error: Error) => error)
  await vi.runAllTimersAsync()

  expect(await result).toBeInstanceOf(Error)
  expect((await result).message).toMatch(/RPC status 8.*EzkLib.*resource limit/)
  expect((await result).message).toContain("Wait a few minutes")
  expect(fetch).toHaveBeenCalledTimes(5)
  expect(onRetry.mock.calls.map(([info]) => info.delayMs)).toEqual([10_000, 20_000, 40_000, 60_000])
})

it("uses the throttling retry policy for HTTP 429 as well", async () => {
  const fetch = vi.fn()
    .mockResolvedValueOnce({ ok: false, status: 429, statusText: "Too Many Requests" })
    .mockResolvedValueOnce(rpcResponse([[], null]))
  vi.stubGlobal("fetch", fetch)
  const onRetry = vi.fn()
  const pending = new Api().getItemsByUploadedDate(null, false, undefined, { onRetry })
  await vi.advanceTimersByTimeAsync(9999)
  expect(fetch).toHaveBeenCalledTimes(1)
  await vi.advanceTimersByTimeAsync(1)
  expect(await pending).toEqual([[], null])
  expect(onRetry).toHaveBeenCalledWith(expect.objectContaining({ rateLimited: true, delayMs: 10_000 }))
})

it("matches the requested RPC when other result or metadata rows come first", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
    ok: true,
    text: async () => JSON.stringify([
      ["di", 28],
      ["wrb.fr", "otherRpc", null, null, null, [8], "generic"],
      ["wrb.fr", "EzkLib", JSON.stringify([[], null]), null, null, [0], "generic"]
    ])
  }))
  expect(await new Api().getItemsByUploadedDate(null, false)).toEqual([[], null])
})

it("reports other Google RPC errors without mislabeling them as missing payloads", async () => {
  const fetch = vi.fn().mockResolvedValue({
    ok: true,
    text: async () => JSON.stringify([["wrb.fr", "EzkLib", null, null, null, [16], "generic"]])
  })
  vi.stubGlobal("fetch", fetch)
  const result = new Api().getItemsByUploadedDate(null, false).then(() => { throw new Error("Expected the RPC to fail") }, (error: Error) => error)
  await vi.runAllTimersAsync()
  expect((await result).message).toContain("RPC status 16")
  expect(fetch).toHaveBeenCalledTimes(3)
})

it("fails an unexplained missing payload instead of returning an empty library", async () => {
  const fetch = vi.fn().mockResolvedValue({
    ok: true,
    text: async () => JSON.stringify([["wrb.fr", "EzkLib", null]])
  })
  vi.stubGlobal("fetch", fetch)
  const result = new Api().getItemsByUploadedDate(null, false).then(() => { throw new Error("Expected the RPC to fail") }, (error: Error) => error)
  await vi.runAllTimersAsync()
  expect((await result).message).toContain("Missing payload in EzkLib response")
  expect(fetch).toHaveBeenCalledTimes(3)
})

it("cancels the backoff without issuing detached retry requests", async () => {
  const fetch = vi.fn().mockResolvedValue(status8Response())
  vi.stubGlobal("fetch", fetch)
  const controller = new AbortController()
  const result = new Api().getItemsByUploadedDate(null, false, undefined, { signal: controller.signal })
    .then(() => { throw new Error("Expected the RPC to fail") }, (error: Error) => error)
  await vi.advanceTimersByTimeAsync(0)
  controller.abort()
  expect((await result).name).toBe("AbortError")
  await vi.runAllTimersAsync()
  expect(fetch).toHaveBeenCalledTimes(1)
  expect(vi.getTimerCount()).toBe(0)
})
