import { afterEach, expect, it, vi } from "vitest"

import Api from "../../Google-Photos-Toolkit/src/api/api"

vi.mock("../../Google-Photos-Toolkit/src/windowGlobalData", () => ({
  windowGlobalData: { path: "/", at: "local-test" }
}))

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})
it("observes actual RPC attempts including retries without adding any requests", async () => {
  vi.useFakeTimers()
  vi.spyOn(console, "error").mockImplementation(() => {})
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
