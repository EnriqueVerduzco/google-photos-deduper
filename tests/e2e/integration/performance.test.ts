import path from "node:path"
import { expect, test, type BrowserContext } from "@playwright/test"
import { build } from "esbuild"

import {
  clearStorage,
  injectScanResults,
  launchExtension,
  makeGroups,
  openAppTab
} from "../fixtures/extension"

let context: BrowserContext, extensionId: string
test.beforeAll(async () => {
  ;({ context, extensionId } = await launchExtension())
})
test.afterAll(async () => {
  await context.close()
})

test("virtualizes large results while preserving selection, preview and snapshot reload", async () => {
  await clearStorage(context)
  const { groups, mediaItems } = makeGroups(1000, 2)
  await injectScanResults(context, groups, mediaItems, 2000)
  await context.serviceWorkers()[0].evaluate(async () => {
    const { scanResults } = await chrome.storage.local.get("scanResults")
    await chrome.storage.local.set({
      scanResults: { ...scanResults, accountEmail: "performance@example.com" }
    })
  })
  const page = await openAppTab(context, extensionId)
  await expect(page.getByText("1000 Duplicate Groups Found")).toBeVisible()
  const list = page.getByTestId("virtual-groups")
  await expect(list).toBeVisible()
  expect(await list.locator(".MuiPaper-root").count()).toBeLessThan(20)
  await expect
    .poll(() =>
      context.serviceWorkers()[0].evaluate(async () => {
        const { scanResults } = await chrome.storage.local.get("scanResults")
        return Boolean(scanResults?.mediaSnapshotId && !scanResults.mediaItems)
      })
    )
    .toBe(true)
  const scrollBottom = () =>
    list.locator(":scope > div").evaluate((element) => {
      element.scrollTop = element.scrollHeight
    })
  await scrollBottom()
  const last = page
    .getByTestId("trash-whole-group-group-999")
    .locator("xpath=ancestor::div[contains(@class, 'MuiPaper-root')][1]")
  await expect(last).toBeVisible()
  await last.locator("input[type=checkbox]").click()
  await expect
    .poll(() =>
      context.serviceWorkers()[0].evaluate(async () => {
        const { selections } = await chrome.storage.local.get("selections")
        return selections.selectedGroupIds.includes("group-999")
      })
    )
    .toBe(false)
  await last.locator(".MuiCard-root").first().hover()
  await page.keyboard.press("Space")
  await expect(page.getByRole("dialog")).toBeVisible()
  await page.keyboard.press("Escape")
  await page.reload()
  await expect(page.getByText("1000 Duplicate Groups Found")).toBeVisible()
  await scrollBottom()
  await expect(last.locator("input[type=checkbox]")).not.toBeChecked()
  expect(await list.locator(".MuiPaper-root").count()).toBeLessThan(20)
  await page.close()
})

test("keeps image pixels identical and reuses exact Smart results in a real browser", async () => {
  const bundle = await build({
    entryPoints: [
      path.resolve(__dirname, "../../..", "workers/embedder.worker.ts")
    ],
    bundle: true,
    write: false,
    platform: "browser",
    format: "iife",
    plugins: [
      {
        name: "pixel-embedding-probe",
        setup(api) {
          api.onResolve({ filter: /^@mediapipe\/tasks-vision$/ }, () => ({
            path: "probe",
            namespace: "probe"
          }))
          api.onLoad({ filter: /.*/, namespace: "probe" }, () => ({
            contents: `export const ImageEmbedder = { createFromOptions: async () => ({ embed: (image) => ({ embeddings: [{ floatEmbedding: image.data }] }) }) }`
          }))
        }
      }
    ]
  })
  await context.route("https://performance.test/**", (route) =>
    route.fulfill({
      contentType: "text/html",
      body: "<html><body>Local performance test</body></html>"
    })
  )
  const page = await context.newPage()
  await page.goto("https://performance.test/")
  await page.evaluate(() => {
    const target = window as any
    target.workerMessages = []
    target.importScripts = () => {}
    target.addEventListener = (_type: string, listener: any) => {
      target.workerListener = listener
    }
    target.postMessage = (message: any) => target.workerMessages.push(message)
  })
  await page.evaluate(bundle.outputFiles[0].text)
  const pixels = await page.evaluate(async () => {
    const target = window as any,
      OriginalCanvas = OffscreenCanvas
    const inputs: Array<{ localIdx: number; blob: Blob }> = [],
      expected: number[][] = []
    for (let i = 0; i < 4; i++) {
      const canvas = new OriginalCanvas(i < 2 ? 2 : i + 1, i < 2 ? 2 : 1)
      const ctx = canvas.getContext("2d")!
      const data = ctx.createImageData(canvas.width, canvas.height)
      for (let k = 0; k < data.data.length; k++)
        data.data[k] = k % 4 === 3 ? (i % 2 ? 0 : 128) : (k * 37 + i * 53) % 256
      ctx.putImageData(data, 0, 0)
      const blob = await canvas.convertToBlob({ type: "image/png" })
      inputs.push({ localIdx: i, blob })
      const bitmap = await createImageBitmap(blob)
      const fresh = new OriginalCanvas(bitmap.width, bitmap.height),
        freshContext = fresh.getContext("2d")!
      freshContext.drawImage(bitmap, 0, 0)
      expected.push(
        Array.from(
          freshContext.getImageData(0, 0, bitmap.width, bitmap.height).data
        )
      )
      bitmap.close()
    }
    let allocations = 0
    target.OffscreenCanvas = new Proxy(OriginalCanvas, {
      construct(ctor, args) {
        allocations++
        return Reflect.construct(ctor, args)
      }
    })
    await target.workerListener({
      data: {
        type: "init",
        data: {
          wasmLoaderUrl: "local",
          wasmBinaryUrl: "local",
          modelBuffer: new ArrayBuffer(0)
        }
      }
    })
    target.workerMessages = []
    await target.workerListener({
      data: { type: "embed", data: { items: inputs } }
    })
    const actual = target.workerMessages
      .at(-1)
      .results.map((row: any) => Array.from(new Float32Array(row.embedding)))
    return { expected, actual, allocations }
  })
  expect(pixels.actual).toEqual(pixels.expected)
  expect(pixels.allocations).toBe(1)
  const cached = await page.evaluate(async () => {
    const target = window as any
    const data = {
      flatEmbeddings: new Float32Array(
        Array.from({ length: 128 }, (_, i) => (i % 2 ? 0 : 1))
      ),
      n: 64,
      dim: 2,
      threshold: 0.99,
      buckets: [Array.from({ length: 64 }, (_, i) => i)],
      mediaKeys: Array.from({ length: 64 }, (_, i) => `k${i}`)
    }
    const run = async () => {
      target.workerMessages = []
      await target.workerListener({ data: { type: "detectSmart", data } })
      return target.workerMessages.at(-1)
    }
    const first = await run(),
      second = await run()
    data.threshold = 1
    const changed = await run()
    return { first, second, changed }
  })
  expect(cached.first.type).toBe("detectionResults")
  expect(cached.first.cacheHits).toBe(0)
  expect(cached.second.cacheHits).toBe(1)
  expect(cached.second.groups).toEqual(cached.first.groups)
  expect(cached.changed.cacheHits).toBe(0)
  await page.close()
})
