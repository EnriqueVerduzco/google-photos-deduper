/// <reference lib="webworker" />

// Standalone Web Worker for MediaPipe image embedding AND community detection.
// Built separately with esbuild so it can be loaded as a classic worker
// from the extension (CSP: script-src 'self').
//
// Message protocol (main → worker):
//   { type: "init", data: { wasmLoaderUrl, wasmBinaryUrl, modelBuffer: ArrayBuffer } }
//   { type: "embed", data: { items: Array<{ localIdx: number, blob: Blob }> } }
//   { type: "detect", data: { flatEmbeddings: Float32Array, n: number, dim: number, threshold: number } }
//   { type: "detectSmart", data: { flatEmbeddings: Float32Array, n: number, dim: number, threshold: number, buckets: number[][] } }
//
// Message protocol (worker → main):
//   { type: "ready" }
//   { type: "results", results: Array<{ localIdx: number, embedding: ArrayBuffer }> }
//   { type: "initError", message: string }
//   { type: "detectionProgress", current: number, total: number }
//   { type: "detectionResults", groups: number[][] }

import { ImageEmbedder } from "@mediapipe/tasks-vision";

import { fullCommunityDetection } from "../lib/full-matcher";
import { detectSmartBucket } from "../lib/smart-matcher";
import { SmartMatchCache } from "../lib/smart-match-cache";

let embedder: ImageEmbedder | null = null;
let canvas: OffscreenCanvas | null = null;
let context: OffscreenCanvasRenderingContext2D | null = null;
let smartCachePromise: Promise<SmartMatchCache | null> | undefined;

self.addEventListener("message", async (event: MessageEvent) => {
  const { type, data } = event.data;

  if (type === "init") {
    try {
      const { wasmLoaderUrl, wasmBinaryUrl, modelBuffer } = data as {
        wasmLoaderUrl: string;
        wasmBinaryUrl: string;
        modelBuffer: ArrayBuffer;
      };

      // Pre-load the WASM loader JS via importScripts (CSP-safe from extension origin).
      // This is the same workaround as the main thread: load the global manually so
      // MediaPipe can find it, then pass wasmLoaderPath: "" to skip its own injection.
      importScripts(wasmLoaderUrl);

      const vision = {
        wasmLoaderPath: "",
        wasmBinaryPath: wasmBinaryUrl,
      };

      embedder = await ImageEmbedder.createFromOptions(vision, {
        baseOptions: { modelAssetBuffer: new Uint8Array(modelBuffer) },
        quantize: false,
        l2Normalize: true,
        runningMode: "IMAGE",
      });

      self.postMessage({ type: "ready" });
    } catch (e) {
      self.postMessage({ type: "initError", message: String(e) });
    }
  }

  if (type === "embed") {
    const { items } = data as {
      items: Array<{ localIdx: number; blob: Blob }>;
    };

    const results: Array<{ localIdx: number; embedding: ArrayBuffer }> = [];

    for (const { localIdx, blob } of items) {
      let bitmap: ImageBitmap | undefined;
      try {
        bitmap = await createImageBitmap(blob);
        if (!canvas) {
          canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
          context = canvas.getContext("2d");
        }
        if (canvas.width !== bitmap.width) canvas.width = bitmap.width;
        if (canvas.height !== bitmap.height) canvas.height = bitmap.height;
        // Preserve the old transparent, fresh-canvas pixel path exactly.
        context!.clearRect(0, 0, canvas.width, canvas.height);
        context!.drawImage(bitmap, 0, 0);
        const imageData = context!.getImageData(0, 0, bitmap.width, bitmap.height);
        const result = embedder!.embed(imageData);
        if (result?.embeddings?.[0]?.floatEmbedding) {
          const buf = new Float32Array(result.embeddings[0].floatEmbedding).buffer;
          results.push({ localIdx, embedding: buf });
        }
      } catch {
        // Keep the existing behavior for genuinely unprocessable images.
      } finally { bitmap?.close(); }
    }

    const transferables = results.map((r) => r.embedding);
    self.postMessage({ type: "results", results }, transferables);
  }

  if (type === "detect") {
    const { flatEmbeddings, n, dim, threshold } = data as {
      flatEmbeddings: Float32Array;
      n: number;
      dim: number;
      threshold: number;
      timestamps?: number[];
    };

    // Unpack flat buffer back into array of row views (zero-copy)
    const embeddings: Float32Array[] = [];
    for (let i = 0; i < n; i++) {
      embeddings.push(flatEmbeddings.subarray(i * dim, (i + 1) * dim));
    }

    const groups = await fullCommunityDetection(embeddings, threshold, {
      onProgress: (current, total) => {
        self.postMessage({ type: "detectionProgress", current, total });
      },
    });
    self.postMessage({ type: "detectionResults", groups });
  }

  if (type === "detectSmart") {
    try {
      const { flatEmbeddings, n, dim, threshold, buckets, mediaKeys } = data as {
        flatEmbeddings: Float32Array; n: number; dim: number; threshold: number;
        buckets: number[][]; mediaKeys?: string[];
      };
      const rows = Array.from({ length: n }, (_, i) => flatEmbeddings.subarray(i * dim, (i + 1) * dim));
      const allGroups: number[][] = [];
      let cacheHits = 0, lastProgress = 0;
      const cache = mediaKeys ? await (smartCachePromise ??= SmartMatchCache.open()) : null;
      for (let bi = 0; bi < buckets.length; bi++) {
        const bucket = buckets[bi];
        const bucketRows = bucket.map((index) => rows[index]);
        let key: string | undefined;
        let groups: number[][] | null = null;
        // Tiny buckets are cheaper to compare than to hash and read from IDB.
        if (cache && bucket.length >= 64 && mediaKeys) {
          try {
            key = await cache.fingerprint(bucket.map((index) => mediaKeys[index]), bucketRows, threshold);
            groups = await cache.get(key, bucket.length);
            if (groups) cacheHits++;
          } catch { /* Cache availability never affects detection. */ }
        }
        if (!groups) {
          groups = detectSmartBucket(bucketRows, threshold);
          if (cache && key) await cache.put(key, bucket.length, groups);
        }
        for (const group of groups) allGroups.push(group.map((index) => bucket[index]));
        const now = performance.now();
        if (bi + 1 === buckets.length || now - lastProgress >= 100) {
          self.postMessage({ type: "detectionProgress", current: bi + 1, total: buckets.length });
          lastProgress = now;
        }
      }
      self.postMessage({ type: "detectionResults", groups: allGroups, cacheHits });
    } catch (error) {
      self.postMessage({ type: "detectionError", message: String(error) });
    }
  }
});
