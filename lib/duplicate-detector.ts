// Duplicate image detection using MediaPipe Image Embedder.
// Ports the Python DuplicateImageDetector to run in the browser.
//
// Pipeline:
// 1. Fetch thumbnails for media items (skipped for items with cached embeddings)
// 2. Compute L2-normalized embeddings via MediaPipe MobileNet V3
// 3. Group duplicates using fast community detection (cosine similarity)

import type { GpdMediaItem, DuplicateGroup } from "./types";
import { StabilityTracker } from "./scan-log";
import type { ScanLogger } from "./scan-log";
import { prepareEmbeddings } from "./embedding-pipeline";
export { fetchThumbnails } from "./thumbnail-downloads";
export type { FetchThumbnailsResult } from "./thumbnail-downloads";
import { topK } from "./top-k";
import { fullDetectionWorkTotal } from "./full-matcher";
import { runSmartWorkerPool } from "./smart-worker-pool";

export { topK } from "./top-k";

/**
 * Select the best item to keep from a duplicate group.
 * Priority: favorited > original quality > higher resolution > oldest upload date.
 *
 * Favorites outrank every quality heuristic. Starring a photo is an explicit
 * signal from the user, which beats anything we can infer, and silently
 * trashing a favorite is the worst outcome this tool can produce. When a group
 * has several favorites — or none — the remaining criteria decide as before.
 */
export function selectDefaultKeep(items: GpdMediaItem[]): string {
  const qualityScore = (x: GpdMediaItem) =>
    x.isOriginalQuality === true ? 2 : x.isOriginalQuality === false ? 0 : 1;
  const favoriteScore = (x: GpdMediaItem) => (x.isFavorite === true ? 1 : 0);
  const best = [...items].sort((a, b) => {
    const favDiff = favoriteScore(b) - favoriteScore(a);
    if (favDiff !== 0) return favDiff;
    const qDiff = qualityScore(b) - qualityScore(a);
    if (qDiff !== 0) return qDiff;
    const pxDiff =
      (b.resWidth ?? 0) * (b.resHeight ?? 0) -
      (a.resWidth ?? 0) * (a.resHeight ?? 0);
    if (pxDiff !== 0) return pxDiff;
    return (a.creationTimestamp ?? 0) - (b.creationTimestamp ?? 0);
  });
  return best[0].mediaKey;
}

/**
 * Deduplicate media items by dedupKey.
 * There can be photos with different mediaKeys but same dedupKey,
 * likely caused by saving shared albums.
 * We should not solve for them, since
 * - Google Photos natively collapses them into one in timeline/albums
 * - API doesn't allow to trash one of them without trashing others
 * So keep only the first item we see for each dedupKey.
 */
export function dedupeByDedupKey(items: GpdMediaItem[]): GpdMediaItem[] {
  const seen = new Set<string>();
  const uniqueItems: GpdMediaItem[] = [];
  for (const item of items) {
    if (item.dedupKey) {
      if (seen.has(item.dedupKey)) continue;
      seen.add(item.dedupKey);
    }
    uniqueItems.push(item);
  }
  return uniqueItems;
}

export interface DetectionProgress {
  phase:
    | "downloading_thumbnails"
    | "computing_embeddings"
    | "detecting_duplicates";
  current: number;
  total: number;
}

export interface ScanTiming {
  totalItems: number;
  candidates: number;
  cacheHits: number;
  fetchThumbnailsMs: number;
  computeEmbeddingsMs: number;
  communityDetectionMs: number;
  totalMs: number;
}

type ProgressCallback = (progress: DetectionProgress) => void;

// ============================================================
// Main entry points
// ============================================================

export async function fullDetectDuplicates(
  mediaItems: GpdMediaItem[],
  threshold: number,
  onProgress?: ProgressCallback,
  signal?: AbortSignal,
  logger?: ScanLogger,
): Promise<{ groups: DuplicateGroup[]; timing: ScanTiming }> {
  const scanStart = performance.now();

  const dedupedItems = dedupeByDedupKey(mediaItems);

  // Filter to items with thumbnails (photos only, skip videos)
  // Include items with thumbnails. Video posters work too — two copies of the
  // same clip have identical poster frames, which produce near-identical
  // embeddings.
  const candidates = dedupedItems.filter((item) => item.thumb);
  console.log(
    `[GPD] detectDuplicates: ${mediaItems.length} items → ${candidates.length} candidates`,
  );

  const emptyTiming = (extra: Partial<ScanTiming> = {}): ScanTiming => ({
    totalItems: mediaItems.length,
    candidates: candidates.length,
    cacheHits: 0,
    fetchThumbnailsMs: 0,
    computeEmbeddingsMs: 0,
    communityDetectionMs: 0,
    totalMs: Math.round(performance.now() - scanStart),
    ...extra,
  });

  if (candidates.length < 2) return { groups: [], timing: emptyTiming() };

  const stabilityTracker = new StabilityTracker((estimate) => logger?.recordStableEstimate(estimate));
  const trackedProgress: ProgressCallback = (progress) => {
    onProgress?.(progress);
    stabilityTracker.update(progress.phase, progress.current, progress.total);
  };
  const { embeddings, validIndices, cacheHits, fetchThumbnailsMs, computeEmbeddingsMs } =
    await prepareEmbeddings(candidates, trackedProgress, signal, logger);

  if (embeddings.length < 2) {
    return {
      groups: [],
      timing: emptyTiming({ cacheHits, fetchThumbnailsMs, computeEmbeddingsMs }),
    };
  }

  // Step 3: Community detection — runs in a worker to keep UI responsive.
  // The setTimeout(0) yield lets React flush the phase change to "detecting_duplicates"
  // before the worker is dispatched, so the UI updates before the long computation begins.
  // Progress updates come from the worker during detection.
  trackedProgress({
    phase: "detecting_duplicates",
    current: 0,
    total: fullDetectionWorkTotal(embeddings.length),
  });
  await new Promise<void>((r) => setTimeout(r, 0));
  const workerUrl = chrome.runtime.getURL("scripts/embedder-worker.js");
  const timestamps = validIndices.map(
    (i) => candidates[i].creationTimestamp ?? 0,
  );
  const t3 = performance.now();
  const indexGroups = await runCommunityDetectionInWorker(
    embeddings,
    threshold,
    timestamps,
    workerUrl,
    trackedProgress,
    signal,
  );
  const communityDetectionMs = Math.round(performance.now() - t3);
  const totalMs = Math.round(performance.now() - scanStart);
  console.log(
    `[GPD] communityDetection: ${indexGroups.length} groups in ${communityDetectionMs}ms`,
  );
  console.log(`[GPD] scan complete: ${totalMs}ms total`);
  await logger?.phaseComplete("communityDetectionMs", communityDetectionMs);

  // Map indices back to media items and build DuplicateGroup objects
  const groups: DuplicateGroup[] = indexGroups.map((indices, i) => {
    // Sort items by upload date ascending so the oldest is first
    const items = indices
      .map((idx) => candidates[validIndices[idx]])
      .sort((a, b) => (a.creationTimestamp ?? 0) - (b.creationTimestamp ?? 0));
    const mediaKeys = items.map((item) => item.mediaKey);
    return {
      id: `group-${i}`,
      mediaKeys,
      originalMediaKey: selectDefaultKeep(items),
      similarity: threshold, // Approximate; all items are at least this similar
    };
  });

  const timing: ScanTiming = {
    totalItems: mediaItems.length,
    candidates: candidates.length,
    cacheHits,
    fetchThumbnailsMs,
    computeEmbeddingsMs,
    communityDetectionMs,
    totalMs,
  };

  return { groups, timing };
}

// ============================================================
// Smart scan: group by timestamp, embed subset, pairwise union-find
// ============================================================

/**
 * Group media items by their `timestamp` field (EXIF taken date).
 * Only returns buckets with ≥ 2 items.
 *
 * windowMs = 0 (default): exact timestamp match.
 * windowMs > 0: items within the same time window are bucketed together.
 *
 * Exported for unit testing.
 */
export function groupByTimestamp(
  items: GpdMediaItem[],
  windowMs = 0,
): GpdMediaItem[][] {
  const buckets = new Map<number, GpdMediaItem[]>();
  for (const item of items) {
    // Guard degenerate timestamps. `undefined` floors to NaN and Map treats
    // every NaN as the same key; `null` floors to 0 and silently joins the
    // epoch bucket. Either way every affected item collapses into one bucket,
    // whose pairwise comparison is quadratic in the size of that collapse.
    // Such items can't be time-matched anyway, so drop them from bucketing.
    if (!Number.isFinite(item.timestamp)) continue;
    const key =
      windowMs > 0
        ? Math.floor(item.timestamp / windowMs) * windowMs
        : item.timestamp;
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key)!.push(item);
  }
  return [...buckets.values()].filter((g) => g.length >= 2);
}

/**
 * Largest timestamp bucket we will compare pairwise in one piece.
 *
 * Comparison cost is quadratic in bucket size, so a single dense bucket can
 * dominate an entire scan: at the measured worker throughput a 5,000-item
 * bucket takes roughly 20 seconds, while a 50,000-item one takes over half an
 * hour. Wide time windows make this easy to hit — one busy hour or a bulk
 * import can drop thousands of photos into the same bucket.
 */
export const MAX_BUCKET_SIZE = 5000;

/**
 * Split buckets larger than `cap` into smaller, time-contiguous chunks so no
 * single pairwise comparison runs unbounded.
 *
 * Chunks are near-equal rather than cap-sized — splitting 5,001 items at a cap
 * of 5,000 yields 2,501 + 2,500, not 5,000 + 1. Items are sorted by timestamp
 * first so each chunk covers a contiguous slice of time, which is where real
 * duplicates cluster.
 *
 * This trades recall for a bounded runtime: duplicates that straddle a chunk
 * boundary are not detected. `bucketsSplit` counts how many source buckets
 * were affected so the UI can say so.
 */
export function splitOversizedBuckets(
  buckets: GpdMediaItem[][],
  cap = MAX_BUCKET_SIZE,
): { buckets: GpdMediaItem[][]; bucketsSplit: number } {
  const out: GpdMediaItem[][] = [];
  let bucketsSplit = 0;

  for (const bucket of buckets) {
    if (bucket.length <= cap) {
      out.push(bucket);
      continue;
    }
    bucketsSplit++;

    const sorted = [...bucket].sort((a, b) => a.timestamp - b.timestamp);
    const chunkCount = Math.ceil(sorted.length / cap);
    const chunkSize = Math.ceil(sorted.length / chunkCount);
    for (let i = 0; i < sorted.length; i += chunkSize)
      out.push(sorted.slice(i, i + chunkSize));
  }

  return { buckets: out, bucketsSplit };
}

/**
 * Given a single timestamp-bucket group, compute duplicate groups using
 * pairwise cosine similarity + union-find.
 *
 * Exported for unit testing. The worker contains an equivalent inline copy.
 */
export function withinGroupDuplicates(
  groupItems: GpdMediaItem[],
  embeddingMap: Map<string, Float32Array>,
  threshold: number,
  groupIdOffset: number,
): DuplicateGroup[] {
  const withEmb = groupItems
    .map((item) => ({ item, emb: embeddingMap.get(item.mediaKey) }))
    .filter(
      (x): x is { item: GpdMediaItem; emb: Float32Array } => !!x.emb,
    );
  if (withEmb.length < 2) return [];

  // Union-Find
  const parent = withEmb.map((_, i) => i);
  const find = (x: number): number =>
    parent[x] === x ? x : (parent[x] = find(parent[x]));
  const union = (a: number, b: number) => {
    parent[find(a)] = find(b);
  };

  const dim = withEmb[0].emb.length;
  for (let i = 0; i < withEmb.length; i++) {
    for (let j = i + 1; j < withEmb.length; j++) {
      let dot = 0;
      for (let k = 0; k < dim; k++) dot += withEmb[i].emb[k] * withEmb[j].emb[k];
      if (dot >= threshold) union(i, j);
    }
  }

  const components = new Map<number, GpdMediaItem[]>();
  for (let i = 0; i < withEmb.length; i++) {
    const root = find(i);
    if (!components.has(root)) components.set(root, []);
    components.get(root)!.push(withEmb[i].item);
  }

  return [...components.values()]
    .filter((g) => g.length >= 2)
    .map((items, i) => {
      const sorted = [...items].sort(
        (a, b) => (a.creationTimestamp ?? 0) - (b.creationTimestamp ?? 0),
      );
      return {
        id: `group-${groupIdOffset + i}`,
        mediaKeys: sorted.map((x) => x.mediaKey),
        originalMediaKey: selectDefaultKeep(items),
        similarity: threshold,
      };
    });
}

export async function smartDetectDuplicates(
  mediaItems: GpdMediaItem[],
  threshold: number,
  windowMs = 1000,
  onProgress?: ProgressCallback,
  signal?: AbortSignal,
  logger?: ScanLogger,
): Promise<{ groups: DuplicateGroup[]; bucketsSplit: number }> {
  const scanStart = performance.now();

  const dedupedItems = dedupeByDedupKey(mediaItems);

  // Include items with thumbnails. Video posters work too — two copies of the
  // same clip have identical poster frames, which produce near-identical
  // embeddings.
  const candidates = dedupedItems.filter((item) => item.thumb);

  // Step 1: Bucket by timestamp — no I/O, instant
  const rawBuckets = groupByTimestamp(candidates, windowMs);

  // Step 1b: Cap bucket size. Pairwise cost is quadratic within a bucket, so
  // one dense hour can otherwise stall the whole scan.
  const { buckets, bucketsSplit } = splitOversizedBuckets(rawBuckets);
  const largest = buckets.reduce((max, b) => Math.max(max, b.length), 0);
  console.log(
    `[GPD] smartDetectDuplicates: ${mediaItems.length} items → ${candidates.length} candidates → ${buckets.length} timestamp buckets (largest ${largest}${bucketsSplit > 0 ? `, ${bucketsSplit} split at cap ${MAX_BUCKET_SIZE}` : ""})`,
  );
  if (buckets.length === 0) return { groups: [], bucketsSplit };

  // Flatten to deduplicated subset
  const seen = new Set<string>();
  const subset: GpdMediaItem[] = [];
  for (const bucket of buckets)
    for (const item of bucket)
      if (!seen.has(item.mediaKey)) {
        seen.add(item.mediaKey);
        subset.push(item);
      }

  const stabilityTracker = new StabilityTracker((estimate) => logger?.recordStableEstimate(estimate));
  const trackedProgress: ProgressCallback = (progress) => {
    onProgress?.(progress);
    stabilityTracker.update(progress.phase, progress.current, progress.total);
  };
  const { embeddings, validIndices } = await prepareEmbeddings(subset, trackedProgress, signal, logger);

  if (embeddings.length < 2) return { groups: [], bucketsSplit };

  // Build bucket index arrays (indices into embeddings[])
  const mediaKeyToEmbIdx = new Map<string, number>();
  for (let i = 0; i < validIndices.length; i++)
    mediaKeyToEmbIdx.set(subset[validIndices[i]].mediaKey, i);

  const workerBuckets = buckets
    .map((bucket) =>
      bucket
        .map((item) => mediaKeyToEmbIdx.get(item.mediaKey))
        .filter((i): i is number => i !== undefined),
    )
    .filter((b) => b.length >= 2);

  if (workerBuckets.length === 0) return { groups: [], bucketsSplit };

  // Offload pairwise comparison to worker
  trackedProgress({ phase: "detecting_duplicates", current: 0, total: 0 });
  await new Promise<void>((r) => setTimeout(r, 0)); // flush React phase update
  const workerUrl = chrome.runtime.getURL("scripts/embedder-worker.js");
  const t3 = performance.now();
  const indexGroups = await runSmartWorkerPool(
    embeddings,
    validIndices.map((index) => subset[index].mediaKey),
    threshold,
    workerBuckets,
    workerUrl,
    trackedProgress,
    signal,
  );
  const communityDetectionMs = Math.round(performance.now() - t3);
  const totalMs = Math.round(performance.now() - scanStart);
  console.log(
    `[GPD] communityDetection: ${indexGroups.length} groups in ${communityDetectionMs}ms`,
  );
  console.log(`[GPD] scan complete: ${totalMs}ms total`);
  await logger?.phaseComplete("communityDetectionMs", communityDetectionMs);

  // Map indices back to GpdMediaItems
  const groups: DuplicateGroup[] = indexGroups.map((indices, i) => {
    const items = indices
      .map((idx) => subset[validIndices[idx]])
      .sort((a, b) => (a.creationTimestamp ?? 0) - (b.creationTimestamp ?? 0));
    return {
      id: `group-${i}`,
      mediaKeys: items.map((x) => x.mediaKey),
      originalMediaKey: selectDefaultKeep(items),
      similarity: threshold,
    };
  });

  return {
    groups: groups.sort((a, b) => b.mediaKeys.length - a.mediaKeys.length),
    bucketsSplit,
  };
}

// ============================================================
// Step 3: Community detection — worker wrapper
// ============================================================

/**
 * Runs communityDetection in a dedicated worker so the main thread stays
 * responsive during the computation.
 *
 * Embeddings are packed into a single transferable Float32Array and sent
 * to the worker. The worker returns number[][] (the group index lists).
 */
export async function runCommunityDetectionInWorker(
  embeddings: Float32Array[],
  threshold: number,
  timestamps: number[],
  workerUrl: string,
  onProgress?: ProgressCallback,
  signal?: AbortSignal,
): Promise<number[][]> {
  signal?.throwIfAborted();
  const n = embeddings.length;
  if (n === 0) return [];
  const dim = embeddings[0].length;

  // Pack all embedding rows into one flat buffer for zero-copy transfer
  const flat = new Float32Array(n * dim);
  for (let i = 0; i < n; i++) flat.set(embeddings[i], i * dim);

  const worker = new Worker(workerUrl);

  return new Promise<number[][]>((resolve, reject) => {
    if (signal?.aborted) {
      worker.terminate();
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }

    const onAbort = () => {
      worker.terminate();
      reject(new DOMException("Aborted", "AbortError"));
    };
    signal?.addEventListener("abort", onAbort, { once: true });

    worker.onmessage = (e) => {
      if (e.data.type === "detectionProgress") {
        onProgress?.({
          phase: "detecting_duplicates",
          current: e.data.current,
          total: e.data.total,
        });
      } else if (e.data.type === "detectionResults") {
        signal?.removeEventListener("abort", onAbort);
        worker.terminate();
        resolve(e.data.groups as number[][]);
      }
    };
    worker.onerror = (e) => {
      signal?.removeEventListener("abort", onAbort);
      worker.terminate();
      reject(new Error(e.message ?? "Worker error during community detection"));
    };

    worker.postMessage(
      {
        type: "detect",
        data: { flatEmbeddings: flat, n, dim, threshold, timestamps },
      },
      [flat.buffer],
    );
  });
}

// ============================================================
// Community detection
// Sorts by timestamp, then walks through comparing each photo
// to the next. Consecutive similar photos are grouped together.
// O(n) instead of O(n²).
// ============================================================

/**
 * Find groups of similar embeddings by sorting on timestamp and
 * comparing each photo to the next one in the sorted list.
 * Returns groups of original indices.
 *
 * Exported for unit testing.
 */
export function communityDetection(
  embeddings: Float32Array[],
  threshold: number,
  timestamps?: number[],
): number[][] {
  const n = embeddings.length;
  if (n < 2) return [];
  const dim = embeddings[0].length;

  // Sort by timestamp so nearby photos are adjacent
  const order = Array.from({ length: n }, (_, i) => i);
  if (timestamps) {
    order.sort((a, b) => (timestamps[a] ?? 0) - (timestamps[b] ?? 0));
  }

  const sorted: Float32Array[] = order.map((i) => embeddings[i]);
  const groups: number[][] = [];
  let currentGroup: number[] = [order[0]];

  for (let i = 0; i < n - 1; i++) {
    // Cosine similarity = dot product (embeddings are L2-normalized)
    const a = sorted[i];
    const b = sorted[i + 1];
    let dot = 0;
    for (let k = 0; k < dim; k++) dot += a[k] * b[k];

    if (dot >= threshold) {
      currentGroup.push(order[i + 1]);
    } else {
      if (currentGroup.length >= 2) groups.push(currentGroup);
      currentGroup = [order[i + 1]];
    }
  }

  if (currentGroup.length >= 2) groups.push(currentGroup);

  groups.sort((a, b) => b.length - a.length);
  return groups;
}

// ============================================================
// Linear algebra helpers (typed arrays, no external deps)
// ============================================================

/**
 * Matrix multiplication: A[startA:endA] x B[startB:endB]^T
 * A and B are arrays of Float32Array (rows).
 * Returns a flat Float32Array of shape [endA-startA, endB-startB].
 *
 * Exported for unit testing.
 */
export function matMul(
  A: Float32Array[],
  startA: number,
  endA: number,
  B: Float32Array[],
  startB: number,
  endB: number,
  dim: number,
): Float32Array {
  const rowsA = endA - startA;
  const rowsB = endB - startB;
  const result = new Float32Array(rowsA * rowsB);

  for (let i = 0; i < rowsA; i++) {
    const aRow = A[startA + i];
    for (let j = 0; j < rowsB; j++) {
      const bRow = B[startB + j];
      let dot = 0;
      for (let k = 0; k < dim; k++) {
        dot += aRow[k] * bRow[k];
      }
      result[i * rowsB + j] = dot;
    }
  }

  return result;
}
