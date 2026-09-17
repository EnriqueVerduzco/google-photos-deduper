// Deterministic, offline comparison in separate processes for memory isolation.
// Usage: node tools/benchmark-full-matcher.mjs [--n=4096] [--dim=1024]
//        [--threshold=0.99] [--repeats=3]
import assert from "node:assert/strict"
import { spawnSync } from "node:child_process"
import { mkdtemp, rm } from "node:fs/promises"
import { cpus, tmpdir } from "node:os"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { build } from "esbuild"

const root = fileURLToPath(new URL("../", import.meta.url))
const options = { n: 4096, dim: 1024, threshold: 0.99, repeats: 3 }
for (const arg of process.argv.slice(2)) {
  const match = /^--(n|dim|threshold|repeats)=(.+)$/.exec(arg)
  assert(match, `Unknown argument: ${arg}`)
  options[match[1]] = Number(match[2])
}
for (const key of ["n", "dim", "repeats"]) {
  assert(
    Number.isSafeInteger(options[key]) && options[key] > 0,
    `${key} must be a positive integer`
  )
}
assert(
  Number.isFinite(options.threshold) &&
    options.threshold > 0 &&
    options.threshold <= 1
)
const temporary = await mkdtemp(path.join(tmpdir(), "gpd-full-benchmark-"))
try {
  const entry = path.join(temporary, "benchmark.cjs")
  await build({
    absWorkingDir: root,
    entryPoints: ["tests/perf/full-matcher-benchmark.ts"],
    outfile: entry,
    bundle: true,
    platform: "node",
    format: "cjs",
    target: "node22"
  })
  const runs = []
  let digest
  for (let repeat = 0; repeat < options.repeats; repeat++) {
    // Alternate order to reduce systematic thermal/load bias.
    for (const mode of repeat % 2
      ? ["triangle", "directed"]
      : ["directed", "triangle"]) {
      const result = spawnSync(
        process.execPath,
        [
          "--expose-gc",
          entry,
          mode,
          String(options.n),
          String(options.dim),
          String(options.threshold)
        ],
        {
          encoding: "utf8",
          cwd: root
        }
      )
      assert.equal(result.status, 0, result.stderr)
      const run = JSON.parse(result.stdout)
      digest ??= run.groupDigest
      assert.equal(run.groupDigest, digest, "Group membership/order differs!")
      runs.push({ repeat: repeat + 1, ...run })
      console.error(`${mode} run ${repeat + 1}: ${run.elapsedMs.toFixed(1)}ms`)
    }
  }
  const median = (values) =>
    values.sort((a, b) => a - b)[Math.floor(values.length / 2)]
  const directedMedianMs = median(
    runs.filter((r) => r.mode === "directed").map((r) => r.elapsedMs)
  )
  const triangleMedianMs = median(
    runs.filter((r) => r.mode === "triangle").map((r) => r.elapsedMs)
  )
  console.log(
    JSON.stringify(
      {
        description:
          "Synthetic offline matcher benchmark; not a real Google Photos scan",
        node: process.version,
        platform: `${process.platform}/${process.arch}`,
        cpu: cpus()[0]?.model,
        options,
        directedMedianMs,
        triangleMedianMs,
        syntheticMatcherSpeedup: directedMedianMs / triangleMedianMs,
        memoryNotes:
          "RSS is whole-process high-water memory; heap/ArrayBuffer peaks are sampled at progress callbacks. Neither is an exact matcher-only peak. JS sparse-array capacity and object overhead are runtime-dependent. Embeddings are shared row references, not copied per core.",
        runs
      },
      null,
      2
    )
  )
} finally {
  await rm(temporary, { recursive: true, force: true })
}
