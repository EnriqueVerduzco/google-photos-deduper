import { build } from "esbuild"

const compiled = await build({
  stdin: {
    contents: 'export { detectSmartBucket } from "./lib/smart-matcher.ts"',
    resolveDir: process.cwd()
  },
  bundle: true,
  write: false,
  platform: "node",
  format: "esm"
})
const { detectSmartBucket } = await import(
  `data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text).toString("base64")}`
)
const n = 512,
  dim = 1024,
  threshold = 0.99
let seed = 12345
const rows = Array.from({ length: n }, (_, i) => {
  const row = Float32Array.from({ length: dim }, () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) | 0
    return (seed >>> 0) / 2 ** 32 - 0.5
  })
  const norm = Math.hypot(...row)
  for (let k = 0; k < dim; k++) row[k] /= norm
  return row
})
for (let i = 7; i < n; i += 7) rows[i] = rows[0].slice()
function reference() {
  const parent = rows.map((_, i) => i)
  const find = (i) => (parent[i] === i ? i : (parent[i] = find(parent[i])))
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      let dot = 0
      for (let k = 0; k < dim; k++) dot += rows[i][k] * rows[j][k]
      if (dot >= threshold) parent[find(i)] = find(j)
    }
  const groups = new Map()
  for (let i = 0; i < n; i++) {
    const root = find(i)
    if (!groups.has(root)) groups.set(root, [])
    groups.get(root).push(i)
  }
  return [...groups.values()].filter((group) => group.length >= 2)
}
reference()
detectSmartBucket(rows, threshold)
const before = performance.now(),
  expected = reference(),
  baselineMs = performance.now() - before
const after = performance.now(),
  actual = detectSmartBucket(rows, threshold),
  optimizedMs = performance.now() - after
if (JSON.stringify(expected) !== JSON.stringify(actual))
  throw new Error("Smart results changed")
console.log(
  JSON.stringify(
    {
      n,
      dim,
      threshold,
      baselineMs,
      optimizedMs,
      speedup: baselineMs / optimizedMs,
      identicalResults: true
    },
    null,
    2
  )
)
