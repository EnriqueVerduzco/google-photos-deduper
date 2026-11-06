import { spawnSync } from "node:child_process"
import { fileURLToPath } from "node:url"

const toolkit = fileURLToPath(new URL("../Google-Photos-Toolkit/", import.meta.url))
const patch = fileURLToPath(new URL("../patches/google-photos-toolkit-api.patch", import.meta.url))
const apply = (...args) => spawnSync("git", ["apply", ...args, patch], {
  cwd: toolkit, encoding: "utf8"
})

if (apply("--reverse", "--check").status === 0) {
  console.log("Google Photos Toolkit fork patch is already applied.")
} else {
  const check = apply("--check")
  if (check.status !== 0) {
    throw new Error(`Toolkit patch conflicts with local changes; resolve them before building.\n${check.stderr}`)
  }
  const result = apply()
  if (result.status !== 0) throw new Error(result.stderr)
  console.log("Applied Google Photos Toolkit fork patch.")
}
