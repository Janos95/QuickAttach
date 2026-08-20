import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const siteRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const modelRoot = path.join(siteRoot, "public", "model");

test("CAD inspector exposes source and collision mappings for every part", async () => {
  const manifest = JSON.parse(
    await readFile(path.join(modelRoot, "manifest.json"), "utf8"),
  );

  assert.equal(manifest.cadAssets.length, 19);
  assert.equal(new Set(manifest.cadAssets.map(({ id }) => id)).size, 19);

  for (const part of manifest.cadAssets) {
    assert.ok(part.id);
    assert.ok(part.label);
    assert.ok(part.category);
    assert.ok(part.sourceFile);
    assert.ok(part.bodyRoots.length > 0);
    assert.ok(part.geomPrefixes.length > 0);
    await access(path.join(modelRoot, part.asset));
  }
});
