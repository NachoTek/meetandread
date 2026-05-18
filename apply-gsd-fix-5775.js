#!/usr/bin/env node

/**
 * GSD Fix for Issue #5775: triage-captures and quick-task missing from UnitContextManifest
 *
 * This script applies the fix from upstream PR #6224 to your local GSD installation.
 * It patches two files:
 * 1. unit-context-manifest.js - adds triage-captures and quick-task entries
 * 2. phases.js - adds sidecar prefix normalization to unitWritesSource
 */

import { readFileSync, writeFileSync, existsSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const GSD_MODULE_PATH = 'C:/Users/david.keymel/AppData/Local/nvm/v22.22.2/node_modules/gsd-pi/dist/resources/extensions/gsd';

const MANIFEST_FILE = join(GSD_MODULE_PATH, 'unit-context-manifest.js');
const PHASES_FILE = join(GSD_MODULE_PATH, 'auto/phases.js');

console.log('GSD Fix for Issue #5775\n');
console.log('This patch will:');
console.log('1. Add triage-captures and quick-task to KNOWN_UNIT_TYPES');
console.log('2. Add manifest entries for both unit types');
console.log('3. Add sidecar prefix normalization to unitWritesSource()\n');

// Verify files exist
if (!existsSync(MANIFEST_FILE)) {
  console.error(`Error: File not found: ${MANIFEST_FILE}`);
  console.error('Please check your GSD installation path.');
  process.exit(1);
}

if (!existsSync(PHASES_FILE)) {
  console.error(`Error: File not found: ${PHASES_FILE}`);
  console.error('Please check your GSD installation path.');
  process.exit(1);
}

// Read files
let manifestContent = readFileSync(MANIFEST_FILE, 'utf-8');
let phasesContent = readFileSync(PHASES_FILE, 'utf-8');

console.log(`Reading: ${MANIFEST_FILE}`);
console.log(`Reading: ${PHASES_FILE}\n`);

// Patch 1: Add triage-captures and quick-task to KNOWN_UNIT_TYPES
const knownTypesMarker = '  "rewrite-docs",';
const knownTypesAddition = `  "rewrite-docs",
  // Sidecar units (triage, quick-task)
  "triage-captures",
  "quick-task",`;

if (!manifestContent.includes('triage-captures')) {
  manifestContent = manifestContent.replace(knownTypesMarker, knownTypesAddition);
  console.log('✓ Added triage-captures and quick-task to KNOWN_UNIT_TYPES');
} else {
  console.log('✓ triage-captures and quick-task already in KNOWN_UNIT_TYPES (skipping)');
}

// Patch 2: Add manifest entries at the end of UNIT_MANIFESTS
const manifestEntries = `  "triage-captures": {
    skills: { mode: "all" },
    knowledge: "scoped",
    memory: "prompt-relevant",
    codebaseMap: false,
    preferences: "active-only",
    contextMode: "triage",
    tools: TOOLS_PLANNING,
    artifacts: {
      inline: ["roadmap", "slice-plan", "slice-summary", "requirements", "decisions", "templates"],
      excerpt: [],
      onDemand: [],
    },
    maxSystemPromptChars: COMMON_BUDGET_MEDIUM,
  },
  "quick-task": {
    skills: { mode: "all" },
    knowledge: "full",
    memory: "prompt-relevant",
    codebaseMap: true,
    preferences: "active-only",
    contextMode: "execution",
    tools: TOOLS_ALL,
    artifacts: {
      inline: ["roadmap", "slice-plan", "task-plan", "requirements", "decisions", "templates"],
      excerpt: [],
      onDemand: [],
    },
    maxSystemPromptChars: COMMON_BUDGET_MEDIUM,
  },`;

// Find the closing of UNIT_MANIFESTS object (look for "};\n\nexport function")
const manifestsEndPattern = /};\n\nexport function/;
const match = manifestsEndPattern.exec(manifestContent);

if (match && !manifestContent.includes('"triage-captures":')) {
  const insertPosition = match.index;
  manifestContent = manifestContent.slice(0, insertPosition) +
    '\n' + manifestEntries + '\n' +
    manifestContent.slice(insertPosition);
  console.log('✓ Added manifest entries for triage-captures and quick-task');
} else if (manifestContent.includes('"triage-captures":')) {
  console.log('✓ Manifest entries already present (skipping)');
} else {
  console.error('Error: Could not find UNIT_MANIFESTS closing brace to insert entries');
  process.exit(1);
}

// Patch 3: Add sidecar prefix normalization to unitWritesSource
const oldFunction = /function unitWritesSource\(unitType\) \{\s*const manifest = resolveManifest\(unitType\);\s*if \(!manifest\) return null;\s*return manifest\.tools\.mode === "all" \|\| manifest\.tools\.mode === "docs";\s*\}/;

const newFunction = `function unitWritesSource(unitType) {
  // Backward compatibility: sidecar queues from older builds may persist
  // prefixed unit types (e.g. "sidecar/quick-task").
  const normalizedUnitType = unitType.startsWith("sidecar/")
    ? unitType.slice("sidecar/".length)
    : unitType;
  const manifest = resolveManifest(normalizedUnitType);
  if (!manifest) return null;
  return manifest.tools.mode === "all" || manifest.tools.mode === "docs";
}`;

if (oldFunction.test(phasesContent)) {
  phasesContent = phasesContent.replace(oldFunction, newFunction);
  console.log('✓ Added sidecar prefix normalization to unitWritesSource()');
} else if (phasesContent.includes('normalizedUnitType')) {
  console.log('✓ Sidecar prefix normalization already present (skipping)');
} else {
  console.error('Error: Could not find unitWritesSource function to patch');
  process.exit(1);
}

// Write patched files
writeFileSync(MANIFEST_FILE, manifestContent, 'utf-8');
writeFileSync(PHASES_FILE, phasesContent, 'utf-8');

console.log('\n---');
console.log('Successfully patched GSD for Issue #5775');
console.log(`Modified: ${MANIFEST_FILE}`);
console.log(`Modified: ${PHASES_FILE}\n`);
console.log('The fix will take effect on the next GSD session.');
console.log('No restart required - patches are applied to the installed extension.\n');
console.log('Note: These changes will be overwritten when you update GSD via npm.');
console.log('This is a temporary workaround until the fix is released in a new version.');