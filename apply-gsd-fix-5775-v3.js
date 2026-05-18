#!/usr/bin/env node

/**
 * GSD Fix for Issue #5775: triage-captures and quick-task missing from UnitContextManifest
 * v3 - Direct string replacement approach
 */

import { readFileSync, writeFileSync, existsSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const GSD_MODULE_PATH = 'C:/Users/david.keymel/AppData/Local/nvm/v22.22.2/node_modules/gsd-pi/dist/resources/extensions/gsd';

const MANIFEST_FILE = join(GSD_MODULE_PATH, 'unit-context-manifest.js');
const PHASES_FILE = join(GSD_MODULE_PATH, 'auto/phases.js');

console.log('GSD Fix for Issue #5775 (v3)\n');

if (!existsSync(MANIFEST_FILE)) {
  console.error(`Error: File not found: ${MANIFEST_FILE}`);
  process.exit(1);
}

if (!existsSync(PHASES_FILE)) {
  console.error(`Error: File not found: ${PHASES_FILE}`);
  process.exit(1);
}

let manifestContent = readFileSync(MANIFEST_FILE, 'utf-8');
let phasesContent = readFileSync(PHASES_FILE, 'utf-8');

// Patch 1: Add to KNOWN_UNIT_TYPES array
const oldKnownTypes = `    "rewrite-docs",
    // Deep planning mode (project-level) units
    "workflow-preferences",`;

const newKnownTypes = `    "rewrite-docs",
    // Sidecar units (triage, quick-task)
    "triage-captures",
    "quick-task",
    // Deep planning mode (project-level) units
    "workflow-preferences",`;

if (!manifestContent.includes('"triage-captures"')) {
  manifestContent = manifestContent.replace(oldKnownTypes, newKnownTypes);
  console.log('✓ Added triage-captures and quick-task to KNOWN_UNIT_TYPES');
} else {
  console.log('✓ triage-captures already in KNOWN_UNIT_TYPES (skipping)');
}

// Patch 2: Add manifest entries before the closing brace of UNIT_MANIFESTS
const oldClosing = `    },
};
// ─── Lookup helper ────────────────────────────────────────────────────────`;

const newManifests = `    },
    "triage-captures": {
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
    },
};
// ─── Lookup helper ────────────────────────────────────────────────────────`;

if (!manifestContent.includes('"triage-captures":')) {
  manifestContent = manifestContent.replace(oldClosing, newManifests);
  console.log('✓ Added manifest entries for triage-captures and quick-task');
} else {
  console.log('✓ Manifest entries already present (skipping)');
}

// Patch 3: Add sidecar prefix normalization
const oldUnitWritesSource = `function unitWritesSource(unitType) {
    const manifest = resolveManifest(unitType);
    if (!manifest) return null;
    return manifest.tools.mode === "all" || manifest.tools.mode === "docs";
}`;

const newUnitWritesSource = `function unitWritesSource(unitType) {
    // Backward compatibility: sidecar queues from older builds may persist
    // prefixed unit types (e.g. "sidecar/quick-task").
    const normalizedUnitType = unitType.startsWith("sidecar/")
        ? unitType.slice("sidecar/".length)
        : unitType;
    const manifest = resolveManifest(normalizedUnitType);
    if (!manifest) return null;
    return manifest.tools.mode === "all" || manifest.tools.mode === "docs";
}`;

if (!phasesContent.includes('normalizedUnitType')) {
  phasesContent = phasesContent.replace(oldUnitWritesSource, newUnitWritesSource);
  console.log('✓ Added sidecar prefix normalization to unitWritesSource()');
} else {
  console.log('✓ Sidecar prefix normalization already present (skipping)');
}

writeFileSync(MANIFEST_FILE, manifestContent, 'utf-8');
writeFileSync(PHASES_FILE, phasesContent, 'utf-8');

console.log('\n---');
console.log('Successfully patched GSD for Issue #5775');
console.log('The fix will take effect on the next GSD session.');
console.log('\nNote: These changes will be overwritten when you update GSD via npm.');