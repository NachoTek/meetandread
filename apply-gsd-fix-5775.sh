#!/bin/bash

# GSD Fix for Issue #5775: triage-captures and quick-task missing from UnitContextManifest
# Bash version for manual application

MANIFEST_FILE="/c/Users/david.keymel/AppData/Local/nvm/v22.22.2/node_modules/gsd-pi/dist/resources/extensions/gsd/unit-context-manifest.js"
PHASES_FILE="/c/Users/david.keymel/AppData/Local/nvm/v22.22.2/node_modules/gsd-pi/dist/resources/extensions/gsd/auto/phases.js"

echo "GSD Fix for Issue #5775"
echo "========================="
echo ""

# Check if files exist
if [ ! -f "$MANIFEST_FILE" ]; then
    echo "Error: File not found: $MANIFEST_FILE"
    exit 1
fi

if [ ! -f "$PHASES_FILE" ]; then
    echo "Error: File not found: $PHASES_FILE"
    exit 1
fi

echo "Patching $MANIFEST_FILE..."
echo "Patching $PHASES_FILE..."
echo ""

# Patch 1: Add triage-captures and quick-task to KNOWN_UNIT_TYPES
if grep -q '"triage-captures"' "$MANIFEST_FILE"; then
    echo "✓ triage-captures already in KNOWN_UNIT_TYPES (skipping)"
else
    # Add after "rewrite-docs",
    sed -i '/"rewrite-docs",/a\  // Sidecar units (triage, quick-task)\n  "triage-captures",\n  "quick-task",' "$MANIFEST_FILE"
    echo "✓ Added triage-captures and quick-task to KNOWN_UNIT_TYPES"
fi

# Patch 2: Add manifest entries
if grep -q '"triage-captures": {' "$MANIFEST_FILE"; then
    echo "✓ Manifest entries already present (skipping)"
else
    # Add before the closing brace of UNIT_MANIFESTS
    cat >> "$MANIFEST_FILE" << 'MANIFEST_ENTRIES'
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
MANIFEST_ENTRIES
    echo "✓ Added manifest entries for triage-captures and quick-task"
fi

# Patch 3: Add sidecar prefix normalization
if grep -q 'normalizedUnitType' "$PHASES_FILE"; then
    echo "✓ Sidecar prefix normalization already present (skipping)"
else
    # Replace the unitWritesSource function
    sed -i '/function unitWritesSource(unitType)/,/return manifest\.tools\.mode === "all" \|\| manifest\.tools\.mode === "docs";/c\
function unitWritesSource(unitType) {\
  // Backward compatibility: sidecar queues from older builds may persist\
  // prefixed unit types (e.g. "sidecar/quick-task").\
  const normalizedUnitType = unitType.startsWith("sidecar/")\
    ? unitType.slice("sidecar/".length)\
    : unitType;\
  const manifest = resolveManifest(normalizedUnitType);\
  if (!manifest) return null;\
  return manifest.tools.mode === "all" || manifest.tools.mode === "docs";\
}' "$PHASES_FILE"
    echo "✓ Added sidecar prefix normalization to unitWritesSource()"
fi

echo ""
echo "---"
echo "Successfully patched GSD for Issue #5775"
echo ""
echo "Note: These changes will be overwritten when you update GSD via npm."
echo "This is a temporary workaround until the fix is released in a new version."