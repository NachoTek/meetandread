# GSD Issue #5775 Fix Applied

## What Was Fixed

The upstream bug causing `Worktree Safety failed (missing-tool-contract): missing Tool Contract for triage-captures` has been patched in your local installation.

## Changes Made

### 1. File: `unit-context-manifest.js`
**Location:** `C:\Users\david.keymel\AppData\Local\nvm\v22.22.2\node_modules\gsd-pi\dist\resources\extensions\gsd\unit-context-manifest.js`

#### Added to KNOWN_UNIT_TYPES array:
```javascript
// Sidecar units (triage, quick-task)
"triage-captures",
"quick-task",
```

#### Added manifest entries:
```javascript
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
```

### 2. File: `phases.js`
**Location:** `C:\Users\david.keymel\AppData\Local\nvm\v22.22.2\node_modules\gsd-pi\dist\resources\extensions\gsd\auto\phases.js`

#### Updated `unitWritesSource` function with sidecar prefix normalization:
```javascript
function unitWritesSource(unitType) {
    // Backward compatibility: sidecar queues from older builds may persist
    // prefixed unit types (e.g. "sidecar/quick-task").
    const normalizedUnitType = unitType.startsWith("sidecar/")
        ? unitType.slice("sidecar/".length)
        : unitType;
    const manifest = resolveManifest(normalizedUnitType);
    if (!manifest)
        return null;
    return manifest.tools.mode === "all" || manifest.tools.mode === "docs";
}
```

## What This Fixes

- **triage-captures** sidecar units can now be dispatched without hitting the missing-tool-contract guard
- **quick-task** units can also be dispatched properly
- Sidecar-prefixed unit types (e.g., "sidecar/triage-captures") are now normalized before manifest lookup

## Verification

You can verify the patches by:

1. Checking KNOWN_UNIT_TYPES includes the new units:
```bash
grep "triage-captures" "C:\Users\david.keymel\AppData\Local\nvm\v22.22.2\node_modules\gsd-pi\dist\resources\extensions\gsd\unit-context-manifest.js"
```

2. Checking the manifest entries exist:
```bash
grep -A15 '"triage-captures": {' "C:\Users\david.keymel\AppData\Local\nvm\v22.22.2\node_modules\gsd-pi\dist\resources\extensions\gsd\unit-context-manifest.js"
```

3. Checking the normalization code:
```bash
grep -A12 "function unitWritesSource" "C:\Users\david.keymel\AppData\Local\nvm\v22.22.2\node_modules\gsd-pi\dist\resources\extensions\gsd\auto\phases.js"
```

## Important Notes

⚠️ **These changes are temporary.** When you update GSD via `npm update -g gsd-pi`, these patches will be overwritten.

The fix is already merged upstream in:
- **PR #6224** (merged May 16, 2026)
- **Commit:** 157365704ba37e88b291b026420d972ee15eaaef

Wait for the next GSD release that includes this fix, then update normally.

## Next Session

The patches are already applied and will take effect in your next GSD session. No restart is required.

---

*Patches applied: 2026-05-18*
*Source: Upstream PR #6224 (gsd-build/gsd-2)*