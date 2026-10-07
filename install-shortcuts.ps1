# install-shortcuts.ps1 - Start-menu shortcuts for MeetAndRead and the
# Issue Reporter (issue #111, docs/specs/issue-reporting.md "Frozen build").
#
# The app ships as a portable zip (no installer): this script IS the
# documented install mechanism for Start-menu shortcuts. Run it once
# after extracting the release zip (right-click -> Run with PowerShell,
# or from a PowerShell prompt):
#
#   powershell -ExecutionPolicy Bypass -File .\install-shortcuts.ps1
#
# It creates two shortcuts in the user's Start-menu Programs folder:
#   - MeetAndRead                -> <bundle>\meetandread.exe
#   - MeetAndRead Issue Reporter -> <bundle>\issue-reporter.exe
#
# The Issue Reporter shortcut is the standalone reporting path (ADR
# 0003): a user whose app won't start at all can still launch the
# reporter and report the startup crash. Re-running the script
# overwrites the shortcuts in place (safe re-run after an upgrade -
# point it at the new extraction directory).

$ErrorActionPreference = 'Stop'

# Resolve the bundle directory from this script's own location: the
# script ships at the root of the onedir bundle (copied there by CI
# and by a local pyinstaller run - see meetandread.spec datas and
# .github/workflows/ci.yml).
$bundleDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$appExe = Join-Path $bundleDir 'meetandread.exe'
$reporterExe = Join-Path $bundleDir 'issue-reporter.exe'

foreach ($exe in @($appExe, $reporterExe)) {
    if (-not (Test-Path -LiteralPath $exe)) {
        Write-Error "Required executable not found next to this script: $exe (expected in bundle: $bundleDir)"
    }
}

$startMenu = [Environment]::GetFolderPath('Programs')
$ws = New-Object -ComObject WScript.Shell

$shortcuts = @(
    @{ Link = Join-Path $startMenu 'MeetAndRead.lnk';            Target = $appExe;      Description = 'MeetAndRead - meeting audio transcription widget' },
    @{ Link = Join-Path $startMenu 'MeetAndRead Issue Reporter.lnk'; Target = $reporterExe; Description = 'Report a MeetAndRead bug by reproducing it under diagnostics capture' }
)

foreach ($s in $shortcuts) {
    $shortcut = $ws.CreateShortcut($s.Link)
    $shortcut.TargetPath = $s.Target
    $shortcut.WorkingDirectory = $bundleDir
    $shortcut.Description = $s.Description
    $shortcut.Save()
    Write-Host "Created $($s.Link)"
}

Write-Host ''
Write-Host 'Start-menu shortcuts created. The Issue Reporter is also'
Write-Host 'directly runnable from the bundle: issue-reporter.exe'
