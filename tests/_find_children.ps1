$ErrorActionPreference = "SilentlyContinue"
$targetPid = $args[0]
# Direct children
$procs = Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq [int]$targetPid }
$childIds = @()
foreach ($p in $procs) {
    $childIds += [int]$p.ProcessId
    $cl = $p.CommandLine
    if ($null -ne $cl -and $cl.Length -gt 160) { $cl = $cl.Substring(0, 160) }
    Write-Output ("CHILD|" + $p.ProcessId + "|" + $p.Name + "|" + $cl)
}
# Grandchildren (the app may be parented one level down via a shim)
foreach ($cid in $childIds) {
    $gprocs = Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $cid }
    foreach ($g in $gprocs) {
        $gcl = $g.CommandLine
        if ($null -ne $gcl -and $gcl.Length -gt 160) { $gcl = $gcl.Substring(0, 160) }
        Write-Output ("GRAND|" + $g.ProcessId + "|" + $g.Name + "|" + $gcl)
    }
}
Write-Output "DONE"
