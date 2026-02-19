$target = (Get-Date).Date.AddHours(22)
$now    = Get-Date

if ($now -ge $target) {
    Write-Host "Already past 22:00 — starting immediately." -ForegroundColor Yellow
} else {
    $wait = ($target - $now).TotalSeconds
    $h    = [math]::Floor($wait / 3600)
    $m    = [math]::Floor(($wait % 3600) / 60)
    Write-Host "TIGA Hunt — Overnight Index Scheduler" -ForegroundColor Cyan
    Write-Host "======================================" -ForegroundColor Cyan
    Write-Host "Waiting until 22:00  (~${h}h ${m}m remaining)" -ForegroundColor Green
    Write-Host "Leave this window open and go home."
    Write-Host ""
    Start-Sleep -Seconds ([int]$wait)
}
