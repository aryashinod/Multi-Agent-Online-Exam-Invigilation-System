# cleanup_exam_monitoring.ps1
# Run this from inside C:\Users\aryas\Desktop\hello (the OnlineExamMonitoring folder)
# Removes sensitive/junk files that should never be in a public repo.

$ErrorActionPreference = "Continue"

Write-Host "=== Backing up sensitive folders outside the repo first ===" -ForegroundColor Cyan
$backupDir = "$HOME\Desktop\hydrasave_sensitive_backup"
New-Item -ItemType Directory -Force -Path $backupDir | Out-Null
Copy-Item -Recurse -Force "data\registered_faces" "$backupDir\registered_faces" -ErrorAction SilentlyContinue
Copy-Item -Recurse -Force "data\ssl"              "$backupDir\ssl"              -ErrorAction SilentlyContinue
Copy-Item -Force "data\users.json"                "$backupDir\users.json"       -ErrorAction SilentlyContinue
Write-Host "Backed up to $backupDir (keep this private, off GitHub)." -ForegroundColor Green

Write-Host "`n=== Removing sensitive files from the repo folder ===" -ForegroundColor Yellow
Remove-Item -Recurse -Force "data\registered_faces"
Remove-Item -Recurse -Force "data\ssl"
Remove-Item -Force "data\users.json"

Write-Host "`n=== Removing __pycache__ folders ===" -ForegroundColor Yellow
Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force

Write-Host "`n=== Removing large pretrained model weights (re-download separately) ===" -ForegroundColor Yellow
Remove-Item -Force "data\models\yolov3-tiny.weights"                    -ErrorAction SilentlyContinue
Remove-Item -Force "data\models\res10_300x300_ssd_iter_140000.caffemodel" -ErrorAction SilentlyContinue

Write-Host "`nDone. Review the new structure below:" -ForegroundColor Green
tree /F
