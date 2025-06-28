# CUDA Tutor Backend Startup Script
Write-Host "Starting CUDA Tutor Backend..." -ForegroundColor Green
Write-Host "Using Python: C:\Users\Captain\AppData\Local\Programs\Python\Python310\python.exe" -ForegroundColor Yellow
Write-Host ""

# Set the Python alias
Set-Alias -Name python -Value "C:\Users\Captain\AppData\Local\Programs\Python\Python310\python.exe" -Scope Global

# Change to the script directory
Set-Location $PSScriptRoot

# Start the backend
try {
    & "C:\Users\Captain\AppData\Local\Programs\Python\Python310\python.exe" backend/app.py
}
catch {
    Write-Host "Error starting backend: $_" -ForegroundColor Red
    Read-Host "Press Enter to exit"
} 