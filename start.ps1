$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectDir
$Log = Join-Path $ProjectDir "startup.log"
Add-Content -LiteralPath $Log "[$(Get-Date -Format s)] start.ps1 launched"

function Find-Python {
    $candidates = @(
        (Join-Path $ProjectDir ".venv\Scripts\python.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"),
        (Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    foreach ($name in @("py", "python")) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command) { return $command.Source }
    }
    return $null
}

function Test-PortOpen {
    param([int]$Port)
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $async = $client.BeginConnect("127.0.0.1", $Port, $null, $null)
        $connected = $async.AsyncWaitHandle.WaitOne(250)
        $isOpen = $connected -and $client.Connected
        $client.Close()
        return $isOpen
    } catch {
        return $false
    }
}

function Test-InsightAIHealth {
    param([int]$Port)
    try {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/_stcore/health" -UseBasicParsing -TimeoutSec 2
        return ($response.StatusCode -eq 200 -and $response.Content.Trim() -eq "ok")
    } catch {
        return $false
    }
}

$Python = Find-Python
if (-not $Python) {
    Add-Content -LiteralPath $Log "[ERROR] Python was not found"
    throw "Python 3.10-3.13 was not found. Install Python, then run this script again."
}

$ApiKey = $env:INSIGHTAI_API_KEY
if ([string]::IsNullOrWhiteSpace($ApiKey)) {
    $SecureKey = Read-Host "请输入 DeepSeek API Key（直接回车可只运行本地公开样例）" -AsSecureString
    if ($SecureKey.Length -gt 0) {
        $KeyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecureKey)
        try {
            $ApiKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($KeyPointer)
        } finally {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($KeyPointer)
        }
        $env:INSIGHTAI_API_KEY = $ApiKey
    }
}

$Port = 8501
if (Test-InsightAIHealth -Port $Port) {
    if ([string]::IsNullOrWhiteSpace($ApiKey)) {
        Write-Host "本次未输入密钥；网页已在 http://localhost:$Port 运行。只能使用公开样例和已有结果。"
    } else {
        Write-Host "密钥已输入，但网页服务已在 http://localhost:$Port 运行。请关闭旧服务并重新启动，密钥才会进入该服务进程。"
    }
    Add-Content -LiteralPath $Log "Existing InsightAI service is healthy on port $Port; reused."
    exit 0
}

$PortBusy = Test-PortOpen -Port $Port
while ($PortBusy -and $Port -lt 8510) {
    $Port = $Port + 1
    $PortBusy = Test-PortOpen -Port $Port
}
if ($PortBusy) {
    Add-Content -LiteralPath $Log "[ERROR] Ports 8501-8510 are occupied."
    throw "8501-8510 端口都已被占用，请关闭旧服务或查阅 startup.log 后再试。"
}

$VenvPython = Join-Path $ProjectDir ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    Add-Content -LiteralPath $Log "Creating virtual environment with $Python"
    & $Python -m venv .venv *>> $Log
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $VenvPython)) {
        throw "Unable to create the project environment. Read startup.log for details."
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $ProjectDir ".deps-ready"))) {
    Write-Host "Installing dependencies for the first run..."
    & $VenvPython -m pip install -r requirements.txt *>> $Log
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency installation failed. Read startup.log for details."
    }
    New-Item -Path (Join-Path $ProjectDir ".deps-ready") -ItemType File -Force | Out-Null
}

Write-Host "Starting Streamlit at http://localhost:$Port"
Add-Content -LiteralPath $Log "Starting Streamlit at port $Port"
& $VenvPython -m streamlit run app.py --server.port $Port --server.headless true --browser.gatherUsageStats false *>> $Log
if ($LASTEXITCODE -ne 0) {
    throw "Streamlit stopped unexpectedly. Read startup.log for details."
}
