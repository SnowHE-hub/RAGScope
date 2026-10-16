param(
    [string]$Document = (Join-Path $PSScriptRoot '../architecture.html'),
    [switch]$FirstOnly
)

$ErrorActionPreference = 'Stop'
$browserExecutable = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'
$documentPath = (Resolve-Path -LiteralPath $Document).Path
$previewRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'previews')).Path
$sourceHash = (Get-FileHash -LiteralPath $documentPath -Algorithm SHA256).Hash.ToLowerInvariant()
$sessionName = 'edge-' + $sourceHash.Substring(0,12) + '-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
$outputRoot = Join-Path $previewRoot $sessionName
$profilePath = Join-Path $outputRoot 'profile'
$null = New-Item -ItemType Directory -Path $profilePath -Force
# Only a dedicated profile under this document's preview directory is used.
if (-not ([IO.Path]::GetFullPath($profilePath).StartsWith([IO.Path]::GetFullPath($previewRoot) + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase))) {
    throw 'Unexpected profile location'
}
$documentUri = ([Uri]::new($documentPath)).AbsoluteUri
$targets = @(
    @{Name='desktop-top'; Width=1440; Height=1000; Fragment=''},
    @{Name='mobile-top'; Width=390; Height=844; Fragment=''},
    @{Name='desktop-architecture'; Width=1440; Height=1000; Fragment='4-系统架构总览'},
    @{Name='mobile-architecture'; Width=390; Height=844; Fragment='4-系统架构总览'},
    @{Name='desktop-evaluation'; Width=1440; Height=1000; Fragment='12-评价系统与实验设计'},
    @{Name='mobile-evaluation'; Width=390; Height=844; Fragment='12-评价系统与实验设计'},
    @{Name='desktop-interview'; Width=1440; Height=1000; Fragment='17-面试问题与个人学习清单'},
    @{Name='mobile-interview'; Width=390; Height=844; Fragment='17-面试问题与个人学习清单'}
)
if ($FirstOnly) { $targets = @($targets[0]) }
$rows = [Collections.Generic.List[Object]]::new()
foreach ($target in $targets) {
    $png = Join-Path $outputRoot ($target.Name + '.png')
    $stdout = Join-Path $outputRoot ($target.Name + '.stdout.txt')
    $stderr = Join-Path $outputRoot ($target.Name + '.stderr.txt')
    $url = $documentUri
    if ($target.Fragment) { $url += '#' + [Uri]::EscapeDataString($target.Fragment) }
    $arguments = @('--headless', '--disable-gpu', '--disable-background-networking', '--disable-component-update',
        '--no-first-run', '--no-default-browser-check', '--disable-extensions', '--disable-sync',
        '--hide-scrollbars', '--force-device-scale-factor=1', '--run-all-compositor-stages-before-draw',
        '--virtual-time-budget=2500', '--timeout=15000',
        ('--user-data-dir=' + $profilePath), ('--window-size=' + $target.Width + ',' + $target.Height),
        ('--screenshot=' + $png), $url)
    $startTime = [DateTimeOffset]::UtcNow
    $process = Start-Process -FilePath $browserExecutable -ArgumentList $arguments -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $done = $process.WaitForExit(30000)
    $rows.Add([PSCustomObject]@{
        name=$target.Name; requested_width=$target.Width; requested_height=$target.Height;
        document=$documentPath; document_sha256=$sourceHash; url=$url;
        executable=$browserExecutable; profile=$profilePath; arguments=$arguments;
        pid=$process.Id; started_utc=$startTime.ToString('o'); terminal=$done;
        exit_code=$(if ($done) {$process.ExitCode} else {$null});
        screenshot=$png; screenshot_exists=(Test-Path -LiteralPath $png);
        stdout=$stdout; stderr=$stderr
    })
    $rows | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $outputRoot 'RENDER_COMMANDS.json') -Encoding utf8
    if (-not $done) { throw "Own render process $($process.Id) did not exit in 30 seconds. No other process was stopped. Inspect the exact recorded process before taking further action." }
    if ($process.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $png)) { throw "Render failed; inspect $stderr. No alternative browser invocation was made." }
}
Write-Output (Join-Path $outputRoot 'RENDER_COMMANDS.json')
