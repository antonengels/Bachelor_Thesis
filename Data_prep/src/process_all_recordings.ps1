#Requires -Version 5.0

<#
.SYNOPSIS
    Verarbeitet alle Ordner in 01_Data mit NID-Parquet-Dateien durch den Merge-/Trip-/Plot-Workflow.

.DESCRIPTION
    Für jeden Unterordner in "C:\Users\Anton\Documents\Bachelorarbeit\01_Data" mit
    NID_1/NID_6/NID_31/NID_32-Parquet-Dateien:
    - Merge durchführen
    - Trips erzeugen (ohne merged.parquet)
    - Plot erstellen (NID-Plots plus ein Plot je Trip)
    
    Alle Outputs werden im jeweiligen Ordner gespeichert.

.PARAMETER SourceDataDir
    Basis-Verzeichnis mit den Recording-Ordnern (Default: C:\Users\Anton\Documents\Bachelorarbeit\01_Data)

.PARAMETER DataPrepDir
    Verzeichnis mit den Python-Skripten (Default: C:\Users\Anton\Documents\Bachelorarbeit\02_Code\Data_prep)

.EXAMPLE
    .\process_all_recordings.ps1
    
.EXAMPLE
    .\process_all_recordings.ps1 -SourceDataDir "D:\Recordings" -DataPrepDir "C:\Code\Data_prep"
#>

param(
    [string]$SourceDataDir = "C:\Users\Anton\Documents\Bachelorarbeit\01_Data",
    [string]$DataPrepDir = "C:\Users\Anton\Documents\Bachelorarbeit\02_Code\Data_prep"
)

# Validierung der Verzeichnisse
if (-not (Test-Path $SourceDataDir)) {
    Write-Error "Quellverzeichnis nicht gefunden: $SourceDataDir"
    exit 1
}

if (-not (Test-Path $DataPrepDir)) {
    Write-Error "Data_prep-Verzeichnis nicht gefunden: $DataPrepDir"
    exit 1
}

$pythonExe = Join-Path $DataPrepDir ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Error "Python-Interpreter nicht gefunden: $pythonExe"
    exit 1
}

$mainScript = Join-Path $DataPrepDir "src\main.py"
if (-not (Test-Path $mainScript)) {
    Write-Error "main.py nicht gefunden: $mainScript"
    exit 1
}

# Finde alle Unterordner
$recordingDirs = Get-ChildItem -Path $SourceDataDir -Directory -ErrorAction SilentlyContinue

if ($recordingDirs.Count -eq 0) {
    Write-Warning "Keine Unterordner in $SourceDataDir gefunden."
    exit 0
}

$totalDirs = @($recordingDirs).Count
$processedCount = 0
$successCount = 0
$failureCount = 0

Write-Host "Starte Verarbeitung..." -ForegroundColor Cyan
Write-Host "Quellverzeichnis: $SourceDataDir" -ForegroundColor Gray
Write-Host "Gefundene Ordner: $totalDirs" -ForegroundColor Gray
Write-Host ""

foreach ($dir in $recordingDirs) {
    $processedCount++
    $dirName = $dir.Name
    $dirPath = $dir.FullName
    
    
    Write-Host "[$processedCount/$totalDirs] [>>] $dirName" -ForegroundColor Cyan
    Write-Host "  Ausgabeverzeichnis: $dirPath" -ForegroundColor Gray
    
    try {
        # Rufe main.py mit den spezifischen Parametern auf
        $startTime = Get-Date
        
        # Extrahiere Prefix aus Ordnernamen (bevorzugt Datum_Index, z.B. 20260408_01)
        $dirNameParts = $dirName -split '_'
        if ($dirNameParts.Count -ge 2 -and $dirNameParts[1] -match '^\d+$') {
            $outputPrefix = "$($dirNameParts[0])_$($dirNameParts[1])"
        } elseif ($dirNameParts.Count -ge 1) {
            $outputPrefix = $dirNameParts[0]
        } else {
            $outputPrefix = $dirName
        }
        $prefixStr = if ([string]::IsNullOrWhiteSpace($outputPrefix)) { "" } else { "$outputPrefix`_" }

        $nidInputs = @(
            (Join-Path $dirPath "${prefixStr}NID_1.parquet"),
            (Join-Path $dirPath "${prefixStr}NID_6.parquet"),
            (Join-Path $dirPath "${prefixStr}NID_31.parquet"),
            (Join-Path $dirPath "${prefixStr}NID_32.parquet")
        )

        $missingNids = @($nidInputs | Where-Object { -not (Test-Path $_) })
        if ($missingNids.Count -gt 0) {
            Write-Host "  [NO] NID-Parquet-Dateien unvollstaendig, Ordner wird uebersprungen." -ForegroundColor Yellow
            foreach ($missing in $missingNids) {
                Write-Host "    fehlt: $missing" -ForegroundColor Yellow
            }
            Write-Host ""
            continue
        }

        $skipExport = $true
        $skipMerge = $false
        $skipPlot = $false

        $mainArgs = @(
            "-u",
            $mainScript,
            "--output-dir", $dirPath,
            "--output-prefix", $outputPrefix
        )
        $mainArgs += "--skip-export"
        if ($skipMerge) { $mainArgs += "--skip-merge" }
        if ($skipPlot) { $mainArgs += "--skip-plot" }
        
        & $pythonExe $mainArgs `
            2>&1 | ForEach-Object {
                $line = $_
                if ($line -match "OK:" -or $line -match "Zeilen:" -or $line -match "Spalten:") {
                    Write-Host "  $line" -ForegroundColor Green
                } elseif ($line -match "Info:|uebersprungen" -or $line -match "fehlt:") {
                    Write-Host "  $line" -ForegroundColor Yellow
                } elseif ($line -match "===" -or $line -match "Befehl:") {
                    Write-Host "  $line" -ForegroundColor Gray
                } elseif ($line -match "Error|Fehler|Traceback|Exception") {
                    Write-Host "  [ERROR] $line" -ForegroundColor Red
                } else {
                    # Alle anderen Zeilen auch zeigen
                    Write-Host "  $line" -ForegroundColor Gray
                }
            }
        
        $exitCode = $LASTEXITCODE
        $duration = ((Get-Date) - $startTime).TotalSeconds
        
        if ($exitCode -eq 0) {
            Write-Host "  [OK] Erfolgreich abgeschlossen ($([Math]::Round($duration))s)" -ForegroundColor Green
            $successCount++
        } else {
            Write-Host "  [ER] Fehler (Exit-Code: $exitCode)" -ForegroundColor Red
            $failureCount++
        }
    }
    catch {
        Write-Host "  [ER] Fehler bei der Ausführung: $_" -ForegroundColor Red
        $failureCount++
    }
    
    Write-Host ""
}

# Zusammenfassung
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "ZUSAMMENFASSUNG" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Insgesamt gescannt: $totalDirs Ordner" -ForegroundColor Gray
Write-Host "Verarbeitet: $processedCount Ordner" -ForegroundColor Gray
Write-Host "[OK] Erfolgreich: $successCount" -ForegroundColor Green
Write-Host "[ER] Fehler: $failureCount" -ForegroundColor $(if ($failureCount -gt 0) { "Red" } else { "Green" })
Write-Host ""

if ($failureCount -gt 0) {
    Write-Host "Verarbeitung abgeschlossen mit $failureCount Fehler(n)." -ForegroundColor Yellow
    exit 1
} else {
    Write-Host "Verarbeitung erfolgreich abgeschlossen." -ForegroundColor Green
    exit 0
}
