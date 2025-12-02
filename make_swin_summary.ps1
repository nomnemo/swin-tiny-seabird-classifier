# Directory containing your .log files
$logDir = "runs_swin\logs"
$outCsv = Join-Path $logDir "swin_runs_summary.csv"

$results = @()

Get-ChildItem -Path $logDir -Filter *.log | ForEach-Object {
    $logFile = $_
    $text = Get-Content $logFile.FullName -Raw

    # Defaults
    $runDir = $null
    $runName = $null
    $model = $null
    $epochs = $null
    $lr = $null
    $weightDecay = $null
    $accumSteps = $null
    $bestValAcc = $null
    $valMacroF1 = $null
    $valMAP = $null
    $testMacroF1 = $null
    $testMAP = $null

    # [info] run directory: runs_swin\swin_mpc200_ep10_lr0100_wd0100_as1
    $m = [regex]::Match($text, '^\[info\]\s*run directory:\s*(.+)$', 'Multiline')
    if ($m.Success) {
        $runDir = $m.Groups[1].Value.Trim()
        $runName = Split-Path $runDir -Leaf
    }

    # [info] model: swin_tiny_patch4_window7_224
    $m = [regex]::Match($text, '^\[info\]\s*model:\s*(.+)$', 'Multiline')
    if ($m.Success) {
        $model = $m.Groups[1].Value.Trim()
    }

    # [info] epochs: 10, lr: 0.0001, weight_decay: 0.01, accum_steps: 1
    $m = [regex]::Match(
        $text,
        '^\[info\]\s*epochs:\s*(\d+),\s*lr:\s*([0-9.eE-]+),\s*weight_decay:\s*([0-9.eE-]+),\s*accum_steps:\s*(\d+)',
        'Multiline'
    )
    if ($m.Success) {
        $epochs      = $m.Groups[1].Value
        $lr          = $m.Groups[2].Value
        $weightDecay = $m.Groups[3].Value
        $accumSteps  = $m.Groups[4].Value
    }

    # ep 01 | ... | val acc 0.757 loss ...
    $matches = [regex]::Matches($text, 'ep\s+\d+\s*\|.*?val acc\s+([0-9.]+)')
    if ($matches.Count -gt 0) {
        $vals = $matches | ForEach-Object { [double]$_.Groups[1].Value }
        $bestValAcc = ($vals | Measure-Object -Maximum).Maximum
    }

    # Validation report macro-F1: 0.869
    $m = [regex]::Match($text, 'Validation report macro-F1:\s*([0-9.]+)')
    if ($m.Success) { $valMacroF1 = $m.Groups[1].Value }

    # Validation report mAP (macro, one-vs-rest): 0.923
    $m = [regex]::Match($text, 'Validation report mAP \(macro, one-vs-rest\):\s*([0-9.]+)')
    if ($m.Success) { $valMAP = $m.Groups[1].Value }

    # Test report macro-F1: 0.879
    $m = [regex]::Match($text, 'Test report macro-F1:\s*([0-9.]+)')
    if ($m.Success) { $testMacroF1 = $m.Groups[1].Value }

    # Test report mAP (macro, one-vs-rest): 0.916
    $m = [regex]::Match($text, 'Test report mAP \(macro, one-vs-rest\):\s*([0-9.]+)')
    if ($m.Success) { $testMAP = $m.Groups[1].Value }

    $results += [pscustomobject]@{
        RunName        = $runName
        LogFile        = $logFile.Name
        Model          = $model
        Epochs         = $epochs
        LR             = $lr
        WeightDecay    = $weightDecay
        AccumSteps     = $accumSteps
        BestValAcc     = $bestValAcc
        ValMacroF1     = $valMacroF1
        ValMAP         = $valMAP
        TestMacroF1    = $testMacroF1
        TestMAP        = $testMAP
    }
}

$results | Export-Csv -Path $outCsv -NoTypeInformation
Write-Host "Saved summary to $outCsv"
