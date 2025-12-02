# Grid-run PowerShell script for 5_train_swin.py
# Runs combinations of hyperparameters sequentially and writes logs per run.
# Usage: Open PowerShell, navigate to repo root and run:
#   powershell -ExecutionPolicy Bypass -File .\run_grid.ps1

# --- user-editable grid ---
$epochs_list = @(10,20,30)
$lrs = @(0.0003, 0.0001, 0.00005)
$wds = @(0.01, 0.03)
$mpcs = @(50,100,200,500)
$accum_steps = @(1)  # single value but you can add more

# path to python script (adjust if needed)
$script = "c:\\Users\\Audub\\Classification\\5_train_swin.py"
$python = "python"

# where to store run logs (will be created)
$logs_dir = Join-Path -Path (Get-Location) -ChildPath "runs_swin\logs"
New-Item -ItemType Directory -Path $logs_dir -Force | Out-Null

# optional: run-only subset by setting $DRY_RUN to $true
$DRY_RUN = $false

Write-Output "Starting grid run: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

foreach ($ep in $epochs_list) {
    foreach ($lr in $lrs) {
        foreach ($wd in $wds) {
            foreach ($mpc in $mpcs) {
                foreach ($as in $accum_steps) {
                    $ts = Get-Date -Format "yyyyMMdd-HHmmss"
                    # sanitize floats for filenames (replace . with p)
                    $lr_safe = ($lr -replace '\\.', 'p')
                    $wd_safe = ($wd -replace '\\.', 'p')

                    $log_name = "run_ep${ep}_lr${lr_safe}_wd${wd_safe}_mpc${mpc}_as${as}_${ts}.log"
                    $log_path = Join-Path $logs_dir $log_name

                    $args = @(
                        $script,
                        "--epochs", $ep,
                        "--lr", $lr,
                        "--weight-decay", $wd,
                        "--max-per-class", $mpc,
                        "--accum-steps", $as
                    )

                    $cmd_display = "$python `"$script`" --epochs $ep --lr $lr --weight-decay $wd --max-per-class $mpc --accum-steps $as"
                    Write-Output "\n=== Running: $cmd_display ==="
                    Write-Output "Log -> $log_path"

                    if ($DRY_RUN) {
                        Write-Output "DRY RUN, not executing."
                    } else {
                        # run the command and tee stdout+stderr to logfile
                        & $python $script --epochs $ep --lr $lr --weight-decay $wd --max-per-class $mpc --accum-steps $as 2>&1 | Tee-Object -FilePath $log_path

                        # check exit code from last program
                        if ($LASTEXITCODE -ne 0) {
                            Write-Output "Run failed (exit code $LASTEXITCODE). Check $log_path"
                            # option: break, continue, or stop the entire grid. We'll continue by default.
                            # break
                        } else {
                            Write-Output "Run completed successfully."
                        }

                        # small sleep to let resources settle (optional)
                        Start-Sleep -Seconds 3
                    }
                }
            }
        }
    }
}

Write-Output "Grid finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
