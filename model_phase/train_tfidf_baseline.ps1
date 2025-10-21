# TF-IDF Baseline Training Script (PowerShell)
# Trains TF-IDF baseline model with automatic hyperparameter tuning via grid search
# Usage: .\model_phase\train_tfidf_baseline.ps1 -Dataset "your-username/game-reviews-sentiment"

param(
    [Parameter(Mandatory=$true)]
    [string]$Dataset,
    
    [Parameter(Mandatory=$false)]
    [double]$GridsearchSubset = 0.1,
    
    [Parameter(Mandatory=$false)]
    [double]$FinalSubset = 1.0,
    
    [Parameter(Mandatory=$false)]
    [string]$OutputBaseDir = "model_phase\results",
    
    [Parameter(Mandatory=$false)]
    [int]$NJobs = 0,
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipGridsearch
)

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "TF-IDF Baseline Model Training" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Dataset: $Dataset"
Write-Host "Grid Search Subset: $GridsearchSubset"
Write-Host "Final Training Subset: $FinalSubset"
Write-Host "Output Directory: $OutputBaseDir"
if ($NJobs -gt 0) {
    Write-Host "CPU Cores: $NJobs"
}
Write-Host ""

# Step 1: Grid Search (if not skipped)
if (-not $SkipGridsearch) {
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "STEP 1/3: Running Grid Search" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "Finding best hyperparameters on $GridsearchSubset subset..."
    Write-Host ""
    
    $GridsearchDir = Join-Path $OutputBaseDir "gridsearch"
    New-Item -ItemType Directory -Path $GridsearchDir -Force | Out-Null
    
    # Grid search parameters
    $MaxFeaturesList = @(5000, 10000, 20000)
    $NgramConfigs = @(@{min=1; max=1}, @{min=1; max=2}, @{min=1; max=3})
    $MaxIterList = @(500, 1000, 2000)
    
    # Results file
    $ResultsFile = Join-Path $GridsearchDir "gridsearch_results.txt"
    "Grid Search Results - $(Get-Date)" | Out-File $ResultsFile
    "Dataset: $Dataset" | Out-File $ResultsFile -Append
    "Subset: $GridsearchSubset" | Out-File $ResultsFile -Append
    "==========================================" | Out-File $ResultsFile -Append
    "" | Out-File $ResultsFile -Append
    
    # Best results tracking
    $BestF1 = 0
    $BestConfig = ""
    $BestOutputDir = ""
    
    # Counter
    $TotalConfigs = $MaxFeaturesList.Length * $NgramConfigs.Length * $MaxIterList.Length
    $Current = 0
    
    Write-Host "Total configurations to test: $TotalConfigs" -ForegroundColor Yellow
    Write-Host ""
    
    # Grid search loop
    foreach ($MaxFeatures in $MaxFeaturesList) {
        foreach ($Ngram in $NgramConfigs) {
            foreach ($MaxIter in $MaxIterList) {
                $Current++
                
                Write-Host "==========================================" -ForegroundColor Cyan
                Write-Host "Configuration $Current/$TotalConfigs" -ForegroundColor Cyan
                Write-Host "==========================================" -ForegroundColor Cyan
                Write-Host "max_features: $MaxFeatures"
                Write-Host "ngram_range: ($($Ngram.min), $($Ngram.max))"
                Write-Host "max_iter: $MaxIter"
                Write-Host ""
                
                # Create unique output directory
                $OutputDir = Join-Path $GridsearchDir "config_${Current}_mf${MaxFeatures}_ng$($Ngram.min)$($Ngram.max)_mi${MaxIter}"
                
                # Build command (no HuggingFace upload during grid search)
                $TrainArgs = @(
                    "model_phase\main_tfidf_baseline.py",
                    "--dataset", $Dataset,
                    "--max_features", $MaxFeatures,
                    "--ngram_min", $Ngram.min,
                    "--ngram_max", $Ngram.max,
                    "--max_iter", $MaxIter,
                    "--subset", $GridsearchSubset,
                    "--output_dir", $OutputDir,
                    "--no_upload"
                )
                
                # Add n_jobs if specified
                if ($NJobs -gt 0) {
                    $TrainArgs += @("--n_jobs", $NJobs)
                }
                
                # Run training
                Write-Host "Running training..." -ForegroundColor Gray
                & python $TrainArgs
                
                # Extract validation F1 score from results.json
                $ResultsJson = Join-Path $OutputDir "results.json"
                if (Test-Path $ResultsJson) {
                    $Results = Get-Content $ResultsJson | ConvertFrom-Json
                    $ValF1 = [math]::Round($Results.validation_f1, 4)
                    $ValAcc = [math]::Round($Results.validation_accuracy, 4)
                    $TrainTime = [math]::Round($Results.training_time, 2)
                    
                    Write-Host ""
                    Write-Host "Results:" -ForegroundColor Green
                    Write-Host "  Validation F1: $ValF1"
                    Write-Host "  Validation Accuracy: $ValAcc"
                    Write-Host "  Training Time: ${TrainTime}s"
                    Write-Host ""
                    
                    # Log to results file
                    "Configuration ${Current}:" | Out-File $ResultsFile -Append
                    "  max_features: $MaxFeatures" | Out-File $ResultsFile -Append
                    "  ngram_range: ($($Ngram.min), $($Ngram.max))" | Out-File $ResultsFile -Append
                    "  max_iter: $MaxIter" | Out-File $ResultsFile -Append
                    "  Validation F1: $ValF1" | Out-File $ResultsFile -Append
                    "  Validation Accuracy: $ValAcc" | Out-File $ResultsFile -Append
                    "  Training Time: ${TrainTime}s" | Out-File $ResultsFile -Append
                    "  Output: $OutputDir" | Out-File $ResultsFile -Append
                    "" | Out-File $ResultsFile -Append
                    
                    # Update best if this is better
                    if ($ValF1 -gt $BestF1) {
                        $BestF1 = $ValF1
                        $BestConfig = "max_features=$MaxFeatures, ngram=($($Ngram.min),$($Ngram.max)), max_iter=$MaxIter"
                        $BestOutputDir = $OutputDir
                        Write-Host "*** NEW BEST CONFIGURATION! ***" -ForegroundColor Yellow
                        Write-Host ""
                    }
                } else {
                    Write-Host "Error: Results file not found at $ResultsJson" -ForegroundColor Red
                    "Configuration ${Current}: ERROR - Results file not found" | Out-File $ResultsFile -Append
                    "" | Out-File $ResultsFile -Append
                }
                
                Write-Host ""
            }
        }
    }
    
    # Save best config summary
    $BestConfigFile = Join-Path $GridsearchDir "best_config.txt"
    "Best Configuration Found" | Out-File $BestConfigFile
    "======================" | Out-File $BestConfigFile -Append
    "Configuration: $BestConfig" | Out-File $BestConfigFile -Append
    "Validation F1: $BestF1" | Out-File $BestConfigFile -Append
    "Model Directory: $BestOutputDir" | Out-File $BestConfigFile -Append
    
    Write-Host "All results saved to: $ResultsFile" -ForegroundColor Green
    Write-Host "Best configuration saved to: $BestConfigFile" -ForegroundColor Green
    
    # Build grid search command
    $GsArgs = @{
        Dataset = $Dataset
        Subset = $GridsearchSubset
        OutputBaseDir = $GridsearchDir
    }
    
    if ($NJobs -gt 0) {
        $GsArgs.NJobs = $NJobs
    }
    
    Write-Host "Running grid search..." -ForegroundColor Cyan
    Write-Host ""
    
    & "$PSScriptRoot\gridsearch_baseline.ps1" @GsArgs
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Grid search failed" -ForegroundColor Red
        exit 1
    }
    
    Write-Host ""
    Write-Host "✓ Grid search complete!" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "STEP 1/3: Skipping Grid Search" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "Using provided hyperparameters..."
    Write-Host ""
    $GridsearchDir = Join-Path $OutputBaseDir "gridsearch"
}

# Step 2: Extract Best Configuration
Write-Host "============================================================" -ForegroundColor Green
Write-Host "STEP 2/3: Extracting Best Configuration" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green

$BestConfigFile = Join-Path $GridsearchDir "best_config.txt"

if (-not (Test-Path $BestConfigFile)) {
    Write-Host "Error: Best config file not found at $BestConfigFile" -ForegroundColor Red
    Write-Host "Make sure grid search completed successfully." -ForegroundColor Red
    exit 1
}

Write-Host "Reading best configuration from: $BestConfigFile"
Write-Host ""
Get-Content $BestConfigFile
Write-Host ""

# Extract hyperparameters from best config
$ConfigContent = Get-Content $BestConfigFile
$ConfigLine = $ConfigContent | Select-String "Configuration:"

# Parse: "Configuration: max_features=10000, ngram=(1,2), max_iter=1000"
if ($ConfigLine -match "max_features=(\d+).*ngram=\((\d+),(\d+)\).*max_iter=(\d+)") {
    $MaxFeatures = [int]$matches[1]
    $NgramMin = [int]$matches[2]
    $NgramMax = [int]$matches[3]
    $MaxIter = [int]$matches[4]
} else {
    Write-Host "Error: Could not parse best configuration" -ForegroundColor Red
    exit 1
}

Write-Host "Extracted hyperparameters:" -ForegroundColor Yellow
Write-Host "  max_features: $MaxFeatures"
Write-Host "  ngram_range: ($NgramMin, $NgramMax)"
Write-Host "  max_iter: $MaxIter"
Write-Host ""

# Step 3: Final Training with Best Configuration
Write-Host "============================================================" -ForegroundColor Green
Write-Host "STEP 3/3: Final Training with Best Configuration" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Training on $FinalSubset subset with best hyperparameters..."
Write-Host "This model will be uploaded to HuggingFace Hub." -ForegroundColor Yellow
Write-Host ""

# Build final training command
$FinalArgs = @(
    "model_phase\main_tfidf_baseline.py",
    "--dataset", $Dataset,
    "--max_features", $MaxFeatures,
    "--ngram_min", $NgramMin,
    "--ngram_max", $NgramMax,
    "--max_iter", $MaxIter,
    "--subset", $FinalSubset
)

if ($NJobs -gt 0) {
    $FinalArgs += @("--n_jobs", $NJobs)
}

Write-Host "Running final training..." -ForegroundColor Cyan
Write-Host ""

& python $FinalArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Final training failed" -ForegroundColor Red
    exit 1
}

# Step 4: Summary
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "PIPELINE COMPLETE!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Summary:" -ForegroundColor Yellow
Write-Host "1. ✓ Grid search found best hyperparameters"
Write-Host "2. ✓ Final model trained with optimal configuration"
Write-Host "3. ✓ Results uploaded to HuggingFace Hub"
Write-Host ""
Write-Host "Best Configuration Used:" -ForegroundColor Yellow
Write-Host "  max_features: $MaxFeatures"
Write-Host "  ngram_range: ($NgramMin, $NgramMax)"
Write-Host "  max_iter: $MaxIter"
Write-Host ""
Write-Host "Check your HuggingFace profile for the uploaded model!" -ForegroundColor Cyan
Write-Host ""
