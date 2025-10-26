# TF-IDF Baseline Training Script (PowerShell)
# Trains TF-IDF baseline model with automatic hyperparameter tuning via grid search
# Usage: .\model_phase\train_tfidf_baseline.ps1 -Dataset "your-username/game-reviews-sentiment"

# TF-IDF parameters (constants - not tuned)
$script:MaxFeatures = 20000
$script:NgramMin = 1
$script:NgramMax = 2

# Grid search parameters (tune C parameter for Logistic Regression)
$script:CValues = @(0.01, 0.1, 1.0, 10.0, 100.0)

# Load dataset from .env if available
$envFile = ".env"
$envDataset = $null
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^HF_DATASET_NAME=(.+)$') {
            $envDataset = $matches[1]
        }
    }
}

param(
    [Parameter(Mandatory=$false)]
    [string]$Dataset = $envDataset,
    
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

# Validate dataset
if (-not $Dataset) {
    Write-Host "Error: --dataset is required (not found in .env or command line)" -ForegroundColor Red
    Write-Host "Usage: .\model_phase\train_tfidf_baseline.ps1 -Dataset your-username/game-reviews-sentiment"
    Write-Host "Or set HF_DATASET_NAME in .env file"
    exit 1
}

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
    $TotalConfigs = $CValues.Length
    $Current = 0
    
    Write-Host "Total configurations to test: $TotalConfigs" -ForegroundColor Yellow
    Write-Host "TF-IDF Settings (fixed): max_features=$MaxFeatures, ngram=($NgramMin,$NgramMax)"
    Write-Host ""
    
    # Grid search loop
    foreach ($CValue in $CValues) {
        $Current++
        
        Write-Host "==========================================" -ForegroundColor Cyan
        Write-Host "Configuration $Current/$TotalConfigs" -ForegroundColor Cyan
        Write-Host "==========================================" -ForegroundColor Cyan
        Write-Host "C (regularization): $CValue"
        Write-Host ""
        
        # Create unique output directory
        $OutputDir = Join-Path $GridsearchDir "config_${Current}_C${CValue}"
        
        # Build command (no HuggingFace upload during grid search)
        $TrainArgs = @(
            "model_phase\main_tfidf_baseline.py",
            "--dataset", $Dataset,
            "--max_features", $MaxFeatures,
            "--ngram_min", $NgramMin,
            "--ngram_max", $NgramMax,
            "--C", $CValue,
            "--subset", $GridsearchSubset,
            "--output_dir", $OutputDir,
            "--no_upload",
            "--skip_test_eval"
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
                    "  C: $CValue" | Out-File $ResultsFile -Append
                    "  Validation F1: $ValF1" | Out-File $ResultsFile -Append
                    "  Validation Accuracy: $ValAcc" | Out-File $ResultsFile -Append
                    "  Training Time: ${TrainTime}s" | Out-File $ResultsFile -Append
                    "  Output: $OutputDir" | Out-File $ResultsFile -Append
                    "" | Out-File $ResultsFile -Append
                    
                    # Update best if this is better
                    if ($ValF1 -gt $BestF1) {
                        $BestF1 = $ValF1
                        $BestConfig = "C=$CValue"
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
    
    # Save best config summary
    $BestConfigFile = Join-Path $GridsearchDir "best_config.txt"
    "Best Configuration Found" | Out-File $BestConfigFile
    "======================" | Out-File $BestConfigFile -Append
    "Configuration: $BestConfig" | Out-File $BestConfigFile -Append
    "Validation F1: $BestF1" | Out-File $BestConfigFile -Append
    "Model Directory: $BestOutputDir" | Out-File $BestConfigFile -Append
    
    Write-Host "All results saved to: $ResultsFile" -ForegroundColor Green
    Write-Host "Best configuration saved to: $BestConfigFile" -ForegroundColor Green
    
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

# Parse: "Configuration: C=1.0"
if ($ConfigLine -match "C=([0-9.]+)") {
    $BestC = [double]$matches[1]
} else {
    Write-Host "Error: Could not parse best configuration" -ForegroundColor Red
    exit 1
}

Write-Host "Extracted hyperparameters:" -ForegroundColor Yellow
Write-Host "  C: $BestC"
Write-Host "  max_features: $MaxFeatures (fixed)"
Write-Host "  ngram_range: ($NgramMin, $NgramMax) (fixed)"
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
    "--C", $BestC,
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
Write-Host "  C: $BestC"
Write-Host "  max_features: $MaxFeatures (fixed)"
Write-Host "  ngram_range: ($NgramMin, $NgramMax) (fixed)"
Write-Host ""
Write-Host "Check your HuggingFace profile for the uploaded model!" -ForegroundColor Cyan
Write-Host ""
