# BGE-M3 Training Script (PowerShell)
# Trains BGE-M3 model with automatic hyperparameter tuning via grid search
# Usage: .\model_phase\train_bge_m3.ps1 -Dataset "your-username/game-reviews-sentiment"

# BGE-M3 parameters (constants - not tuned)
$MAX_LENGTH = 512
$BATCH_SIZE = 32
$KERNEL = "rbf"

# Grid search parameters (tune C and gamma for RBF kernel)
$C_VALUES = @(0.1, 1, 3, 10)
$GAMMA_VALUES = @(0.125, 0.25, 0.5)

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
    [double]$GridSearchSubset = 0.1,
    
    [Parameter(Mandatory=$false)]
    [double]$FinalSubset = 1.0,
    
    [Parameter(Mandatory=$false)]
    [string]$OutputBaseDir = "model_phase\results",
    
    [Parameter(Mandatory=$false)]
    [switch]$UseWandb,
    
    [Parameter(Mandatory=$false)]
    [switch]$NoWandb,
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipGridSearch
)

$ErrorActionPreference = "Stop"

# Validate dataset
if (-not $Dataset) {
    Write-Host "Error: Dataset is required (not found in .env or command line)" -ForegroundColor Red
    Write-Host "Usage: .\model_phase\train_bge_m3.ps1 -Dataset your-username/game-reviews-sentiment"
    exit 1
}

# Determine WandB setting
$UseWandbFlag = (-not $NoWandb) -or $UseWandb

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "BGE-M3 Model Training" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Dataset: $Dataset"
Write-Host "Grid Search Subset: $GridSearchSubset"
Write-Host "Final Training Subset: $FinalSubset"
Write-Host "WandB Logging: $(if ($UseWandbFlag) { 'Enabled' } else { 'Disabled' })"
Write-Host ""

# Check for GPU
Write-Host "Checking for GPU availability..." -ForegroundColor Cyan
try {
    $gpuCheck = python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>&1
    Write-Host $gpuCheck
} catch {
    Write-Host "Warning: Could not check GPU availability" -ForegroundColor Yellow
}
Write-Host ""

# Step 1: Grid Search (if not skipped)
if (-not $SkipGridSearch) {
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "STEP 1/3: Running Efficient Grid Search" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "Finding best hyperparameters on $GridSearchSubset subset..."
    Write-Host "(Embeddings will be loaded ONCE, then SVM trained multiple times)"
    Write-Host ""
    
    $GridSearchDir = Join-Path $OutputBaseDir "gridsearch"
    
    # Build grid search command
    $GridSearchArgs = @(
        "model_phase\main_bge_m3.py",
        "--grid_search",
        "--dataset", $Dataset,
        "--max_length", $MAX_LENGTH,
        "--batch_size", $BATCH_SIZE,
        "--kernel", $KERNEL,
        "--subset", $GridSearchSubset,
        "--output_dir", $GridSearchDir,
        "--C_values"
    )
    $GridSearchArgs += $C_VALUES
    $GridSearchArgs += "--gamma_values"
    $GridSearchArgs += $GAMMA_VALUES
    
    Write-Host "Running: python $($GridSearchArgs -join ' ')"
    Write-Host ""
    
    & python $GridSearchArgs
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Grid search failed" -ForegroundColor Red
        exit 1
    }
    
    Write-Host ""
    Write-Host "✓ Grid search complete!" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "STEP 1/3: Skipping Grid Search" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "Using provided hyperparameters..."
    Write-Host ""
    $GridSearchDir = Join-Path $OutputBaseDir "gridsearch"
}

# Step 2: Extract Best Configuration
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "STEP 2/3: Extracting Best Configuration" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$BestConfigFile = Join-Path $GridSearchDir "best_config.txt"

if (-not (Test-Path $BestConfigFile)) {
    Write-Host "Error: Best config file not found at $BestConfigFile" -ForegroundColor Red
    Write-Host "Make sure grid search completed successfully."
    exit 1
}

Write-Host "Reading best configuration from: $BestConfigFile"
Write-Host ""
Get-Content $BestConfigFile
Write-Host ""

# Extract hyperparameters from best config
$ConfigLine = Get-Content $BestConfigFile | Select-String "Configuration:"
if ($ConfigLine -match 'C=([0-9.e+-]+)') {
    $BestC = $matches[1]
} else {
    Write-Host "Error: Could not extract C from best config" -ForegroundColor Red
    exit 1
}
if ($ConfigLine -match 'gamma=([a-z0-9.e+-]+)') {
    $BestGamma = $matches[1]
} else {
    Write-Host "Error: Could not extract gamma from best config" -ForegroundColor Red
    exit 1
}

Write-Host "Extracted hyperparameters:"
Write-Host "  C: $BestC"
Write-Host "  gamma: $BestGamma"
Write-Host "  kernel: $KERNEL (fixed)"
Write-Host "  max_length: $MAX_LENGTH (fixed)"
Write-Host "  batch_size: $BATCH_SIZE (fixed)"
Write-Host ""

# Step 3: Final Training with Best Configuration
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "STEP 3/3: Final Training with Best Configuration" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Training on $FinalSubset subset with best hyperparameters..."
Write-Host "This model will be uploaded to HuggingFace Hub."
Write-Host ""

# Create experiment name for final training
$FinalExperimentName = "bge_m3_official_${BestC}"

# Build final training command
$FinalArgs = @(
    "model_phase\main_bge_m3.py",
    "--dataset", $Dataset,
    "--max_length", $MAX_LENGTH,
    "--batch_size", $BATCH_SIZE,
    "--C", $BestC,
    "--gamma", $BestGamma,
    "--kernel", $KERNEL,
    "--subset", $FinalSubset,
    "--experiment_name", $FinalExperimentName
)

if ($UseWandbFlag) {
    $FinalArgs += "--use_wandb"
}

Write-Host "Running: python $($FinalArgs -join ' ')"
Write-Host ""

& python $FinalArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Error: Final training failed" -ForegroundColor Red
    exit 1
}

# Step 4: Summary
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "PIPELINE COMPLETE!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Summary:"
Write-Host "1. ✓ Grid search found best hyperparameters"
Write-Host "2. ✓ Final model trained with optimal configuration"
Write-Host "3. ✓ Results uploaded to HuggingFace Hub"
Write-Host ""
Write-Host "Best Configuration Used:"
Write-Host "  C: $BestC"
Write-Host "  gamma: $BestGamma"
Write-Host "  kernel: $KERNEL (fixed)"
Write-Host "  max_length: $MAX_LENGTH (fixed)"
Write-Host "  batch_size: $BATCH_SIZE (fixed)"
Write-Host ""
Write-Host "Check your HuggingFace profile for the uploaded model!" -ForegroundColor Cyan
Write-Host ""
