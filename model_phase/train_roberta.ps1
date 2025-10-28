# RoBERTa Training Script (PowerShell)
# Fine-tunes RoBERTa for sentiment analysis with automatic hyperparameter tuning via grid search
# Usage: .\model_phase\train_roberta.ps1 -Dataset "your-username/game-reviews-sentiment"

# RoBERTa parameters (constants - not tuned)
$MAX_LENGTH = 256
$BATCH_SIZE = 32
$NUM_EPOCHS = 3
$WARMUP_STEPS = 0
$WEIGHT_DECAY = 0.01

# Grid search parameters (tune learning rate for RoBERTa)
$LEARNING_RATE_VALUES = @(1e-5, 2e-5, 3e-5, 5e-5)

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
    [double]$GridSearchSubset = 0.01,
    
    [Parameter(Mandatory=$false)]
    [double]$FinalSubset = 1.0,
    
    [Parameter(Mandatory=$false)]
    [string]$OutputBaseDir = "model_phase\results",
    
    [Parameter(Mandatory=$false)]
    [switch]$UseWandb,
    
    [Parameter(Mandatory=$false)]
    [switch]$NoWandb,
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipGridSearch,
    
    [Parameter(Mandatory=$false)]
    [string]$ResumeFromCheckpoint = "auto",
    
    [Parameter(Mandatory=$false)]
    [switch]$NoCheckpoints
)

$ErrorActionPreference = "Stop"

# Validate dataset
if (-not $Dataset) {
    Write-Host "Error: Dataset is required (not found in .env or command line)" -ForegroundColor Red
    Write-Host "Usage: .\model_phase\train_roberta.ps1 -Dataset your-username/game-reviews-sentiment"
    Write-Host "Or set HF_DATASET_NAME in .env file"
    exit 1
}

# Determine WandB setting
$UseWandbFlag = (-not $NoWandb) -or $UseWandb

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "RoBERTa Model Training" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Dataset: $Dataset"
Write-Host "Grid Search Subset: $GridSearchSubset"
Write-Host "Final Training Subset: $FinalSubset"
Write-Host "Output Directory: $OutputBaseDir"
Write-Host "WandB Logging: $(if ($UseWandbFlag) { 'Enabled' } else { 'Disabled' })"
Write-Host "Checkpoints: $(if ($NoCheckpoints) { 'Disabled' } else { 'Enabled' })"
if ($ResumeFromCheckpoint) {
    Write-Host "Resume from: $ResumeFromCheckpoint"
}
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
    Write-Host "STEP 1/3: Running Grid Search" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "Finding best hyperparameters on $GridSearchSubset subset..."
    Write-Host ""
    
    $GridSearchDir = Join-Path $OutputBaseDir "gridsearch"
    
    # Results file
    $ResultsFile = Join-Path $GridSearchDir "gridsearch_results.txt"
    New-Item -ItemType Directory -Path $GridSearchDir -Force | Out-Null
    
    "Grid Search Results - $(Get-Date)" | Out-File -FilePath $ResultsFile
    "Dataset: $Dataset" | Out-File -FilePath $ResultsFile -Append
    "Subset: $GridSearchSubset" | Out-File -FilePath $ResultsFile -Append
    "==========================================" | Out-File -FilePath $ResultsFile -Append
    "" | Out-File -FilePath $ResultsFile -Append
    
    # Best results tracking
    $BestF1 = 0
    $BestConfig = ""
    $BestOutputDir = ""
    
    # Counter
    $TotalConfigs = $LEARNING_RATE_VALUES.Count
    $Current = 0
    
    Write-Host "Total configurations to test: $TotalConfigs"
    Write-Host "RoBERTa Settings (fixed): max_length=$MAX_LENGTH, batch_size=$BATCH_SIZE, epochs=$NUM_EPOCHS"
    Write-Host ""
    
    # Grid search loop
    foreach ($LrValue in $LEARNING_RATE_VALUES) {
        $Current++
        
        Write-Host "==========================================" -ForegroundColor Yellow
        Write-Host "Configuration $Current/$TotalConfigs" -ForegroundColor Yellow
        Write-Host "==========================================" -ForegroundColor Yellow
        Write-Host "Learning Rate: $LrValue"
        Write-Host ""
        
        # Create unique output directory
        $OutputDir = Join-Path $GridSearchDir "config_${Current}_LR${LrValue}"
        
        # Create experiment name for WandB
        $ExperimentName = "roberta_ex_${Current}"
        
        # Build command (no HuggingFace upload during grid search)
        $TrainArgs = @(
            "model_phase\main_roberta.py",
            "--dataset", $Dataset,
            "--max_length", $MAX_LENGTH,
            "--batch_size", $BATCH_SIZE,
            "--learning_rate", $LrValue,
            "--num_epochs", $NUM_EPOCHS,
            "--warmup_steps", $WARMUP_STEPS,
            "--weight_decay", $WEIGHT_DECAY,
            "--subset", $GridSearchSubset,
            "--output_dir", $OutputDir,
            "--no_upload",
            "--skip_test_eval",
            "--experiment_name", $ExperimentName
        )
        
        # Add wandb if specified
        if ($UseWandbFlag) {
            $TrainArgs += "--use_wandb"
        }
        
        # Add checkpoint options
        if ($NoCheckpoints) {
            $TrainArgs += "--no_checkpoints"
        }
        
        # Run training
        Write-Host "Running: python $($TrainArgs -join ' ')"
        & python $TrainArgs
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Error: Training failed for configuration $Current" -ForegroundColor Red
            "Configuration $Current`: ERROR - Training failed" | Out-File -FilePath $ResultsFile -Append
            "" | Out-File -FilePath $ResultsFile -Append
            continue
        }
        
        # Extract validation F1 score from results.json
        $ResultsJson = Join-Path $OutputDir "results.json"
        if (Test-Path $ResultsJson) {
            $Results = Get-Content $ResultsJson | ConvertFrom-Json
            $ValF1 = [math]::Round($Results.validation_f1, 4)
            $ValAcc = [math]::Round($Results.validation_accuracy, 4)
            $TrainTime = [math]::Round($Results.training_time, 2)
            
            Write-Host ""
            Write-Host "Results:"
            Write-Host "  Validation F1: $ValF1"
            Write-Host "  Validation Accuracy: $ValAcc"
            Write-Host "  Training Time: ${TrainTime}s"
            Write-Host ""
            
            # Log to results file
            "Configuration $Current`:" | Out-File -FilePath $ResultsFile -Append
            "  Learning Rate: $LrValue" | Out-File -FilePath $ResultsFile -Append
            "  Validation F1: $ValF1" | Out-File -FilePath $ResultsFile -Append
            "  Validation Accuracy: $ValAcc" | Out-File -FilePath $ResultsFile -Append
            "  Training Time: ${TrainTime}s" | Out-File -FilePath $ResultsFile -Append
            "  Output: $OutputDir" | Out-File -FilePath $ResultsFile -Append
            "" | Out-File -FilePath $ResultsFile -Append
            
            # Update best if this is better
            if ($ValF1 -gt $BestF1) {
                $BestF1 = $ValF1
                $BestConfig = "learning_rate=$LrValue"
                $BestOutputDir = $OutputDir
                $BestLearningRate = $LrValue
                Write-Host "*** NEW BEST CONFIGURATION! ***" -ForegroundColor Green
                Write-Host ""
            }
        } else {
            Write-Host "Error: Results file not found at $ResultsJson" -ForegroundColor Red
            "Configuration $Current`: ERROR - Results file not found" | Out-File -FilePath $ResultsFile -Append
            "" | Out-File -FilePath $ResultsFile -Append
        }
        
        Write-Host ""
    }
    
    # Save best config summary
    $BestConfigFile = Join-Path $GridSearchDir "best_config.txt"
    "Best Configuration Found" | Out-File -FilePath $BestConfigFile
    "======================" | Out-File -FilePath $BestConfigFile -Append
    "Configuration: $BestConfig" | Out-File -FilePath $BestConfigFile -Append
    "Validation F1: $BestF1" | Out-File -FilePath $BestConfigFile -Append
    "Model Directory: $BestOutputDir" | Out-File -FilePath $BestConfigFile -Append
    
    Write-Host "All results saved to: $ResultsFile"
    Write-Host "Best configuration saved to: $BestConfigFile"
    
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
# Parse the line: "Configuration: learning_rate=2e-5"
$ConfigLine = Get-Content $BestConfigFile | Select-String "Configuration:"
if ($ConfigLine -match 'learning_rate=([0-9e.-]+)') {
    $BestLearningRate = $matches[1]
} else {
    Write-Host "Error: Could not extract learning rate from best config" -ForegroundColor Red
    exit 1
}

Write-Host "Extracted hyperparameters:"
Write-Host "  Learning Rate: $BestLearningRate"
Write-Host "  max_length: $MAX_LENGTH (fixed)"
Write-Host "  batch_size: $BATCH_SIZE (fixed)"
Write-Host "  num_epochs: $NUM_EPOCHS (fixed)"
Write-Host ""

# Step 3: Final Training with Best Configuration
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "STEP 3/3: Final Training with Best Configuration" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Training on $FinalSubset subset with best hyperparameters..."
Write-Host "This model will be uploaded to HuggingFace Hub."
Write-Host ""

# Create experiment name for final training
$FinalExperimentName = "roberta_official_${BestLearningRate}"

# Build final training command
$FinalArgs = @(
    "model_phase\main_roberta.py",
    "--dataset", $Dataset,
    "--max_length", $MAX_LENGTH,
    "--batch_size", $BATCH_SIZE,
    "--learning_rate", $BestLearningRate,
    "--num_epochs", $NUM_EPOCHS,
    "--warmup_steps", $WARMUP_STEPS,
    "--weight_decay", $WEIGHT_DECAY,
    "--subset", $FinalSubset,
    "--experiment_name", $FinalExperimentName,
    "--resume_from_checkpoint", "auto"
)

if ($UseWandbFlag) {
    $FinalArgs += "--use_wandb"
}

# Add checkpoint options
if ($NoCheckpoints) {
    $FinalArgs += "--no_checkpoints"
}

# Note: We always use auto checkpoint finding now, no need to add explicit --resume_from_checkpoint
# unless user explicitly provided a different path via command line parameter

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
Write-Host "  Learning Rate: $BestLearningRate"
Write-Host "  max_length: $MAX_LENGTH (fixed)"
Write-Host "  batch_size: $BATCH_SIZE (fixed)"
Write-Host "  num_epochs: $NUM_EPOCHS (fixed)"
Write-Host ""
Write-Host "Check your HuggingFace profile for the uploaded model!" -ForegroundColor Cyan
Write-Host ""
