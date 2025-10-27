# RoBERTa Training Script (PowerShell)
# Fine-tunes RoBERTa for sentiment analysis on game reviews
# Usage: .\model_phase\train_roberta.ps1 -Dataset "your-username/game-reviews-sentiment"

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
    [string]$ModelName = "roberta-base",
    
    [Parameter(Mandatory=$false)]
    [int]$MaxLength = 512,
    
    [Parameter(Mandatory=$false)]
    [int]$BatchSize = 16,
    
    [Parameter(Mandatory=$false)]
    [double]$LearningRate = 0.00002,
    
    [Parameter(Mandatory=$false)]
    [int]$NumEpochs = 3,
    
    [Parameter(Mandatory=$false)]
    [int]$WarmupSteps = 0,
    
    [Parameter(Mandatory=$false)]
    [double]$WeightDecay = 0.01,
    
    [Parameter(Mandatory=$false)]
    [double]$Subset = 1.0,
    
    [Parameter(Mandatory=$false)]
    [string]$OutputDir = $null,
    
    [Parameter(Mandatory=$false)]
    [switch]$UseWandb = $true,
    
    [Parameter(Mandatory=$false)]
    [switch]$NoWandb,
    
    [Parameter(Mandatory=$false)]
    [switch]$NoUpload,
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipTestEval
)

$ErrorActionPreference = "Stop"

# Validate dataset
if (-not $Dataset) {
    Write-Host "Error: --dataset is required (not found in .env or command line)" -ForegroundColor Red
    Write-Host "Usage: .\model_phase\train_roberta.ps1 -Dataset your-username/game-reviews-sentiment"
    Write-Host "Or set HF_DATASET_NAME in .env file"
    exit 1
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "RoBERTa Fine-tuning for Sentiment Analysis" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Dataset: $Dataset"
Write-Host "  Model: $ModelName"
Write-Host "  Max Length: $MaxLength"
Write-Host "  Batch Size: $BatchSize"
Write-Host "  Learning Rate: $LearningRate"
Write-Host "  Epochs: $NumEpochs"
Write-Host "  Warmup Steps: $WarmupSteps"
Write-Host "  Weight Decay: $WeightDecay"
Write-Host "  Data Subset: $Subset"
if ($OutputDir) {
    Write-Host "  Output Directory: $OutputDir"
}
Write-Host ""

# Check for GPU
Write-Host "Checking for GPU availability..." -ForegroundColor Cyan
try {
    $gpuCheck = python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>&1
    Write-Host $gpuCheck -ForegroundColor Green
} catch {
    Write-Host "Warning: Could not check GPU availability" -ForegroundColor Yellow
}
Write-Host ""

# Build training command
$TrainArgs = @(
    "model_phase\main_roberta.py",
    "--dataset", $Dataset,
    "--model_name", $ModelName,
    "--max_length", $MaxLength,
    "--batch_size", $BatchSize,
    "--learning_rate", $LearningRate,
    "--num_epochs", $NumEpochs,
    "--warmup_steps", $WarmupSteps,
    "--weight_decay", $WeightDecay,
    "--subset", $Subset
)

if ($OutputDir) {
    $TrainArgs += @("--output_dir", $OutputDir)
}

# Enable WandB by default unless explicitly disabled
if (-not $NoWandb) {
    $TrainArgs += "--use_wandb"
    Write-Host "  WandB Logging: Enabled" -ForegroundColor Green
} else {
    Write-Host "  WandB Logging: Disabled" -ForegroundColor Yellow
}

if ($NoUpload) {
    $TrainArgs += "--no_upload"
}

if ($SkipTestEval) {
    $TrainArgs += "--skip_test_eval"
}

Write-Host "============================================================" -ForegroundColor Green
Write-Host "Starting Training" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""

# Run training
& python $TrainArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Error: Training failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Training Complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Model trained successfully!" -ForegroundColor Cyan
if (-not $NoUpload) {
    Write-Host "Check your HuggingFace profile for the uploaded model!" -ForegroundColor Cyan
}
Write-Host ""
