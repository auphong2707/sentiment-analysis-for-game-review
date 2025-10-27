"""
RoBERTa Model: Fine-tuning for Sentiment Analysis

Core Idea: Sentiment is best understood by looking at a word's context from both 
left and right simultaneously, using a deeply pre-trained transformer model.

How it Works:
- Uses RoBERTa (Robustly Optimized BERT Pretraining Approach), a model pre-trained 
  on massive amounts of text
- Fine-tunes this model on game reviews, adapting its powerful language understanding
- Processes entire review bidirectionally, understanding context from all directions

Key Features:
- State-of-the-art accuracy with bidirectional context
- Deep understanding of nuance and complex sentiment
- Handles negations and context-dependent meanings
- GPU acceleration for faster training
"""

import sys
import os
from pathlib import Path
import argparse
import json
import time
from datetime import datetime

# Set environment variable to disable chat template checking BEFORE importing anything
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import wandb (optional dependency)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Add project root to Python path
current_file = Path(__file__).absolute()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from torch.utils.data import Dataset

# Monkey-patch BEFORE importing transformers
from huggingface_hub import hf_api


# Now safe to import transformers
from transformers import (
    AutoTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import numpy as np

# Import utilities
from model_phase.utilities import (
    load_dataset_from_hf,
    setup_output_directory,
    init_wandb_if_available,
    log_to_wandb,
    finish_wandb,
    save_results_to_json,
    print_training_summary,
    upload_results_to_hf
)

# Constants
MODEL_NAME = 'FacebookAI/roberta-base'


class GameReviewDataset(Dataset):
    """PyTorch Dataset for game reviews."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512, label2id=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create label mapping
        if label2id is None:
            unique_labels = sorted(set(labels))
            self.label2id = {label: idx for idx, label in enumerate(unique_labels)}
            self.id2label = {idx: label for label, idx in self.label2id.items()}
        else:
            self.label2id = label2id
            self.id2label = {idx: label for label, idx in label2id.items()}
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.label2id[label], dtype=torch.long)
        }


class WandbCallback(TrainerCallback):
    """Custom callback for WandB logging."""
    
    def __init__(self, use_wandb=False):
        self.use_wandb = use_wandb
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics to WandB."""
        if self.use_wandb and WANDB_AVAILABLE and logs:
            # Only log if wandb is initialized
            try:
                wandb.log(logs, step=state.global_step)
            except Exception:
                pass


class RoBERTaSentimentClassifier:
    """
    Sentiment classifier using fine-tuned RoBERTa with HuggingFace Trainer.
    """
    
    def __init__(self, 
                 num_labels=3,
                 max_length=512,
                 batch_size=16,
                 learning_rate=2e-5,
                 num_epochs=3,
                 warmup_steps=0,
                 weight_decay=0.01,
                 output_dir=None,
                 checkpoint_dir=None):
        """
        Initialize the classifier.
        
        Args:
            num_labels: Number of sentiment classes
            max_length: Maximum sequence length
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            num_epochs: Number of training epochs
            warmup_steps: Number of warmup steps for learning rate scheduler
            weight_decay: Weight decay for optimizer
            output_dir: Directory to save model outputs
            checkpoint_dir: Directory to save checkpoints during training
        """
        self.model_name = MODEL_NAME
        self.num_labels = num_labels
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        
        # Device info
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {device}")
        if device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Initialize tokenizer
        print(f"\nLoading tokenizer: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            use_fast=True
        )
        
        self.model = None  # Will be initialized when we know label mapping
        self.label2id = None
        self.id2label = None
        self.trainer = None
        
    def _create_model(self, label2id, id2label):
        """Create model with proper label mappings."""
        self.label2id = label2id
        self.id2label = id2label
        
        # Use fallback model name if needed
        model_name_to_load = self.model_name.replace("FacebookAI/", "")
        
        print(f"\nLoading model: {model_name_to_load}")
        self.model = RobertaForSequenceClassification.from_pretrained(
            model_name_to_load,
            num_labels=self.num_labels,
            label2id=label2id,
            id2label=id2label
        )
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
    
    def fit(self, train_texts, train_labels, val_texts, val_labels, use_wandb=False, resume_from_checkpoint=None, save_checkpoints=True):
        """
        Train the model on text data using HuggingFace Trainer.
        
        Args:
            train_texts: List of training review texts
            train_labels: List of training sentiment labels
            val_texts: List of validation review texts
            val_labels: List of validation sentiment labels
            use_wandb: Whether to log metrics to WandB
            resume_from_checkpoint: Path to checkpoint directory to resume training from
            save_checkpoints: Whether to save checkpoints during training
        """
        print("\n" + "="*60)
        print("Preparing Data")
        print("="*60)
        
        # Create datasets
        train_dataset = GameReviewDataset(
            train_texts, train_labels, self.tokenizer, self.max_length
        )
        val_dataset = GameReviewDataset(
            val_texts, val_labels, self.tokenizer, self.max_length, 
            label2id=train_dataset.label2id
        )
        
        # Create model with proper label mappings
        if self.model is None:
            self._create_model(train_dataset.label2id, train_dataset.id2label)
        
        print(f"\nTraining samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Batch size: {self.batch_size}")
        
        # Setup output directory
        if self.output_dir is None:
            self.output_dir = Path("model_phase/results/roberta_trainer")
        else:
            self.output_dir = Path(self.output_dir)
        
        # Setup checkpoint directory
        if save_checkpoints and self.checkpoint_dir:
            checkpoint_output_dir = self.checkpoint_dir
        elif save_checkpoints:
            checkpoint_output_dir = str(self.output_dir / "checkpoints")
        else:
            checkpoint_output_dir = None
        
        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch" if save_checkpoints else "no",
            save_total_limit=3 if save_checkpoints else None,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to="wandb" if (use_wandb and WANDB_AVAILABLE) else "none",
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
            dataloader_num_workers=0,
            disable_tqdm=False,
            save_safetensors=True,
        )
        
        print("\n" + "="*60)
        print("Training Configuration")
        print("="*60)
        print(f"Epochs: {self.num_epochs}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Batch size: {self.batch_size}")
        print(f"Warmup steps: {self.warmup_steps}")
        print(f"Weight decay: {self.weight_decay}")
        print(f"Output directory: {self.output_dir}")
        if save_checkpoints:
            print(f"Checkpoint saving: Enabled")
        print(f"Mixed precision (FP16): {torch.cuda.is_available()}")
        
        # Setup callbacks
        callbacks = []
        if use_wandb and WANDB_AVAILABLE:
            callbacks.append(WandbCallback(use_wandb=True))
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks,
        )
        
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        
        # Train the model
        train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Get training metrics
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        
        # Evaluate on validation set
        print("\n" + "="*60)
        print("Final Validation")
        print("="*60)
        eval_metrics = self.trainer.evaluate()
        self.trainer.log_metrics("eval", eval_metrics)
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Best Validation F1: {eval_metrics.get('eval_f1', 0):.4f}")
        
        # Convert metrics to training stats format
        training_stats = [{
            'train_loss': metrics.get('train_loss', 0),
            'eval_loss': eval_metrics.get('eval_loss', 0),
            'eval_accuracy': eval_metrics.get('eval_accuracy', 0),
            'eval_f1': eval_metrics.get('eval_f1', 0),
        }]
        
        return training_stats
    
    def predict(self, texts):
        """
        Predict sentiment for new texts.
        
        Args:
            texts: List of review texts
            
        Returns:
            List of predicted sentiment labels
        """
        if self.trainer is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Create dataset with dummy labels
        dataset = GameReviewDataset(
            texts,
            ['positive'] * len(texts),  # Dummy labels
            self.tokenizer,
            self.max_length,
            label2id=self.label2id
        )
        
        # Use trainer to predict
        predictions = self.trainer.predict(dataset)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        
        # Convert to label strings
        predicted_labels = [self.id2label[pred] for pred in pred_labels]
        return predicted_labels
    
    def predict_proba(self, texts):
        """
        Predict sentiment probabilities for new texts.
        
        Args:
            texts: List of review texts
            
        Returns:
            Array of shape (n_samples, n_classes) with probabilities
        """
        if self.trainer is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Create dataset with dummy labels
        dataset = GameReviewDataset(
            texts,
            ['positive'] * len(texts),  # Dummy labels
            self.tokenizer,
            self.max_length,
            label2id=self.label2id
        )
        
        # Use trainer to predict
        predictions = self.trainer.predict(dataset)
        
        # Apply softmax to get probabilities
        probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()
        
        return probs
    
    def save(self, output_dir):
        """Save model, tokenizer, and configuration."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving model to {output_dir}")
        
        if self.trainer is not None:
            # Use trainer's save method
            self.trainer.save_model(str(output_dir))
        else:
            # Fallback to direct save
            self.model.save_pretrained(output_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save label mappings
        label_mapping = {
            'label2id': self.label2id,
            'id2label': self.id2label
        }
        with open(output_dir / 'label_mapping.json', 'w') as f:
            json.dump(label_mapping, f, indent=2)
        
        print(f"  ✓ Model saved")
        print(f"  ✓ Tokenizer saved")
        print(f"  ✓ Label mapping saved")
    
    @classmethod
    def load(cls, output_dir):
        """Load model from directory."""
        output_dir = Path(output_dir)
        
        with open(output_dir / 'config.json', 'r') as f:
            config = json.load(f)
        
        model = cls(
            model_name=config['model_name'],
            num_labels=config['num_labels'],
            max_length=config['max_length'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            num_epochs=config['num_epochs'],
            warmup_steps=config['warmup_steps'],
            weight_decay=config['weight_decay']
        )
        
        model.tokenizer = AutoTokenizer.from_pretrained(output_dir / 'tokenizer', use_fast=True)
        model.model = RobertaForSequenceClassification.from_pretrained(output_dir / 'model')
        model.model.to(model.device)
        model.label2id = config['label2id']
        model.id2label = {int(k): v for k, v in config['id2label'].items()}
        
        print(f"✓ Model loaded from {output_dir}")
        return model


def evaluate_classifier(model, texts, labels, split_name="Test"):
    """
    Evaluate a RoBERTa classifier and return comprehensive metrics.
    
    Args:
        model: Trained RoBERTa classifier
        texts: List of text samples
        labels: True labels
        split_name: Name of the split for display
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating on {split_name} set")
    print(f"{'='*60}")
    
    # Predict
    start_time = time.time()
    predictions = model.predict(texts)
    inference_time = time.time() - start_time
    
    # Convert string labels to indices for sklearn metrics
    unique_labels = sorted(set(labels) | set(predictions))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    labels_idx = [label_to_idx[label] for label in labels]
    predictions_idx = [label_to_idx[pred] for pred in predictions]
    
    # Calculate metrics
    accuracy = accuracy_score(labels_idx, predictions_idx)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels_idx, predictions_idx, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    class_report = classification_report(
        labels, predictions, output_dict=True, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions, labels=unique_labels)
    
    # Print results
    print(f"\n{split_name} Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision (weighted): {precision:.4f}")
    print(f"  Recall (weighted): {recall:.4f}")
    print(f"  F1-score (weighted): {f1:.4f}")
    print(f"  Inference time: {inference_time:.2f}s")
    print(f"  Samples/second: {len(texts)/inference_time:.2f}")
    
    print(f"\nPer-class metrics:")
    for class_name in unique_labels:
        if class_name in class_report:
            metrics = class_report[class_name]
            print(f"  {class_name.capitalize()}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1-score: {metrics['f1-score']:.4f}")
            print(f"    Support: {int(metrics['support'])}")
    
    print(f"\nConfusion Matrix:")
    print(f"  Classes: {unique_labels}")
    print(cm)
    
    # Compile results
    results = {
        f'{split_name.lower()}_accuracy': float(accuracy),
        f'{split_name.lower()}_precision': float(precision),
        f'{split_name.lower()}_recall': float(recall),
        f'{split_name.lower()}_f1': float(f1),
        f'{split_name.lower()}_inference_time': float(inference_time),
        f'{split_name.lower()}_samples_per_second': float(len(texts)/inference_time),
        f'{split_name.lower()}_classification_report': class_report,
        f'{split_name.lower()}_confusion_matrix': cm.tolist()
    }
    
    return results


def main(dataset_name,
         max_length=512,
         batch_size=16,
         learning_rate=2e-5,
         num_epochs=3,
         warmup_steps=0,
         weight_decay=0.01,
         subset=1.0,
         output_dir=None,
         use_wandb=False,
         upload_to_hf=True,
         hf_repo=None,
         skip_test_eval=False,
         save_checkpoints=True,
         resume_from_checkpoint=None):
    """
    Main training and evaluation pipeline for RoBERTa.
    
    Args:
        dataset_name: HuggingFace dataset name
        max_length: Maximum sequence length
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        warmup_steps: Warmup steps for learning rate scheduler
        weight_decay: Weight decay for optimizer
        subset: Fraction of data to use
        output_dir: Directory to save results
        use_wandb: Whether to use WandB for tracking
        upload_to_hf: Whether to upload results to HuggingFace Hub
        hf_repo: HuggingFace repository name for results
        skip_test_eval: Whether to skip test set evaluation (for grid search)
        save_checkpoints: Whether to save training checkpoints
        resume_from_checkpoint: Path to checkpoint file to resume training from
    """
    print("\n" + "="*60)
    print("RoBERTa Fine-tuning for Sentiment Analysis")
    print("="*60)
    
    # Setup output directory
    output_dir = setup_output_directory(output_dir, model_name="roberta")
    
    # Initialize WandB if requested
    wandb_initialized = False
    if use_wandb and WANDB_AVAILABLE:
        try:
            # Get project name from .env or use default
            wandb_project = os.getenv('WANDB_PROJECT', 'game-review-sentiment')
            
            wandb.init(
                project=wandb_project,
                name=f"roberta_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "model": MODEL_NAME,
                    "max_length": max_length,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                    "warmup_steps": warmup_steps,
                    "weight_decay": weight_decay,
                    "subset": subset
                }
            )
            wandb_initialized = True
            print(f"✓ WandB initialized - Project: {wandb_project}")
        except Exception as e:
            print(f"⚠️  Could not initialize WandB: {e}")
            wandb_initialized = False
    elif use_wandb and not WANDB_AVAILABLE:
        print("⚠️  WandB not available. Install with: pip install wandb")
        wandb_initialized = False
    else:
        print("ℹ️  WandB logging disabled")
    
    # Load data
    train_data, val_data, test_data = load_dataset_from_hf(
        dataset_name,
        subset_percentage=subset
    )
    
    # Initialize model
    print(f"\n{'='*60}")
    print("Initializing RoBERTa model")
    print(f"{'='*60}")
    
    # Set checkpoint directory if checkpoints are enabled
    checkpoint_dir = output_dir / 'checkpoints' if save_checkpoints else None
    
    model = RoBERTaSentimentClassifier(
        num_labels=3,
        max_length=max_length,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir
    )
    
    # Train model
    train_start = time.time()
    training_stats = model.fit(
        train_data['text'],
        train_data['label'],
        val_data['text'],
        val_data['label'],
        use_wandb=wandb_initialized,
        resume_from_checkpoint=resume_from_checkpoint,
        save_checkpoints=save_checkpoints
    )
    train_time = time.time() - train_start
    
    print(f"\n✓ Training completed in {train_time:.2f}s ({train_time/60:.2f} minutes)")
    
    # Evaluate on validation set (final metrics)
    val_results = evaluate_classifier(
        model, val_data['text'], val_data['label'], "Validation"
    )
    
    # Conditionally evaluate on test set
    if skip_test_eval:
        print(f"\n{'='*60}")
        print("Skipping test set evaluation (grid search mode)")
        print(f"{'='*60}")
        # Create empty test results for consistency
        test_results = {
            'test_accuracy': 0.0,
            'test_precision': 0.0,
            'test_recall': 0.0,
            'test_f1': 0.0,
            'test_inference_time': 0.0,
            'test_samples_per_second': 0.0,
            'test_classification_report': {},
            'test_confusion_matrix': []
        }
    else:
        # Evaluate on test set
        test_results = evaluate_classifier(
            model, test_data['text'], test_data['label'], "Test"
        )
    
    # Log final results to wandb
    if wandb_initialized and WANDB_AVAILABLE:
        # Always log validation results
        wandb_metrics = {
            'final_val_accuracy': val_results['validation_accuracy'],
            'final_val_f1': val_results['validation_f1'],
            'final_val_precision': val_results['validation_precision'],
            'final_val_recall': val_results['validation_recall'],
            'training_time_seconds': train_time,
            'training_time_minutes': train_time / 60
        }
        
        # Only log test results if we evaluated on test set
        if not skip_test_eval:
            wandb_metrics.update({
                'final_test_accuracy': test_results['test_accuracy'],
                'final_test_f1': test_results['test_f1'],
                'final_test_precision': test_results['test_precision'],
                'final_test_recall': test_results['test_recall']
            })
        
        log_to_wandb(wandb_metrics, use_wandb=True)
        
        # Log confusion matrix and additional metrics to wandb (only if test was evaluated)
        if not skip_test_eval:
            try:
                # Create confusion matrix data for test set
                class_names = ['negative', 'mixed', 'positive']
                cm = np.array(test_results['test_confusion_matrix'])
                
                # Create a table representation of the confusion matrix
                cm_data = []
                for i, true_label in enumerate(class_names):
                    row = [true_label] + [int(cm[i][j]) for j in range(len(class_names))]
                    cm_data.append(row)
                
                cm_table = wandb.Table(
                    data=cm_data,
                    columns=["True \\ Predicted"] + class_names
                )
                
                # Log the confusion matrix table
                wandb.log({"test_confusion_matrix_table": cm_table})
                
                # Also log as a simple nested list for easy access
                wandb.log({"test_confusion_matrix_raw": cm.tolist()})
                
                # Log per-class metrics in a single call
                if 'test_classification_report' in test_results:
                    class_report = test_results['test_classification_report']
                    per_class_metrics = {}
                    for class_name in ['positive', 'mixed', 'negative']:
                        if class_name in class_report:
                            metrics = class_report[class_name]
                            per_class_metrics[f'test_{class_name}_precision'] = metrics['precision']
                            per_class_metrics[f'test_{class_name}_recall'] = metrics['recall']
                            per_class_metrics[f'test_{class_name}_f1'] = metrics['f1-score']
                            per_class_metrics[f'test_{class_name}_support'] = metrics['support']
                    
                    if per_class_metrics:
                        wandb.log(per_class_metrics)
                
                # Set summary metrics (these persist across runs)
                if training_stats and len(training_stats) > 0:
                    wandb.summary['best_val_f1'] = max([stat['f1'] for stat in training_stats])
                wandb.summary['final_test_f1'] = test_results['test_f1']
                wandb.summary['final_test_accuracy'] = test_results['test_accuracy']
                
            except Exception as e:
                print(f"⚠️  Could not log additional metrics to wandb: {e}")
    
    # Compile all results
    all_results = {
        'model_config': {
            'model_name': MODEL_NAME,
            'max_length': max_length,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'warmup_steps': warmup_steps,
            'weight_decay': weight_decay,
            'subset': subset
        },
        'training_time': train_time,
        'training_stats': training_stats,
        'dataset_info': {
            'train_size': len(train_data['text']),
            'val_size': len(val_data['text']),
            'test_size': len(test_data['text'])
        },
        **val_results,
        **test_results
    }
    
    # Save results and model
    save_results_to_json(all_results, output_dir / 'results.json')
    model.save(output_dir)
    
    # Finish WandB run
    if wandb_initialized and WANDB_AVAILABLE:
        try:
            wandb.finish()
            print("✓ WandB run finished")
        except Exception as e:
            print(f"⚠️  Error finishing WandB run: {e}")
    
    # Print summary
    print_training_summary(all_results, output_dir)
    
    # Upload to HuggingFace if requested
    if upload_to_hf:
        upload_results_to_hf(
            results=all_results,
            output_dir=output_dir,
            model_name="roberta",
            hf_repo_name=hf_repo
        )
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train RoBERTa for sentiment analysis'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=os.getenv('HF_DATASET_NAME'),
        help='HuggingFace dataset name (default: from .env HF_DATASET_NAME)'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='Maximum sequence length (default: 512)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for training (default: 16)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=2e-5,
        help='Learning rate (default: 2e-5)'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=3,
        help='Number of training epochs (default: 3)'
    )
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=0,
        help='Warmup steps for learning rate scheduler (default: 0)'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.01,
        help='Weight decay for optimizer (default: 0.01)'
    )
    parser.add_argument(
        '--subset',
        type=float,
        default=1.0,
        help='Fraction of training data to use (0 to 1, default: 1.0)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for results (default: auto-generated)'
    )
    parser.add_argument(
        '--use_wandb',
        action='store_true',
        help='Use WandB for experiment tracking'
    )
    parser.add_argument(
        '--upload_to_hf',
        action='store_true',
        default=True,
        help='Upload results to HuggingFace Hub (default: True)'
    )
    parser.add_argument(
        '--no_upload',
        action='store_true',
        help='Skip uploading results to HuggingFace Hub'
    )
    parser.add_argument(
        '--hf_repo',
        type=str,
        default=None,
        help='HuggingFace repository name for results (default: auto-generated)'
    )
    parser.add_argument(
        '--skip_test_eval',
        action='store_true',
        help='Skip test set evaluation (useful for grid search)'
    )
    parser.add_argument(
        '--save_checkpoints',
        action='store_true',
        default=True,
        help='Save training checkpoints (default: True)'
    )
    parser.add_argument(
        '--no_checkpoints',
        action='store_true',
        help='Disable checkpoint saving'
    )
    parser.add_argument(
        '--resume_from_checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint file to resume training from'
    )
    
    args = parser.parse_args()
    
    # Validate dataset name
    if not args.dataset:
        parser.error("--dataset is required (or set HF_DATASET_NAME in .env)")
    
    # Determine upload setting
    upload_to_hf = args.upload_to_hf and not args.no_upload
    
    # Determine checkpoint setting
    save_checkpoints = args.save_checkpoints and not args.no_checkpoints
    
    main(
        dataset_name=args.dataset,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        subset=args.subset,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        upload_to_hf=upload_to_hf,
        hf_repo=args.hf_repo,
        skip_test_eval=args.skip_test_eval,
        save_checkpoints=save_checkpoints,
        resume_from_checkpoint=args.resume_from_checkpoint
    )
