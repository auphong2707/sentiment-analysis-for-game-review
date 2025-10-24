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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to Python path
current_file = Path(__file__).absolute()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

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


class GameReviewDataset(Dataset):
    """PyTorch Dataset for game reviews."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create label mapping
        unique_labels = sorted(set(labels))
        self.label2id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        
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


class RoBERTaSentimentClassifier:
    """
    Sentiment classifier using fine-tuned RoBERTa.
    """
    
    def __init__(self, 
                 model_name='roberta-base',
                 num_labels=3,
                 max_length=512,
                 batch_size=16,
                 learning_rate=2e-5,
                 num_epochs=3,
                 warmup_steps=0,
                 weight_decay=0.01,
                 device=None):
        """
        Initialize the classifier.
        
        Args:
            model_name: HuggingFace model name (default: roberta-base)
            num_labels: Number of sentiment classes
            max_length: Maximum sequence length
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            num_epochs: Number of training epochs
            warmup_steps: Number of warmup steps for learning rate scheduler
            weight_decay: Weight decay for optimizer
            device: Device to use (cuda/cpu), auto-detected if None
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"\nUsing device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Initialize tokenizer and model
        print(f"\nLoading tokenizer and model: {model_name}")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = None  # Will be initialized when we know label mapping
        self.label2id = None
        self.id2label = None
        
    def _create_model(self, label2id, id2label):
        """Create model with proper label mappings."""
        self.label2id = label2id
        self.id2label = id2label
        
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            label2id=label2id,
            id2label=id2label
        )
        self.model.to(self.device)
        
    def fit(self, train_texts, train_labels, val_texts, val_labels):
        """
        Train the model on text data.
        
        Args:
            train_texts: List of training review texts
            train_labels: List of training sentiment labels
            val_texts: List of validation review texts
            val_labels: List of validation sentiment labels
        """
        print("\n" + "="*60)
        print("Preparing Data")
        print("="*60)
        
        # Create datasets
        train_dataset = GameReviewDataset(
            train_texts, train_labels, self.tokenizer, self.max_length
        )
        val_dataset = GameReviewDataset(
            val_texts, val_labels, self.tokenizer, self.max_length
        )
        
        # Create model with proper label mappings
        if self.model is None:
            self._create_model(train_dataset.label2id, train_dataset.id2label)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        print(f"\nTraining samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Total training batches: {len(train_loader)}")
        
        # Prepare optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        total_steps = len(train_loader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        
        print("\n" + "="*60)
        print("Training Model")
        print("="*60)
        print(f"Epochs: {self.num_epochs}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Total steps: {total_steps}")
        
        # Training loop
        best_val_f1 = 0
        training_stats = []
        
        for epoch in range(self.num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print(f"{'='*60}")
            
            # Training phase
            train_loss = self._train_epoch(train_loader, optimizer, scheduler)
            
            # Validation phase
            val_metrics = self._evaluate(val_loader, "Validation")
            
            # Track stats
            epoch_stats = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                **val_metrics
            }
            training_stats.append(epoch_stats)
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Val F1: {val_metrics['f1']:.4f}")
            
            # Save best model
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                print(f"  ✓ New best validation F1: {best_val_f1:.4f}")
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Best Validation F1: {best_val_f1:.4f}")
        
        return training_stats
    
    def _train_epoch(self, train_loader, optimizer, scheduler):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def _evaluate(self, data_loader, split_name="Validation"):
        """Evaluate on a dataset."""
        self.model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc=f"Evaluating {split_name}")
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'loss': float(total_loss / len(data_loader))
        }
        
        return metrics
    
    def predict(self, texts):
        """
        Predict sentiment for new texts.
        
        Args:
            texts: List of review texts
            
        Returns:
            List of predicted sentiment labels
        """
        self.model.eval()
        
        dataset = GameReviewDataset(
            texts,
            ['positive'] * len(texts),  # Dummy labels
            self.tokenizer,
            self.max_length
        )
        dataset.label2id = self.label2id
        dataset.id2label = self.id2label
        
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        all_predictions = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
        
        # Convert to labels
        predicted_labels = [self.id2label[pred] for pred in all_predictions]
        return predicted_labels
    
    def predict_proba(self, texts):
        """
        Predict probability for each class.
        
        Args:
            texts: List of review texts
            
        Returns:
            Numpy array of probabilities (samples x classes)
        """
        self.model.eval()
        
        dataset = GameReviewDataset(
            texts,
            ['positive'] * len(texts),  # Dummy labels
            self.tokenizer,
            self.max_length
        )
        dataset.label2id = self.label2id
        dataset.id2label = self.id2label
        
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        all_probas = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting probabilities"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                probas = torch.softmax(logits, dim=-1)
                all_probas.extend(probas.cpu().numpy())
        
        return np.array(all_probas)
    
    def save(self, output_dir):
        """Save model, tokenizer, and configuration."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(output_dir / 'model')
        self.tokenizer.save_pretrained(output_dir / 'tokenizer')
        
        # Save config
        config = {
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'warmup_steps': self.warmup_steps,
            'weight_decay': self.weight_decay,
            'label2id': self.label2id,
            'id2label': self.id2label
        }
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Model saved to {output_dir}")
    
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
        
        model.tokenizer = RobertaTokenizer.from_pretrained(output_dir / 'tokenizer')
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
         model_name='roberta-base',
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
         hf_repo=None):
    """
    Main training and evaluation pipeline for RoBERTa.
    
    Args:
        dataset_name: HuggingFace dataset name
        model_name: Pre-trained model name (default: roberta-base)
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
    """
    print("\n" + "="*60)
    print("RoBERTa Fine-tuning for Sentiment Analysis")
    print("="*60)
    
    # Setup output directory
    output_dir = setup_output_directory(output_dir, model_name="roberta")
    
    # Initialize WandB if requested
    wandb_initialized = init_wandb_if_available(
        project_name="game-review-sentiment",
        experiment_name=f"roberta_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "model": model_name,
            "max_length": max_length,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "warmup_steps": warmup_steps,
            "weight_decay": weight_decay,
            "subset": subset
        },
        use_wandb=use_wandb
    )
    
    # Load data
    train_data, val_data, test_data = load_dataset_from_hf(
        dataset_name,
        subset_percentage=subset
    )
    
    # Initialize model
    print(f"\n{'='*60}")
    print("Initializing RoBERTa model")
    print(f"{'='*60}")
    model = RoBERTaSentimentClassifier(
        model_name=model_name,
        num_labels=3,
        max_length=max_length,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay
    )
    
    # Train model
    train_start = time.time()
    training_stats = model.fit(
        train_data['text'],
        train_data['label'],
        val_data['text'],
        val_data['label']
    )
    train_time = time.time() - train_start
    
    print(f"\n✓ Training completed in {train_time:.2f}s ({train_time/60:.2f} minutes)")
    
    # Evaluate on validation set (final metrics)
    val_results = evaluate_classifier(
        model, val_data['text'], val_data['label'], "Validation"
    )
    
    # Evaluate on test set
    test_results = evaluate_classifier(
        model, test_data['text'], test_data['label'], "Test"
    )
    
    # Compile all results
    all_results = {
        'model_config': {
            'model_name': model_name,
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
    
    # Log to WandB
    log_to_wandb(all_results, use_wandb=wandb_initialized)
    finish_wandb(use_wandb=wandb_initialized)
    
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
        '--model_name',
        type=str,
        default='roberta-base',
        help='Pre-trained model name (default: roberta-base)'
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
    
    args = parser.parse_args()
    
    # Validate dataset name
    if not args.dataset:
        parser.error("--dataset is required (or set HF_DATASET_NAME in .env)")
    
    # Determine upload setting
    upload_to_hf = args.upload_to_hf and not args.no_upload
    
    main(
        dataset_name=args.dataset,
        model_name=args.model_name,
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
        hf_repo=args.hf_repo
    )
