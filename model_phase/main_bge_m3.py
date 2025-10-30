"""
BGE-M3 Model: Multilingual Embeddings for Sentiment Analysis

Core Idea: Use pre-trained BGE-M3 embeddings as features for sentiment classification.

How it Works:
- Uses BGE-M3 (BAAI General Embedding Model v3) to generate text embeddings
- Trains a classifier on top of these embeddings
- Leverages multilingual capabilities and semantic understanding

Key Features:
- State-of-the-art embedding quality
- Efficient inference with cached embeddings
- Works well with limited training data
"""

import sys
import os
from pathlib import Path
import argparse
import json
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Add project root to Python path
current_file = Path(__file__).absolute()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import pickle

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

# Import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

MODEL_NAME = 'BAAI/bge-m3'


class GameReviewDataset(Dataset):
    """PyTorch Dataset for game reviews with BGE-M3 embeddings."""
    
    def __init__(self, texts, labels, label2id=None):
        self.texts = texts
        self.labels = labels
        
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
        return {
            'text': str(self.texts[idx]),
            'label': self.label2id[self.labels[idx]]
        }


class WandbCallback:
    """Custom callback for WandB logging during training."""
    
    def __init__(self, use_wandb=False):
        self.use_wandb = use_wandb
    
    def log_metrics(self, metrics, step=None):
        """Log metrics to WandB."""
        if self.use_wandb and WANDB_AVAILABLE:
            try:
                if step is not None:
                    wandb.log(metrics, step=step)
                else:
                    wandb.log(metrics)
            except Exception:
                pass


class BGEM3SentimentClassifier:
    """Sentiment classifier using BGE-M3 embeddings + SVM."""
    
    def __init__(self, 
                 max_length=512,
                 batch_size=32,
                 C=1.0,
                 gamma='scale',
                 kernel='rbf',
                 random_state=42):
        self.model_name = MODEL_NAME
        self.max_length = max_length
        self.batch_size = batch_size
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.random_state = random_state
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Load BGE-M3 model (frozen for embedding extraction)
        print(f"\nLoading BGE-M3 model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.embedding_model = AutoModel.from_pretrained(self.model_name)
        self.embedding_model.to(self.device)
        self.embedding_model.eval()  # Freeze embedding model
        
        # Freeze all parameters
        for param in self.embedding_model.parameters():
            param.requires_grad = False
        
        # SVM Classifier
        self.classifier = SVC(
            C=C,
            gamma=gamma,
            kernel=kernel,
            max_iter=20000,
            random_state=random_state,
            verbose=True,
            probability=True  # Enable probability estimates
        )
        
        self.label2id = None
        self.id2label = None
        self.is_fitted = False
        
    def _encode_texts(self, texts, desc="Encoding"):
        """Generate embeddings for texts."""
        embeddings = []
        
        # Create batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                # Use CLS token embedding
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)
            
            if (i // self.batch_size + 1) % 10 == 0:
                print(f"  Processed {i + len(batch_texts)}/{len(texts)} samples")
        
        return np.vstack(embeddings)
    
    def fit(self, train_texts, train_labels, val_texts, val_labels, use_wandb=False):
        """Train the classifier."""
        print("\n" + "="*60)
        print("Preparing Data")
        print("="*60)
        
        # Create datasets
        train_dataset = GameReviewDataset(train_texts, train_labels)
        val_dataset = GameReviewDataset(val_texts, val_labels, label2id=train_dataset.label2id)
        
        # Store label mappings
        self.label2id = train_dataset.label2id
        self.id2label = train_dataset.id2label
        
        print(f"\nTraining samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Batch size: {self.batch_size}")
        
        print("\n" + "="*60)
        print("Training BGE-M3 Classifier")
        print("="*60)
        
        # Setup WandB callback
        wandb_callback = WandbCallback(use_wandb=use_wandb)
        
        # Generate embeddings (frozen BGE-M3 model)
        print(f"\n[1/2] Generating BGE-M3 embeddings (frozen model)...")
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - Max length: {self.max_length}")
        embed_start = time.time()
        X_train = self._encode_texts(train_texts, "Training")
        embed_time = time.time() - embed_start
        print(f"  - Embedding shape: {X_train.shape}")
        print(f"  - Embedding time: {embed_time:.2f}s")
        print(f"  - Embedding model is frozen (no gradient updates)")
        
        wandb_callback.log_metrics({'embedding_time': embed_time})
        
        # Encode labels
        y_train = np.array([item['label'] for item in train_dataset])
        
        # Train SVM classifier on frozen embeddings
        print(f"\n[2/2] Training SVM classifier...")
        print(f"  - C: {self.C}")
        print(f"  - gamma: {self.gamma}")
        print(f"  - Kernel: {self.kernel}")
        train_start = time.time()
        self.classifier.fit(X_train, y_train)
        train_time = time.time() - train_start
        print(f"  - Training time: {train_time:.2f}s")
        
        wandb_callback.log_metrics({'classifier_train_time': train_time})
        
        self.is_fitted = True
        print("\n✓ Training complete!")
        
        return embed_time + train_time
    
    def predict(self, texts):
        """Predict sentiment."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction!")
        
        X_embeddings = self._encode_texts(texts, "Prediction")
        y_pred = self.classifier.predict(X_embeddings)
        return [self.id2label[pred] for pred in y_pred]
    
    def predict_proba(self, texts):
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction!")
        
        X_embeddings = self._encode_texts(texts, "Prediction")
        return self.classifier.predict_proba(X_embeddings)
    
    def save(self, output_dir):
        """Save model components."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving model to {output_dir}")
        
        # Save classifier and label encoder
        with open(output_dir / 'classifier.pkl', 'wb') as f:
            pickle.dump(self.classifier, f)
        
        # Create label encoder for saving
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array([self.id2label[i] for i in sorted(self.id2label.keys())])
        
        with open(output_dir / 'label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)
        
        # Save config
        config = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'C': self.C,
            'gamma': self.gamma,
            'kernel': self.kernel,
            'random_state': self.random_state
        }
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"  ✓ Model saved")
    
    @classmethod
    def load(cls, output_dir):
        """Load model from directory."""
        output_dir = Path(output_dir)
        
        with open(output_dir / 'config.json', 'r') as f:
            config = json.load(f)
        
        # Extract only constructor parameters
        constructor_params = {
            'max_length': config['max_length'],
            'batch_size': config['batch_size'],
            'C': config['C'],
            'gamma': config.get('gamma', 'scale'),
            'kernel': config['kernel'],
            'random_state': config['random_state']
        }
        
        model = cls(**constructor_params)
        
        with open(output_dir / 'classifier.pkl', 'rb') as f:
            model.classifier = pickle.load(f)
        
        with open(output_dir / 'label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
            model.label2id = {label: idx for idx, label in enumerate(label_encoder.classes_)}
            model.id2label = {idx: label for label, idx in model.label2id.items()}
        
        model.is_fitted = True
        print(f"✓ Model loaded from {output_dir}")
        return model


def evaluate_classifier(model, texts, labels, split_name="Test", use_wandb=False):
    """Evaluate classifier and return metrics."""
    print(f"\n{'='*60}")
    print(f"Evaluating on {split_name} set")
    print(f"{'='*60}")
    
    # Predict
    start_time = time.time()
    predictions = model.predict(texts)
    inference_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    class_report = classification_report(
        labels, predictions, output_dict=True, zero_division=0
    )
    
    # Confusion matrix
    unique_labels = sorted(set(labels) | set(predictions))
    cm = confusion_matrix(labels, predictions, labels=unique_labels)
    
    # Print results
    print(f"\n{split_name} Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-score: {f1:.4f}")
    print(f"  Inference time: {inference_time:.2f}s")
    print(f"  Samples/second: {len(texts)/inference_time:.2f}")
    
    # Log to wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb_callback = WandbCallback(use_wandb=True)
        wandb_callback.log_metrics({
            f'{split_name.lower()}_accuracy': accuracy,
            f'{split_name.lower()}_f1': f1,
            f'{split_name.lower()}_precision': precision,
            f'{split_name.lower()}_recall': recall
        })
    
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
         batch_size=32,
         C=1.0,
         gamma='scale',
         kernel='rbf',
         subset=1.0,
         output_dir=None,
         use_wandb=False,
         upload_to_hf=True,
         hf_repo=None,
         skip_test_eval=False,
         experiment_name=None):
    """Main training pipeline."""
    print("\n" + "="*60)
    print("BGE-M3 Embeddings for Sentiment Analysis")
    print("="*60)
    
    # Setup output directory
    output_dir = setup_output_directory(output_dir, model_name="bge_m3")
    
    # Initialize WandB
    wandb_initialized = False
    if use_wandb and WANDB_AVAILABLE:
        try:
            wandb_project = os.getenv('WANDB_PROJECT', 'game-review-sentiment')
            run_name = experiment_name or f"bge_m3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            wandb.init(
                project=wandb_project,
                name=run_name,
                config={
                    "model": MODEL_NAME,
                    "classifier": "SVM",
                    "max_length": max_length,
                    "batch_size": batch_size,
                    "C": C,
                    "gamma": gamma,
                    "kernel": kernel,
                    "subset": subset
                }
            )
            wandb_initialized = True
            print(f"✓ WandB initialized - Project: {wandb_project}, Run: {run_name}")
        except Exception as e:
            print(f"⚠️  Could not initialize WandB: {e}")
    
    # Load data
    train_data, val_data, test_data = load_dataset_from_hf(
        dataset_name,
        subset_percentage=subset
    )
    
    # Initialize model
    print(f"\n{'='*60}")
    print("Initializing BGE-M3 model")
    print(f"{'='*60}")
    
    model = BGEM3SentimentClassifier(
        max_length=max_length,
        batch_size=batch_size,
        C=C,
        gamma=gamma,
        kernel=kernel
    )
    
    # Train model
    train_start = time.time()
    train_time = model.fit(
        train_data['text'],
        train_data['label'],
        val_data['text'],
        val_data['label'],
        use_wandb=wandb_initialized
    )
    total_time = time.time() - train_start
    
    print(f"\n✓ Training completed in {total_time:.2f}s")
    
    # Evaluate on validation set
    val_results = evaluate_classifier(
        model, val_data['text'], val_data['label'], "Validation",
        use_wandb=wandb_initialized
    )
    
    # Evaluate on test set
    if skip_test_eval:
        print(f"\n{'='*60}")
        print("Skipping test set evaluation (grid search mode)")
        print(f"{'='*60}")
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
        test_results = evaluate_classifier(
            model, test_data['text'], test_data['label'], "Test",
            use_wandb=wandb_initialized
        )
    
    # Compile results
    all_results = {
        'model_config': {
            'model_name': MODEL_NAME,
            'classifier': 'SVM',
            'max_length': max_length,
            'batch_size': batch_size,
            'C': C,
            'gamma': gamma,
            'kernel': kernel,
            'subset': subset
        },
        'training_time': total_time,
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
    
    # Finish WandB
    if wandb_initialized:
        finish_wandb(use_wandb=True)
    
    # Print summary
    print_training_summary(all_results, output_dir)
    
    # Upload to HuggingFace
    if upload_to_hf:
        upload_results_to_hf(
            results=all_results,
            output_dir=output_dir,
            model_name="bge_m3",
            hf_repo_name=hf_repo
        )
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train BGE-M3 for sentiment analysis')
    parser.add_argument('--dataset', type=str, default=os.getenv('HF_DATASET_NAME'),
                        help='HuggingFace dataset name')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for embedding generation')
    parser.add_argument('--C', type=float, default=1.0,
                        help='Regularization parameter for SVM')
    parser.add_argument('--gamma', type=str, default='scale',
                        help='Kernel coefficient for RBF (scale, auto, or float)')
    parser.add_argument('--kernel', type=str, default='rbf',
                        choices=['linear', 'poly', 'rbf', 'sigmoid'],
                        help='SVM kernel type')
    parser.add_argument('--subset', type=float, default=1.0,
                        help='Fraction of data to use')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use WandB for tracking')
    parser.add_argument('--upload_to_hf', action='store_true', default=True,
                        help='Upload to HuggingFace Hub')
    parser.add_argument('--no_upload', action='store_true',
                        help='Skip HuggingFace upload')
    parser.add_argument('--hf_repo', type=str, default=None,
                        help='HuggingFace repo name')
    parser.add_argument('--skip_test_eval', action='store_true',
                        help='Skip test evaluation')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Custom experiment name for WandB')
    
    args = parser.parse_args()
    
    if not args.dataset:
        parser.error("--dataset is required (or set HF_DATASET_NAME in .env)")
    
    upload_to_hf = args.upload_to_hf and not args.no_upload
    
    # Parse gamma (can be string or float)
    gamma = args.gamma
    if gamma not in ['scale', 'auto']:
        try:
            gamma = float(gamma)
        except ValueError:
            parser.error(f"gamma must be 'scale', 'auto', or a float, got: {gamma}")
    
    main(
        dataset_name=args.dataset,
        max_length=args.max_length,
        batch_size=args.batch_size,
        C=args.C,
        gamma=gamma,
        kernel=args.kernel,
        subset=args.subset,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        upload_to_hf=upload_to_hf,
        hf_repo=args.hf_repo,
        skip_test_eval=args.skip_test_eval,
        experiment_name=args.experiment_name
    )
