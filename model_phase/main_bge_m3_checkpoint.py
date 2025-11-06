"""
BGE-M3 Model with Checkpoint/Continue Support

This version supports checkpointing at every major stage:
1. Data loading
2. Embedding generation (train/val/test)
3. SVM training
4. Validation evaluation
5. Test evaluation

When session times out, you can resume from the last checkpoint.
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


class CheckpointManager:
    """Manages checkpoints for resumable training."""
    
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / 'checkpoint_state.json'
        self.state = self._load_state()
    
    def _load_state(self):
        """Load checkpoint state from disk."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                state = json.load(f)
                print(f"\n✓ Loaded checkpoint from {self.checkpoint_file}")
                print(f"  Last stage: {state.get('last_completed_stage', 'None')}")
                return state
        return {
            'last_completed_stage': None,
            'completed_stages': [],
            'metadata': {}
        }
    
    def _save_state(self):
        """Save checkpoint state to disk."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.state, f, indent=2)
        print(f"  ✓ Checkpoint saved to {self.checkpoint_file}")
    
    def is_stage_completed(self, stage_name):
        """Check if a stage has been completed."""
        return stage_name in self.state['completed_stages']
    
    def mark_stage_completed(self, stage_name, metadata=None):
        """Mark a stage as completed."""
        if stage_name not in self.state['completed_stages']:
            self.state['completed_stages'].append(stage_name)
        self.state['last_completed_stage'] = stage_name
        if metadata:
            self.state['metadata'][stage_name] = metadata
        self._save_state()
        print(f"  ✓ Stage '{stage_name}' marked as completed")
    
    def get_metadata(self, stage_name):
        """Get metadata for a stage."""
        return self.state['metadata'].get(stage_name, {})
    
    def save_embeddings(self, stage_name, embeddings, labels=None):
        """Save embeddings to checkpoint."""
        filepath = self.checkpoint_dir / f'{stage_name}_embeddings.npz'
        if labels is not None:
            np.savez_compressed(filepath, embeddings=embeddings, labels=labels)
        else:
            np.savez_compressed(filepath, embeddings=embeddings)
        print(f"  ✓ Saved embeddings to {filepath} ({embeddings.shape})")
    
    def load_embeddings(self, stage_name):
        """Load embeddings from checkpoint."""
        filepath = self.checkpoint_dir / f'{stage_name}_embeddings.npz'
        if filepath.exists():
            data = np.load(filepath)
            embeddings = data['embeddings']
            labels = data['labels'] if 'labels' in data else None
            print(f"  ✓ Loaded embeddings from {filepath} ({embeddings.shape})")
            return embeddings, labels
        return None, None
    
    def save_model(self, model, stage_name='svm_trained'):
        """Save trained SVM model."""
        filepath = self.checkpoint_dir / f'{stage_name}_classifier.pkl'
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"  ✓ Saved model to {filepath}")
    
    def load_model(self, stage_name='svm_trained'):
        """Load trained SVM model."""
        filepath = self.checkpoint_dir / f'{stage_name}_classifier.pkl'
        if filepath.exists():
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            print(f"  ✓ Loaded model from {filepath}")
            return model
        return None
    
    # Dataset checkpoint removed - always reload from HuggingFace (fast and avoids issues)
    
    def save_results(self, results, stage_name):
        """Save evaluation results."""
        filepath = self.checkpoint_dir / f'{stage_name}_results.json'
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  ✓ Saved results to {filepath}")
    
    def load_results(self, stage_name):
        """Load evaluation results."""
        filepath = self.checkpoint_dir / f'{stage_name}_results.json'
        if filepath.exists():
            with open(filepath, 'r') as f:
                results = json.load(f)
            print(f"  ✓ Loaded results from {filepath}")
            return results
        return None
    
    def get_progress_summary(self):
        """Get a summary of training progress."""
        completed = self.state['completed_stages']
        all_stages = [
            'data_loaded',
            'train_embeddings',
            'val_embeddings',
            'test_embeddings',
            'svm_trained',
            'validation_complete',
            'test_complete'
        ]
        
        summary = {
            'completed_stages': completed,
            'remaining_stages': [s for s in all_stages if s not in completed],
            'progress_percentage': len(completed) / len(all_stages) * 100
        }
        return summary


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


class BGEM3SentimentClassifier:
    """Sentiment classifier using BGE-M3 embeddings + SVM with checkpoint support."""
    
    def __init__(self, 
                 max_length=512,
                 batch_size=32,
                 C=1.0,
                 gamma='scale',
                 kernel='rbf',
                 random_state=42,
                 checkpoint_manager=None):
        self.model_name = MODEL_NAME
        self.max_length = max_length
        self.batch_size = batch_size
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.random_state = random_state
        self.checkpoint_manager = checkpoint_manager
        
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
        self.classifier = None
        
        self.label2id = None
        self.id2label = None
        self.is_fitted = False
        
    def _encode_texts_with_checkpoint(self, texts, stage_name, desc="Encoding"):
        """Generate embeddings for texts with checkpoint support."""
        
        # Check if embeddings already exist
        if self.checkpoint_manager and self.checkpoint_manager.is_stage_completed(stage_name):
            print(f"\n✓ Stage '{stage_name}' already completed, loading from checkpoint...")
            embeddings, _ = self.checkpoint_manager.load_embeddings(stage_name)
            if embeddings is not None:
                return embeddings
        
        # Check for partial/incomplete embeddings
        partial_file = None
        start_idx = 0
        embeddings = []
        
        if self.checkpoint_manager:
            partial_file = self.checkpoint_manager.checkpoint_dir / f'{stage_name}_partial.npz'
            if partial_file.exists():
                print(f"\n⚡ Found partial embeddings, resuming from checkpoint...")
                partial_data = np.load(partial_file)
                embeddings = [partial_data['embeddings']]
                start_idx = int(partial_data['last_index'])
                print(f"  Resuming from sample {start_idx}/{len(texts)} ({start_idx/len(texts)*100:.1f}%)")
        
        if start_idx == 0:
            print(f"\n[{stage_name}] Generating embeddings...")
            print(f"  Total samples: {len(texts)}")
            print(f"  Batch size: {self.batch_size}")
        
        start_time = time.time()
        
        # Create batches
        for i in range(start_idx, len(texts), self.batch_size):
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
            
            processed = i + len(batch_texts)
            
            # Progress logging
            if (i // self.batch_size + 1) % 10 == 0 or processed == len(texts):
                elapsed = time.time() - start_time
                samples_per_sec = processed / elapsed if elapsed > 0 else 0
                eta = (len(texts) - processed) / samples_per_sec if samples_per_sec > 0 else 0
                print(f"  Progress: {processed}/{len(texts)} samples "
                      f"({processed/len(texts)*100:.1f}%) - "
                      f"{samples_per_sec:.1f} samples/s - "
                      f"ETA: {eta:.0f}s")
            
            # 🔥 INCREMENTAL CHECKPOINT: Save every 100 batches (~6400 samples with batch_size=64)
            if self.checkpoint_manager and (i // self.batch_size + 1) % 100 == 0:
                current_embeddings = np.vstack(embeddings)
                np.savez_compressed(
                    partial_file,
                    embeddings=current_embeddings,
                    last_index=processed
                )
                print(f"  💾 Partial checkpoint saved at {processed}/{len(texts)} samples")
        
        embeddings = np.vstack(embeddings)
        elapsed = time.time() - start_time
        
        print(f"  ✓ Embeddings generated: {embeddings.shape}")
        print(f"  ✓ Time: {elapsed:.2f}s ({len(texts)/elapsed:.1f} samples/s)")
        
        # Save final checkpoint and clean up partial
        if self.checkpoint_manager:
            self.checkpoint_manager.save_embeddings(stage_name, embeddings)
            self.checkpoint_manager.mark_stage_completed(
                stage_name, 
                {'shape': embeddings.shape, 'time': elapsed}
            )
            # Delete partial checkpoint
            if partial_file and partial_file.exists():
                partial_file.unlink()
                print(f"  🗑️  Cleaned up partial checkpoint")
        
        return embeddings
    
    def fit(self, train_texts, train_labels, val_texts, val_labels, use_wandb=False):
        """Train the classifier with checkpoint support."""
        print("\n" + "="*60)
        print("Training with Checkpoint Support")
        print("="*60)
        
        # Create datasets for label mapping
        train_dataset = GameReviewDataset(train_texts, train_labels)
        val_dataset = GameReviewDataset(val_texts, val_labels, label2id=train_dataset.label2id)
        
        # Store label mappings
        self.label2id = train_dataset.label2id
        self.id2label = train_dataset.id2label
        
        print(f"\nDataset Info:")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        print(f"  Classes: {list(self.label2id.keys())}")
        
        # Generate train embeddings (with checkpoint)
        X_train = self._encode_texts_with_checkpoint(
            train_texts, 'train_embeddings', "Training"
        )
        y_train = np.array([item['label'] for item in train_dataset])
        
        # Generate validation embeddings (with checkpoint)
        X_val = self._encode_texts_with_checkpoint(
            val_texts, 'val_embeddings', "Validation"
        )
        y_val = np.array([item['label'] for item in val_dataset])
        
        # Train SVM (with checkpoint)
        if self.checkpoint_manager and self.checkpoint_manager.is_stage_completed('svm_trained'):
            print(f"\n✓ SVM training already completed, loading from checkpoint...")
            self.classifier = self.checkpoint_manager.load_model('svm_trained')
            self.is_fitted = True
        else:
            print(f"\n[svm_trained] Training SVM classifier...")
            print(f"  C: {self.C}")
            print(f"  gamma: {self.gamma}")
            print(f"  Kernel: {self.kernel}")
            print(f"  Training samples: {X_train.shape[0]:,}")
            print(f"  Features: {X_train.shape[1]:,}")
            print(f"\n  🚀 Starting SVM training (this may take 30-60 minutes)...")
            print(f"  ⏱️  Estimated time: ~{X_train.shape[0]/1000:.0f}-{X_train.shape[0]/500:.0f} minutes")
            print(f"  ⚠️  If timeout here, reduce C or use kernel='linear' for faster training")
            
            train_start = time.time()
            self.classifier = SVC(
                C=self.C,
                gamma=self.gamma,
                kernel=self.kernel,
                max_iter=30000,
                random_state=self.random_state,
                verbose=True  # Will print progress during training
            )
            
            try:
                self.classifier.fit(X_train, y_train)
                train_time = time.time() - train_start
                
                print(f"\n  ✓ Training completed in {train_time:.2f}s ({train_time/60:.1f} minutes)")
                
                # Save checkpoint IMMEDIATELY after training
                if self.checkpoint_manager:
                    self.checkpoint_manager.save_model(self.classifier, 'svm_trained')
                    self.checkpoint_manager.mark_stage_completed(
                        'svm_trained',
                        {'train_time': train_time, 'C': self.C, 'gamma': str(self.gamma)}
                    )
                    print(f"  💾 Model checkpoint saved successfully")
                
                self.is_fitted = True
                
            except KeyboardInterrupt:
                print(f"\n⚠️  Training interrupted! No checkpoint saved.")
                print(f"   Next run will restart SVM training from beginning.")
                raise
        
        print("\n✓ Training complete!")
        return 0  # Return dummy time
    
    def predict(self, texts):
        """Predict sentiment."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction!")
        
        # Generate embeddings (no checkpoint for prediction)
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        y_pred = self.classifier.predict(embeddings)
        return [self.id2label[pred] for pred in y_pred]
    
    def predict_from_embeddings(self, embeddings):
        """Predict from pre-computed embeddings."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction!")
        
        y_pred = self.classifier.predict(embeddings)
        return [self.id2label[pred] for pred in y_pred]
    
    def save(self, output_dir):
        """Save model components."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving model to {output_dir}")
        
        # Save classifier
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
            'gamma': str(self.gamma),
            'kernel': self.kernel,
            'random_state': self.random_state,
            'label2id': self.label2id,
            'id2label': self.id2label
        }
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"  ✓ Model saved")


def evaluate_with_checkpoint(model, texts, labels, split_name, checkpoint_manager, use_wandb=False):
    """Evaluate classifier with checkpoint support."""
    stage_name = f'{split_name.lower()}_complete'
    
    # Check if evaluation already done
    if checkpoint_manager and checkpoint_manager.is_stage_completed(stage_name):
        print(f"\n✓ Evaluation '{split_name}' already completed, loading from checkpoint...")
        results = checkpoint_manager.load_results(stage_name)
        if results:
            print(f"  Accuracy: {results[f'{split_name.lower()}_accuracy']:.4f}")
            print(f"  F1-score: {results[f'{split_name.lower()}_f1']:.4f}")
            return results
    
    print(f"\n{'='*60}")
    print(f"[{stage_name}] Evaluating on {split_name} set")
    print(f"{'='*60}")
    
    # Get embeddings (use checkpoint if available)
    embed_stage = f'{split_name.lower()}_embeddings'
    if checkpoint_manager and checkpoint_manager.is_stage_completed(embed_stage):
        X_embeddings, _ = checkpoint_manager.load_embeddings(embed_stage)
    else:
        # Generate embeddings
        X_embeddings = model._encode_texts_with_checkpoint(
            texts, embed_stage, split_name
        )
    
    # Predict
    print(f"\n  Predicting on {len(texts)} samples...")
    start_time = time.time()
    predictions = model.predict_from_embeddings(X_embeddings)
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
        try:
            wandb.log({
                f'{split_name.lower()}_accuracy': accuracy,
                f'{split_name.lower()}_f1': f1,
                f'{split_name.lower()}_precision': precision,
                f'{split_name.lower()}_recall': recall
            })
        except:
            pass
    
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
    
    # Save checkpoint
    if checkpoint_manager:
        checkpoint_manager.save_results(results, stage_name)
        checkpoint_manager.mark_stage_completed(stage_name, {'accuracy': accuracy, 'f1': f1})
    
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
         experiment_name=None,
         resume_from=None):
    """Main training pipeline with checkpoint support."""
    
    print("\n" + "="*60)
    print("BGE-M3 Training with Checkpoint/Continue Support")
    print("="*60)
    
    # Setup output directory
    if output_dir is None:
        if experiment_name:
            output_dir = f'model_phase/results/{experiment_name}'
        else:
            output_dir = f'model_phase/results/bge_m3_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 🔍 AUTO-DETECT checkpoint từ Kaggle input
    checkpoint_dir = output_dir / 'checkpoints'
    kaggle_checkpoint_found = False
    
    # Check nếu đang chạy trên Kaggle và có checkpoint dataset
    if Path('/kaggle/input').exists():
        print("\n🔍 Searching for checkpoints in /kaggle/input/...")
        
        # Tìm checkpoint trong các input datasets
        for input_subdir in Path('/kaggle/input').iterdir():
            if input_subdir.is_dir():
                potential_checkpoint = input_subdir / 'checkpoints'
                checkpoint_state_file = potential_checkpoint / 'checkpoint_state.json'
                
                if checkpoint_state_file.exists():
                    print(f"✅ Found checkpoint in: {potential_checkpoint}")
                    
                    # Copy checkpoint sang working directory
                    import shutil
                    if checkpoint_dir.exists():
                        shutil.rmtree(checkpoint_dir)
                    shutil.copytree(potential_checkpoint, checkpoint_dir)
                    
                    kaggle_checkpoint_found = True
                    print(f"✅ Checkpoint copied to: {checkpoint_dir}")
                    break
        
        if not kaggle_checkpoint_found:
            print("ℹ️  No checkpoint found in /kaggle/input/")
            print("   Starting fresh training...")
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(checkpoint_dir)
    
    # Show progress
    progress = checkpoint_manager.get_progress_summary()
    print(f"\nCheckpoint Progress:")
    print(f"  Completed: {progress['completed_stages']}")
    print(f"  Remaining: {progress['remaining_stages']}")
    print(f"  Progress: {progress['progress_percentage']:.1f}%")
    
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
                },
                resume='allow',
                id=run_name
            )
            wandb_initialized = True
            print(f"✓ WandB initialized - Project: {wandb_project}, Run: {run_name}")
        except Exception as e:
            print(f"⚠️  Could not initialize WandB: {e}")
    
    # Load data (ALWAYS reload from HuggingFace - fast and avoids Arrow file issues)
    print(f"\n[data_loaded] Loading dataset from HuggingFace...")
    print(f"  ℹ️  Note: Always reload dataset (fast ~2min, avoids checkpoint issues)")
    
    train_data, val_data, test_data = load_dataset_from_hf(
        dataset_name,
        subset_percentage=subset
    )
    
    # Mark as completed for progress tracking (but don't save dataset checkpoint)
    if not checkpoint_manager.is_stage_completed('data_loaded'):
        checkpoint_manager.mark_stage_completed(
            'data_loaded',
            {
                'train_size': len(train_data['text']),
                'val_size': len(val_data['text']),
                'test_size': len(test_data['text'])
            }
        )
    
    print(f"  Train: {len(train_data['text'])} samples")
    print(f"  Val: {len(val_data['text'])} samples")
    print(f"  Test: {len(test_data['text'])} samples")
    
    # Initialize model
    print(f"\n{'='*60}")
    print("Initializing BGE-M3 model")
    print(f"{'='*60}")
    
    model = BGEM3SentimentClassifier(
        max_length=max_length,
        batch_size=batch_size,
        C=C,
        gamma=gamma,
        kernel=kernel,
        checkpoint_manager=checkpoint_manager
    )
    
    # Train model (with checkpoint)
    train_start = time.time()
    model.fit(
        train_data['text'],
        train_data['label'],
        val_data['text'],
        val_data['label'],
        use_wandb=wandb_initialized
    )
    total_time = time.time() - train_start
    
    # Evaluate on validation set (with checkpoint)
    val_results = evaluate_with_checkpoint(
        model, val_data['text'], val_data['label'], 
        "Validation", checkpoint_manager, use_wandb=wandb_initialized
    )
    
    # Evaluate on test set (with checkpoint)
    test_results = evaluate_with_checkpoint(
        model, test_data['text'], test_data['label'],
        "Test", checkpoint_manager, use_wandb=wandb_initialized
    )
    
    # Compile final results
    all_results = {
        'model_config': {
            'model_name': MODEL_NAME,
            'classifier': 'SVM',
            'max_length': max_length,
            'batch_size': batch_size,
            'C': C,
            'gamma': str(gamma),
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
    
    # Save final results and model
    print(f"\n{'='*60}")
    print("Saving Final Results")
    print(f"{'='*60}")
    
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
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train BGE-M3 with checkpoint support')
    parser.add_argument('--dataset', type=str, default=os.getenv('HF_DATASET_NAME'),
                        help='HuggingFace dataset name')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for embedding generation')
    parser.add_argument('--C', type=float, default=3.0,
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
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Custom experiment name')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume from checkpoint directory')
    
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
    
    # Set output_dir based on resume_from if provided
    if args.resume_from:
        # Extract parent directory from checkpoint path
        resume_path = Path(args.resume_from)
        if resume_path.name == 'checkpoints':
            output_dir = str(resume_path.parent)
        else:
            output_dir = str(resume_path)
        print(f"Resuming from: {output_dir}")
    else:
        output_dir = args.output_dir
    
    main(
        dataset_name=args.dataset,
        max_length=args.max_length,
        batch_size=args.batch_size,
        C=args.C,
        gamma=gamma,
        kernel=args.kernel,
        subset=args.subset,
        output_dir=output_dir,
        use_wandb=args.use_wandb,
        upload_to_hf=upload_to_hf,
        hf_repo=args.hf_repo,
        experiment_name=args.experiment_name,
        resume_from=args.resume_from
    )
