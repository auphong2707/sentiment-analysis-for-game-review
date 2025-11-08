"""
XGBoost Model: Using Pre-computed BGE-M3 Embeddings

Core Idea: Load pre-computed BGE-M3 embeddings from checkpoint and train XGBoost classifier.

How it Works:
- Loads embeddings from checkpoint files (no embedding computation needed)
- Trains XGBoost classifier on these embeddings
- Supports grid search for hyperparameter tuning

Key Features:
- Fast training by skipping embedding generation
- Uses efficient XGBoost for classification
- Supports hyperparameter tuning
"""

import os
import sys
import argparse
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Add project root to Python path
current_file = Path(__file__).absolute()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
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

# Import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class CheckpointLoader:
    """Loads pre-computed embeddings and labels from checkpoint."""
    
    def __init__(self, checkpoint_dir, dataset_name=None):
        self.checkpoint_dir = Path(checkpoint_dir)
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        # Load checkpoint state
        state_file = self.checkpoint_dir / 'checkpoint_state.json'
        if not state_file.exists():
            raise FileNotFoundError(f"Checkpoint state file not found: {state_file}")
        
        with open(state_file) as f:
            self.state = json.load(f)
        
        print(f"\n{'='*60}")
        print("Checkpoint Information")
        print(f"{'='*60}")
        print(f"Location: {self.checkpoint_dir}")
        print(f"Completed stages: {len(self.state['completed_stages'])}")
        
        # Check required stages
        required_stages = ['train_embeddings', 'val_embeddings']
        for stage in required_stages:
            if stage not in self.state['completed_stages']:
                raise ValueError(f"Required stage '{stage}' not found in checkpoint")
        
        # Store dataset name for loading labels
        self.dataset_name = dataset_name
        self._dataset_cache = {}
    
    def _load_labels_from_dataset(self, split_name):
        """Load labels from HuggingFace dataset."""
        if split_name in self._dataset_cache:
            return self._dataset_cache[split_name]
        
        if not self.dataset_name:
            raise ValueError("dataset_name is required to load labels")
        
        print(f"  Loading {split_name} labels from HuggingFace dataset...")
        from datasets import load_dataset
        dataset = load_dataset(self.dataset_name, split=split_name)
        labels = np.array(dataset['review_category'])
        
        self._dataset_cache[split_name] = labels
        return labels
    
    def load_embeddings(self, split_name, subset_percentage=1.0):
        """Load embeddings for a specific split (train/val/test)."""
        stage_name = f'{split_name}_embeddings'
        
        if stage_name not in self.state['completed_stages']:
            raise ValueError(f"Stage '{stage_name}' not found in checkpoint")
        
        # Load embeddings
        embed_file = self.checkpoint_dir / f'{stage_name}_embeddings.npz'
        if not embed_file.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embed_file}")
        
        print(f"\nLoading {split_name} embeddings from checkpoint...")
        data = np.load(embed_file, allow_pickle=True)
        embeddings = data['embeddings']
        
        # Try to load labels from checkpoint, if not available load from dataset
        if 'labels' in data:
            labels = data['labels']
            print(f"  ✓ Labels loaded from checkpoint")
        else:
            print(f"  ⚠️  Labels not in checkpoint, loading from dataset...")
            labels = self._load_labels_from_dataset(split_name)
        
        # Apply subset if needed
        if subset_percentage < 1.0:
            n_samples = int(len(embeddings) * subset_percentage)
            print(f"  Applying {subset_percentage*100:.1f}% subset: {len(embeddings)} -> {n_samples} samples")
            
            # Stratified sampling to maintain label distribution
            from sklearn.model_selection import train_test_split
            indices = np.arange(len(embeddings))
            subset_indices, _ = train_test_split(
                indices, 
                train_size=subset_percentage,
                stratify=labels,
                random_state=42
            )
            embeddings = embeddings[subset_indices]
            labels = labels[subset_indices]
        
        print(f"  ✓ Loaded {split_name}: {embeddings.shape}")
        print(f"    Samples: {len(embeddings)}")
        print(f"    Features: {embeddings.shape[1]}")
        print(f"    Unique labels: {len(np.unique(labels))}")
        
        return embeddings, labels
    
    def get_metadata(self, stage_name):
        """Get metadata for a specific stage."""
        return self.state['metadata'].get(stage_name, {})


class XGBoostSentimentClassifier:
    """XGBoost classifier for sentiment analysis using pre-computed embeddings."""
    
    def __init__(self,
                 n_estimators=100,
                 max_depth=6,
                 learning_rate=0.3,
                 subsample=1.0,
                 colsample_bytree=1.0,
                 random_state=42,
                 use_gpu=True):
        """
        Initialize XGBoost classifier.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate (eta)
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns when constructing each tree
            random_state: Random seed
            use_gpu: Whether to use GPU for training
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.use_gpu = use_gpu
        
        self.model = None
        self.label_encoder = None
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, use_wandb=False):
        """Train XGBoost classifier."""
        print(f"\n{'='*60}")
        print("Training XGBoost Classifier")
        print(f"{'='*60}")
        print(f"Training samples: {len(X_train)}")
        print(f"Features: {X_train.shape[1]}")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        # XGBoost parameters
        params = {
            'objective': 'multi:softmax',
            'num_class': len(self.label_encoder.classes_),
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'random_state': self.random_state,
            'eval_metric': 'mlogloss',
            'tree_method': 'hist',  # Use histogram-based algorithm
        }
        
        # Use GPU if available and requested
        if self.use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    params['tree_method'] = 'gpu_hist'
                    params['gpu_id'] = 0
                    print("  Using GPU acceleration")
            except:
                print("  GPU not available, using CPU")
        
        print(f"\nHyperparameters:")
        print(f"  n_estimators: {self.n_estimators}")
        print(f"  max_depth: {self.max_depth}")
        print(f"  learning_rate: {self.learning_rate}")
        print(f"  subsample: {self.subsample}")
        print(f"  colsample_bytree: {self.colsample_bytree}")
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train_encoded)
        
        # Create evaluation set if validation data provided
        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            dval = xgb.DMatrix(X_val, label=y_val_encoded)
            evals.append((dval, 'val'))
        
        # Train model
        print("\nTraining model...")
        start_time = time.time()
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            verbose_eval=False
        )
        
        train_time = time.time() - start_time
        
        print(f"\n✓ Training completed in {train_time:.2f}s")
        
        return train_time
    
    def predict(self, X):
        """Predict labels for input embeddings."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        dtest = xgb.DMatrix(X)
        y_pred_encoded = self.model.predict(dtest)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded.astype(int))
        
        return y_pred
    
    def save(self, output_dir):
        """Save model and label encoder."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        model_file = output_dir / 'xgboost_model.json'
        self.model.save_model(str(model_file))
        
        # Save label encoder
        import pickle
        encoder_file = output_dir / 'label_encoder.pkl'
        with open(encoder_file, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save config
        config = {
            'model_type': 'XGBoost',
            'embedding_model': 'BAAI/bge-m3',
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'random_state': self.random_state,
            'classes': self.label_encoder.classes_.tolist(),
            'num_classes': len(self.label_encoder.classes_),
            # Thêm id2label mapping để dễ interpret
            'id2label': {i: str(label) for i, label in enumerate(self.label_encoder.classes_)},
            'label2id': {str(label): i for i, label in enumerate(self.label_encoder.classes_)}
        }
        
        config_file = output_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✓ Model saved to: {output_dir}")
        print(f"  - {model_file.name}")
        print(f"  - {encoder_file.name}")
        print(f"  - {config_file.name}")


def evaluate_classifier(model, X, y, split_name="Test", use_wandb=False):
    """Evaluate classifier and return metrics."""
    print(f"\n{'='*60}")
    print(f"Evaluating on {split_name} set")
    print(f"{'='*60}")
    
    # Predict
    start_time = time.time()
    predictions = model.predict(X)
    inference_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(y, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predictions, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    class_report = classification_report(
        y, predictions, output_dict=True, zero_division=0
    )
    
    # Confusion matrix
    unique_labels = sorted(set(y) | set(predictions))
    cm = confusion_matrix(y, predictions, labels=unique_labels)
    
    # Print results
    print(f"\n{split_name} Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-score: {f1:.4f}")
    print(f"  Inference time: {inference_time:.2f}s")
    print(f"  Samples/second: {len(y)/inference_time:.2f}")
    
    # Log to wandb
    if use_wandb and WANDB_AVAILABLE:
        log_to_wandb({
            f'{split_name.lower()}_accuracy': accuracy,
            f'{split_name.lower()}_precision': precision,
            f'{split_name.lower()}_recall': recall,
            f'{split_name.lower()}_f1': f1,
        })
    
    # Compile results
    results = {
        f'{split_name.lower()}_accuracy': float(accuracy),
        f'{split_name.lower()}_precision': float(precision),
        f'{split_name.lower()}_recall': float(recall),
        f'{split_name.lower()}_f1': float(f1),
        f'{split_name.lower()}_inference_time': float(inference_time),
        f'{split_name.lower()}_samples_per_second': float(len(y)/inference_time),
        f'{split_name.lower()}_classification_report': class_report,
        f'{split_name.lower()}_confusion_matrix': cm.tolist()
    }
    
    return results


def run_grid_search(checkpoint_dir,
                    dataset_name=None,
                    n_estimators_values=[100, 200],
                    max_depth_values=[6, 8],
                    learning_rate_values=[0.1, 0.3],
                    subset=0.1,
                    output_dir='model_phase/results/gridsearch_xgboost'):
    """
    Run grid search with pre-loaded embeddings.
    Note: Grid search only uses train + val (NO test evaluation).
    """
    print("\n" + "="*60)
    print("XGBoost Grid Search (Using Pre-computed Embeddings)")
    print("="*60)
    print("Note: Grid search only evaluates on validation set (test is skipped)")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load embeddings from checkpoint
    loader = CheckpointLoader(checkpoint_dir, dataset_name=dataset_name)
    
    print(f"\n{'='*60}")
    print("Loading embeddings from checkpoint (train + val only)")
    print(f"{'='*60}")
    
    X_train, y_train = loader.load_embeddings('train', subset_percentage=subset)
    X_val, y_val = loader.load_embeddings('val', subset_percentage=subset)
    
    # Grid search
    print(f"\n{'='*60}")
    print("Grid Search on XGBoost hyperparameters")
    print(f"{'='*60}")
    
    # Generate all combinations
    from itertools import product
    config_combinations = list(product(n_estimators_values, max_depth_values, learning_rate_values))
    
    total_configs = len(config_combinations)
    print(f"Total configurations: {total_configs}")
    print("")
    
    results = []
    best_f1 = 0
    best_config = None
    
    for current, (n_est, max_d, lr) in enumerate(config_combinations, 1):
        print(f"\n{'='*60}")
        print(f"Configuration {current}/{total_configs}")
        print(f"{'='*60}")
        print(f"  n_estimators: {n_est}")
        print(f"  max_depth: {max_d}")
        print(f"  learning_rate: {lr}")
        
        # Train model
        model = XGBoostSentimentClassifier(
            n_estimators=n_est,
            max_depth=max_d,
            learning_rate=lr
        )
        
        train_time = model.fit(X_train, y_train, X_val, y_val)
        
        # Evaluate on validation set
        val_results = evaluate_classifier(model, X_val, y_val, "Validation")
        
        # Store results
        config_result = {
            'n_estimators': n_est,
            'max_depth': max_d,
            'learning_rate': lr,
            'train_time': train_time,
            'val_accuracy': val_results['validation_accuracy'],
            'val_f1': val_results['validation_f1'],
            'val_precision': val_results['validation_precision'],
            'val_recall': val_results['validation_recall']
        }
        results.append(config_result)
        
        # Update best config
        if val_results['validation_f1'] > best_f1:
            best_f1 = val_results['validation_f1']
            best_config = config_result
            print(f"\n  ✓ New best F1: {best_f1:.4f}")
    
    # Save results
    print(f"\n{'='*60}")
    print("Grid Search Complete!")
    print(f"{'='*60}")
    
    # Save all results
    results_file = output_dir / 'gridsearch_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'results': results,
            'best_config': best_config
        }, f, indent=2)
    
    # Save best config
    best_config_file = output_dir / 'best_config.txt'
    with open(best_config_file, 'w') as f:
        f.write(f"Best Configuration Found:\n")
        f.write(f"  n_estimators: {best_config['n_estimators']}\n")
        f.write(f"  max_depth: {best_config['max_depth']}\n")
        f.write(f"  learning_rate: {best_config['learning_rate']}\n")
        f.write(f"  Val F1: {best_config['val_f1']:.4f}\n")
        f.write(f"  Val Accuracy: {best_config['val_accuracy']:.4f}\n")
    
    print(f"\n✓ Results saved to: {results_file}")
    print(f"✓ Best config saved to: {best_config_file}")
    
    print(f"\nBest Configuration:")
    print(f"  n_estimators: {best_config['n_estimators']}")
    print(f"  max_depth: {best_config['max_depth']}")
    print(f"  learning_rate: {best_config['learning_rate']}")
    print(f"  Val F1: {best_config['val_f1']:.4f}")
    print(f"  Val Accuracy: {best_config['val_accuracy']:.4f}")
    
    return best_config


def main(checkpoint_dir,
         dataset_name=None,
         n_estimators=100,
         max_depth=6,
         learning_rate=0.3,
         subsample=1.0,
         colsample_bytree=1.0,
         subset=1.0,
         output_dir=None,
         use_wandb=False,
         upload_to_hf=True,
         hf_repo=None,
         experiment_name=None,
         grid_search=False,
         n_estimators_values=None,
         max_depth_values=None,
         learning_rate_values=None):
    """Main training pipeline."""
    
    # Grid search mode
    if grid_search:
        if n_estimators_values is None:
            n_estimators_values = [100, 200]
        if max_depth_values is None:
            max_depth_values = [6, 8]
        if learning_rate_values is None:
            learning_rate_values = [0.1, 0.3]
        
        gridsearch_dir = output_dir or 'model_phase/results/gridsearch_xgboost'
        best_config = run_grid_search(
            checkpoint_dir,
            dataset_name=dataset_name,
            n_estimators_values=n_estimators_values,
            max_depth_values=max_depth_values,
            learning_rate_values=learning_rate_values,
            subset=0.1,
            output_dir=gridsearch_dir
        )
        return best_config
    
    print("\n" + "="*60)
    print("XGBoost Training (Using Pre-computed Embeddings)")
    print("="*60)
    
    # Setup output directory
    output_dir = setup_output_directory(output_dir, model_name="xgboost")
    output_dir = Path(output_dir)
    
    # 🔍 AUTO-DETECT checkpoint từ Kaggle input (nếu checkpoint_dir không tồn tại)
    checkpoint_path = Path(checkpoint_dir)
    kaggle_checkpoint_found = False
    
    if not checkpoint_path.exists() and Path('/kaggle/input').exists():
        print("\n🔍 Checkpoint không tồn tại tại đường dẫn chỉ định.")
        print("   Đang tìm checkpoint trong /kaggle/input/...")
        
        # Tìm checkpoint trong các input datasets
        for input_subdir in Path('/kaggle/input').iterdir():
            if input_subdir.is_dir():
                potential_checkpoint = input_subdir / 'checkpoints'
                checkpoint_state_file = potential_checkpoint / 'checkpoint_state.json'
                
                if checkpoint_state_file.exists():
                    print(f"✅ Tìm thấy checkpoint trong: {potential_checkpoint}")
                    
                    # Copy checkpoint sang working directory
                    import shutil
                    target_checkpoint = output_dir / 'checkpoints_loaded'
                    if target_checkpoint.exists():
                        shutil.rmtree(target_checkpoint)
                    shutil.copytree(potential_checkpoint, target_checkpoint)
                    
                    checkpoint_dir = str(target_checkpoint)
                    kaggle_checkpoint_found = True
                    print(f"✅ Checkpoint đã được copy sang: {target_checkpoint}")
                    break
        
        if not kaggle_checkpoint_found:
            print("ℹ️  Không tìm thấy checkpoint trong /kaggle/input/")
            print("   Vui lòng kiểm tra lại đường dẫn checkpoint_dir")
    elif checkpoint_path.exists():
        print(f"\n✅ Sử dụng checkpoint từ: {checkpoint_path}")
    
    # Initialize WandB
    wandb_initialized = False
    if use_wandb and WANDB_AVAILABLE:
        experiment_name = experiment_name or f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb_initialized = init_wandb_if_available(
            project_name="game-review-sentiment",
            experiment_name=experiment_name,
            config={
                'model': 'xgboost',
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'subset': subset
            }
        )
    
    # Load embeddings from checkpoint
    loader = CheckpointLoader(checkpoint_dir, dataset_name=dataset_name)
    
    print(f"\n{'='*60}")
    print("Loading embeddings from checkpoint")
    print(f"{'='*60}")
    
    X_train, y_train = loader.load_embeddings('train', subset_percentage=subset)
    X_val, y_val = loader.load_embeddings('val', subset_percentage=subset)
    X_test, y_test = loader.load_embeddings('test', subset_percentage=subset)
    
    # Initialize model
    print(f"\n{'='*60}")
    print("Initializing XGBoost model")
    print(f"{'='*60}")
    
    model = XGBoostSentimentClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree
    )
    
    # Train model
    train_start = time.time()
    train_time = model.fit(X_train, y_train, X_val, y_val, use_wandb=wandb_initialized)
    total_time = time.time() - train_start
    
    print(f"\n✓ Training completed in {total_time:.2f}s")
    
    # Evaluate on validation set
    print(f"\n{'='*60}")
    print("Evaluating on Validation Set")
    print(f"{'='*60}")
    val_results = evaluate_classifier(
        model, X_val, y_val, "Validation",
        use_wandb=wandb_initialized
    )
    
    # Evaluate on test set (always run - test embeddings are available)
    print(f"\n{'='*60}")
    print("Evaluating on Test Set")
    print(f"{'='*60}")
    test_results = evaluate_classifier(
        model, X_test, y_test, "Test",
        use_wandb=wandb_initialized
    )
    
    # Compile results
    all_results = {
        'model_config': {
            'model_name': 'XGBoost',
            'embedding_model': 'BAAI/bge-m3',
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'subset': subset
        },
        'training_time': total_time,
        'dataset_info': {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test)
        },
        **val_results,
        **test_results
    }
    
    # Save results and model
    save_results_to_json(all_results, output_dir / 'results.json')
    model.save(output_dir)
    
    # Finish WandB
    if wandb_initialized:
        finish_wandb()
    
    # Print summary
    print_training_summary(all_results, output_dir)
    
    # Upload to HuggingFace (auto-detect repo name if not provided)
    if upload_to_hf:
        if hf_repo is None:
            # Fallback to dataset name or env variable
            hf_repo = dataset_name or os.getenv('HF_DATASET_NAME')
            if hf_repo:
                print(f"\n📤 Sử dụng HF repo: {hf_repo} (auto-detected)")
        
        if hf_repo:
            upload_results_to_hf(
                results=all_results,
                output_dir=output_dir,
                model_name="xgboost",
                hf_repo_name=hf_repo
            )
        else:
            print("\n⚠️  Bỏ qua upload HuggingFace (không có repo name)")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train XGBoost using pre-computed embeddings')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory containing pre-computed embeddings')
    parser.add_argument('--dataset', type=str, default=os.getenv('HF_DATASET_NAME'),
                        help='HuggingFace dataset name (for metadata only)')
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='Number of boosting rounds')
    parser.add_argument('--max_depth', type=int, default=6,
                        help='Maximum tree depth')
    parser.add_argument('--learning_rate', type=float, default=0.3,
                        help='Learning rate (eta)')
    parser.add_argument('--subsample', type=float, default=1.0,
                        help='Subsample ratio of training instances')
    parser.add_argument('--colsample_bytree', type=float, default=1.0,
                        help='Subsample ratio of columns')
    parser.add_argument('--subset', type=float, default=1.0,
                        help='Fraction of checkpoint data to use')
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
                        help='Custom experiment name for WandB')
    parser.add_argument('--grid_search', action='store_true',
                        help='Run grid search mode')
    parser.add_argument('--n_estimators_values', type=int, nargs='+', default=None,
                        help='n_estimators values for grid search')
    parser.add_argument('--max_depth_values', type=int, nargs='+', default=None,
                        help='max_depth values for grid search')
    parser.add_argument('--learning_rate_values', type=float, nargs='+', default=None,
                        help='learning_rate values for grid search')
    
    args = parser.parse_args()
    
    upload_to_hf = args.upload_to_hf and not args.no_upload
    
    main(
        checkpoint_dir=args.checkpoint_dir,
        dataset_name=args.dataset,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        subset=args.subset,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        upload_to_hf=upload_to_hf,
        hf_repo=args.hf_repo,
        experiment_name=args.experiment_name,
        grid_search=args.grid_search,
        n_estimators_values=args.n_estimators_values,
        max_depth_values=args.max_depth_values,
        learning_rate_values=args.learning_rate_values
    )
