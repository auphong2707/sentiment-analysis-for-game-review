"""
Utility functions for sentiment analysis models.

This module contains reusable functions for data loading, evaluation, 
and other common operations.
"""

import time
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from datasets import load_dataset


def load_dataset_from_hf(dataset_name, subset_percentage=1.0):
    """
    Load dataset from HuggingFace Hub.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace Hub
        subset_percentage: Fraction of data to use (0-1, for quick experiments)
        
    Returns:
        Tuple of (train_data, val_data, test_data) where each is a dict 
        with 'text' and 'label' keys
    """
    print(f"\n{'='*60}")
    print(f"Loading dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # Load dataset
    dataset = load_dataset(dataset_name)
    
    # Extract splits
    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset = dataset['test']
    
    # Apply subset if needed
    if subset_percentage < 1.0:
        train_size = int(len(train_dataset) * subset_percentage)
        val_size = int(len(val_dataset) * subset_percentage)
        test_size = int(len(test_dataset) * subset_percentage)
        
        train_dataset = train_dataset.select(range(train_size))
        val_dataset = val_dataset.select(range(val_size))
        test_dataset = test_dataset.select(range(test_size))
        
        print(f"⚠️  Using {subset_percentage*100}% subset of data")
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset):,} samples")
    print(f"  Validation: {len(val_dataset):,} samples")
    print(f"  Test: {len(test_dataset):,} samples")
    
    # Convert to simple format
    train_data = {
        'text': train_dataset['review_text'],
        'label': train_dataset['review_category']
    }
    val_data = {
        'text': val_dataset['review_text'],
        'label': val_dataset['review_category']
    }
    test_data = {
        'text': test_dataset['review_text'],
        'label': test_dataset['review_category']
    }
    
    return train_data, val_data, test_data


def evaluate_classifier(model, texts, labels, split_name="Test"):
    """
    Evaluate a classifier and return comprehensive metrics.
    
    Args:
        model: Trained classifier with predict() method
        texts: List of text samples
        labels: True labels
        split_name: Name of the split for display (e.g., "Test", "Validation")
        
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
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    class_report = classification_report(
        labels, predictions, output_dict=True, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions, labels=model.label_encoder.classes_)
    
    # Print results
    print(f"\n{split_name} Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision (weighted): {precision:.4f}")
    print(f"  Recall (weighted): {recall:.4f}")
    print(f"  F1-score (weighted): {f1:.4f}")
    print(f"  Inference time: {inference_time:.2f}s")
    print(f"  Samples/second: {len(texts)/inference_time:.2f}")
    
    print(f"\nPer-class metrics:")
    for class_name in model.label_encoder.classes_:
        if class_name in class_report:
            metrics = class_report[class_name]
            print(f"  {class_name.capitalize()}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1-score: {metrics['f1-score']:.4f}")
            print(f"    Support: {int(metrics['support'])}")
    
    print(f"\nConfusion Matrix:")
    print(f"  Classes: {list(model.label_encoder.classes_)}")
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


def print_feature_importance(feature_importance, top_n=10):
    """
    Pretty print feature importance analysis.
    
    Args:
        feature_importance: Dictionary from model.get_feature_importance()
        top_n: Number of top features to display
    """
    print(f"\n{'='*60}")
    print("Feature Importance Analysis")
    print(f"{'='*60}")
    
    for class_name, features in feature_importance.items():
        print(f"\n{class_name.upper()}:")
        print("  Top positive features (predict this class):")
        for word, score in features['positive_features'][:top_n]:
            print(f"    {word}: {score:.4f}")
        print("  Top negative features (predict against this class):")
        for word, score in features['negative_features'][:top_n]:
            print(f"    {word}: {score:.4f}")


def setup_output_directory(output_dir, model_name="baseline"):
    """
    Setup output directory with timestamp if not provided.
    
    Args:
        output_dir: Path to output directory or None for auto-generation
        model_name: Name prefix for auto-generated directory
        
    Returns:
        Path object for the output directory
    """
    from pathlib import Path
    from datetime import datetime
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"model_phase/results/{model_name}_{timestamp}"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def init_wandb_if_available(project_name, experiment_name, config, use_wandb=False):
    """
    Initialize WandB if available and requested.
    
    Args:
        project_name: WandB project name
        experiment_name: Name for this experiment run
        config: Configuration dictionary to log
        use_wandb: Whether to use WandB
        
    Returns:
        True if WandB was initialized, False otherwise
    """
    if not use_wandb:
        return False
    
    try:
        import wandb
        wandb.init(
            project=project_name,
            name=experiment_name,
            config=config
        )
        return True
    except ImportError:
        print("⚠️  WandB not available. Install with: pip install wandb")
        return False


def log_to_wandb(metrics, use_wandb=False):
    """
    Log metrics to WandB if initialized.
    
    Args:
        metrics: Dictionary of metrics to log
        use_wandb: Whether WandB is being used
    """
    if use_wandb:
        try:
            import wandb
            wandb.log(metrics)
        except:
            pass


def finish_wandb(use_wandb=False):
    """
    Finish WandB run if initialized.
    
    Args:
        use_wandb: Whether WandB is being used
    """
    if use_wandb:
        try:
            import wandb
            wandb.finish()
        except:
            pass


def save_results_to_json(results, output_path):
    """
    Save results dictionary to JSON file.
    
    Args:
        results: Dictionary of results to save
        output_path: Path to save JSON file
    """
    import json
    from pathlib import Path
    
    output_path = Path(output_path)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {output_path}")


def print_training_summary(results, output_dir):
    """
    Print a summary of training results.
    
    Args:
        results: Dictionary containing training results
        output_dir: Directory where results were saved
    """
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")
    
    # Extract key metrics
    if 'test_accuracy' in results:
        print(f"\nTest Metrics:")
        print(f"  Accuracy: {results['test_accuracy']:.4f}")
        print(f"  Precision: {results['test_precision']:.4f}")
        print(f"  Recall: {results['test_recall']:.4f}")
        print(f"  F1-score: {results['test_f1']:.4f}")
    
    if 'validation_accuracy' in results:
        print(f"\nValidation Metrics:")
        print(f"  Accuracy: {results['validation_accuracy']:.4f}")
        print(f"  F1-score: {results['validation_f1']:.4f}")
    
    if 'training_time' in results:
        print(f"\nTraining Time: {results['training_time']:.2f}s")
