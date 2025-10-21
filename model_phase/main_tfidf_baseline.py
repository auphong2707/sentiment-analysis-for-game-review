"""
Baseline Model: TF-IDF with Logistic Regression for Sentiment Analysis

Core Idea: Sentiment is determined by the frequency of specific words.

How it Works: 
- Uses TF-IDF (Term Frequency-Inverse Document Frequency) to weigh important words
- Applies Logistic Regression for multi-class classification
- Ignores word order but captures word importance

Key Features:
- Fast training and prediction
- Interpretable model (can see which words matter most)
- Good baseline to measure against deep learning models
"""

import sys
import os
from pathlib import Path
import argparse
import json
import pickle
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import multiprocessing

# Import utilities
from model_phase.utilities import (
    load_dataset_from_hf,
    evaluate_classifier,
    print_feature_importance,
    setup_output_directory,
    init_wandb_if_available,
    log_to_wandb,
    finish_wandb,
    save_results_to_json,
    print_training_summary,
    upload_results_to_hf
)


class TFIDFSentimentClassifier:
    """
    Baseline sentiment classifier using TF-IDF and Logistic Regression.
    """
    
    def __init__(self, 
                 max_features=10000,
                 ngram_range=(1, 2),
                 max_iter=1000,
                 random_state=42,
                 n_jobs=None):
        """
        Initialize the classifier.
        
        Args:
            max_features: Maximum number of features for TF-IDF
            ngram_range: Range of n-grams to extract (default: unigrams and bigrams)
            max_iter: Maximum iterations for Logistic Regression
            random_state: Random seed for reproducibility
            n_jobs: Number of CPU cores to use (default: CPU count - 1)
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.max_iter = max_iter
        self.random_state = random_state
        
        # Calculate n_jobs: use CPU count - 1 (leave one for orchestration)
        if n_jobs is None:
            cpu_count = multiprocessing.cpu_count()
            self.n_jobs = max(1, cpu_count - 1)  # At least 1, but leave 1 for orchestration
        else:
            self.n_jobs = n_jobs
        
        # Initialize components
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        
        self.classifier = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            multi_class='multinomial',
            solver='lbfgs',
            n_jobs=self.n_jobs,
            verbose=1
        )
        
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
    def fit(self, texts, labels):
        """
        Train the model on text data.
        
        Args:
            texts: List of review texts
            labels: List of sentiment labels (positive, mixed, negative)
        """
        print("\n[1/3] Encoding labels...")
        y_encoded = self.label_encoder.fit_transform(labels)
        
        print(f"\n[2/3] Extracting TF-IDF features...")
        print(f"  - Max features: {self.max_features}")
        print(f"  - N-gram range: {self.ngram_range}")
        X_tfidf = self.vectorizer.fit_transform(texts)
        print(f"  - Feature matrix shape: {X_tfidf.shape}")
        
        print(f"\n[3/3] Training Logistic Regression...")
        print(f"  - Max iterations: {self.max_iter}")
        print(f"  - CPU cores used: {self.n_jobs}/{multiprocessing.cpu_count()} (saving 1 for orchestration)")
        self.classifier.fit(X_tfidf, y_encoded)
        
        self.is_fitted = True
        print("✓ Training complete!")
        
    def predict(self, texts):
        """
        Predict sentiment for new texts.
        
        Args:
            texts: List of review texts
            
        Returns:
            Predicted sentiment labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction!")
        
        X_tfidf = self.vectorizer.transform(texts)
        y_pred_encoded = self.classifier.predict(X_tfidf)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return y_pred
    
    def predict_proba(self, texts):
        """
        Predict probability for each class.
        
        Args:
            texts: List of review texts
            
        Returns:
            Probability matrix (samples x classes)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction!")
        
        X_tfidf = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X_tfidf)
    
    def get_feature_importance(self, top_n=20):
        """
        Get most important features (words) for each class.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary with class names and their top features
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance!")
        
        feature_names = self.vectorizer.get_feature_names_out()
        importances = {}
        
        for idx, class_name in enumerate(self.label_encoder.classes_):
            # Get coefficients for this class
            coef = self.classifier.coef_[idx]
            
            # Get top positive and negative features
            top_positive_idx = np.argsort(coef)[-top_n:][::-1]
            top_negative_idx = np.argsort(coef)[:top_n]
            
            importances[class_name] = {
                'positive_features': [(feature_names[i], coef[i]) for i in top_positive_idx],
                'negative_features': [(feature_names[i], coef[i]) for i in top_negative_idx]
            }
        
        return importances
    
    def save(self, output_dir):
        """Save model, vectorizer, and label encoder."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        with open(output_dir / 'classifier.pkl', 'wb') as f:
            pickle.dump(self.classifier, f)
        
        with open(output_dir / 'label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save config
        config = {
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'max_iter': self.max_iter,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs
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
        
        model = cls(**config)
        
        with open(output_dir / 'vectorizer.pkl', 'rb') as f:
            model.vectorizer = pickle.load(f)
        
        with open(output_dir / 'classifier.pkl', 'rb') as f:
            model.classifier = pickle.load(f)
        
        with open(output_dir / 'label_encoder.pkl', 'rb') as f:
            model.label_encoder = pickle.load(f)
        
        model.is_fitted = True
        
        print(f"✓ Model loaded from {output_dir}")
        return model


def main(dataset_name, 
         max_features=10000,
         ngram_range=(1, 2),
         max_iter=1000,
         subset=1.0,
         output_dir=None,
         use_wandb=False,
         n_jobs=None,
         upload_to_hf=True,
         hf_repo=None):
    """
    Main training and evaluation pipeline.
    
    Args:
        dataset_name: HuggingFace dataset name
        max_features: Maximum TF-IDF features
        ngram_range: N-gram range for TF-IDF
        max_iter: Max iterations for Logistic Regression
        subset: Fraction of data to use
        output_dir: Directory to save results
        use_wandb: Whether to use WandB for tracking
        n_jobs: Number of CPU cores to use (default: CPU count - 1)
        upload_to_hf: Whether to upload results to HuggingFace Hub
        hf_repo: HuggingFace repository name for results (default: auto-generated)
    """
    print("\n" + "="*60)
    print("TF-IDF + Logistic Regression Baseline")
    print("="*60)
    
    # Display CPU info
    cpu_count = multiprocessing.cpu_count()
    cores_to_use = max(1, cpu_count - 1) if n_jobs is None else n_jobs
    print(f"\nSystem Info:")
    print(f"  Total CPU cores: {cpu_count}")
    print(f"  Cores to use: {cores_to_use} (saving 1 for orchestration)")
    
    # Setup output directory
    output_dir = setup_output_directory(output_dir, model_name="tfidf_baseline")
    
    # Initialize WandB if requested
    wandb_initialized = init_wandb_if_available(
        project_name="game-review-sentiment",
        experiment_name=f"tfidf_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "model": "TF-IDF + Logistic Regression",
            "max_features": max_features,
            "ngram_range": ngram_range,
            "max_iter": max_iter,
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
    print("Initializing model")
    print(f"{'='*60}")
    model = TFIDFSentimentClassifier(
        max_features=max_features,
        ngram_range=ngram_range,
        max_iter=max_iter,
        n_jobs=n_jobs
    )
    
    # Train model
    print(f"\n{'='*60}")
    print("Training model")
    print(f"{'='*60}")
    train_start = time.time()
    model.fit(train_data['text'], train_data['label'])
    train_time = time.time() - train_start
    
    print(f"\n✓ Training completed in {train_time:.2f}s")
    
    # Evaluate on validation set
    val_results = evaluate_classifier(
        model, val_data['text'], val_data['label'], "Validation"
    )
    
    # Evaluate on test set
    test_results = evaluate_classifier(
        model, test_data['text'], test_data['label'], "Test"
    )
    
    # Get and print feature importance
    feature_importance = model.get_feature_importance(top_n=20)
    print_feature_importance(feature_importance, top_n=10)
    
    # Compile all results
    all_results = {
        'model_config': {
            'max_features': max_features,
            'ngram_range': ngram_range,
            'max_iter': max_iter,
            'subset': subset
        },
        'training_time': train_time,
        'dataset_info': {
            'train_size': len(train_data['text']),
            'val_size': len(val_data['text']),
            'test_size': len(test_data['text'])
        },
        **val_results,
        **test_results,
        'feature_importance': {
            class_name: {
                'top_positive': [(word, float(score)) for word, score in features['positive_features'][:20]],
                'top_negative': [(word, float(score)) for word, score in features['negative_features'][:20]]
            }
            for class_name, features in feature_importance.items()
        }
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
            model_name="tfidf_baseline",
            hf_repo_name=hf_repo
        )
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train TF-IDF + Logistic Regression baseline for sentiment analysis'
    )
    parser.add_argument(
        '--dataset', 
        type=str, 
        default=os.getenv('HF_DATASET_NAME'),
        help='HuggingFace dataset name (default: from .env HF_DATASET_NAME)'
    )
    parser.add_argument(
        '--max_features',
        type=int,
        default=10000,
        help='Maximum number of TF-IDF features (default: 10000)'
    )
    parser.add_argument(
        '--ngram_min',
        type=int,
        default=1,
        help='Minimum n-gram size (default: 1)'
    )
    parser.add_argument(
        '--ngram_max',
        type=int,
        default=2,
        help='Maximum n-gram size (default: 2)'
    )
    parser.add_argument(
        '--max_iter',
        type=int,
        default=1000,
        help='Maximum iterations for Logistic Regression (default: 1000)'
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
        '--n_jobs',
        type=int,
        default=None,
        help='Number of CPU cores to use (default: CPU count - 1, leaving 1 for orchestration)'
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
        help='HuggingFace repository name for results (default: auto-generated from username)'
    )
    
    args = parser.parse_args()
    
    # Validate dataset name
    if not args.dataset:
        parser.error("--dataset is required (or set HF_DATASET_NAME in .env)")
    
    # Determine upload setting
    upload_to_hf = args.upload_to_hf and not args.no_upload
    
    main(
        dataset_name=args.dataset,
        max_features=args.max_features,
        ngram_range=(args.ngram_min, args.ngram_max),
        max_iter=args.max_iter,
        subset=args.subset,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        n_jobs=args.n_jobs,
        upload_to_hf=upload_to_hf,
        hf_repo=args.hf_repo
    )
