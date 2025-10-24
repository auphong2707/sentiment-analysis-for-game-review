"""
Baseline Model: Recurrent Neural Network (LSTM) with Word Embeddings for Sentiment Analysis

Core Idea: The sequence of words matters.

How it Works:
- Reads text word-by-word using pre-trained word embeddings (e.g., GloVe)
- LSTM "remembers" previous words to understand the context of later ones
- Processes text sequentially, capturing negations and context-dependent meaning

Key Features:
- Understands context and word order
- Captures sequential dependencies
- More complex than TF-IDF baseline
- Requires GPU for efficient training
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
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from tqdm import tqdm

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


# ====================================================
# Dataset Class
# ====================================================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label_encoder, max_len=200):
        self.texts = texts
        self.labels = label_encoder.transform(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.texts[idx])
        tokens = tokens[:self.max_len]
        pad_len = self.max_len - len(tokens)
        tokens = tokens + [0] * pad_len  # pad with zeros
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


# ====================================================
# LSTM Model Architecture
# ====================================================
class LSTMModel(nn.Module):
    """
    Revised LSTM Model Architecture requested by user:

    1. Embedding Layer:
       - vocab_size (provided)
       - embedding dim = 128 (trainable)

    2. Two stacked Bidirectional LSTM layers:
       - each LSTM has hidden_dim units per direction (128)
       - dropout between layers handled by LSTM dropout param

    3. Fully connected layers:
       - Dense(64, ReLU) -> Dropout
       - Dense(64, ReLU) -> Dropout
       - Dense(16, ReLU) -> Dropout

    4. Output: Dense(1, Sigmoid) for binary classification
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, dropout_rate=0.5):
        super().__init__()

        # Embedding layer (trainable)
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Two stacked bidirectional LSTM layers. Setting num_layers=2 and
        # bidirectional=True results in 2 layers with forward/backward directions.
        # dropout parameter applies between LSTM layers when num_layers > 1.
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate
        )

        # Dropout used between dense layers
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected classification head
        # The LSTM final hidden for bidirectional last layer will be hidden_dim * 2
        fc_in = hidden_dim * 2
        self.fc1 = nn.Linear(fc_in, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 16)
        self.output = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch, seq_len]
        emb = self.embedding(x)  # [batch, seq_len, embed_dim]

        # LSTM output: lstm_out [batch, seq_len, hidden_dim * num_directions]
        # h_n: [num_layers * num_directions, batch, hidden_dim]
        lstm_out, (h_n, c_n) = self.lstm(emb)

        # Get the last layer's forward and backward hidden states and concatenate
        # For num_layers=2 and bidirectional=True:
        # h_n shape = (4, batch, hidden_dim) -> layers: [l0_f, l0_b, l1_f, l1_b]
        # We want the top layer (l1_f and l1_b) => indices -2 and -1
        forward_h = h_n[-2]
        backward_h = h_n[-1]
        hidden = torch.cat((forward_h, backward_h), dim=1)  # [batch, hidden_dim*2]

        # Fully connected head with dropout between layers
        x = self.fc1(hidden)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)

        out = self.output(x)
        out = self.sigmoid(out)
        return out


class LSTMSentimentClassifier:
    """
    LSTM-based sentiment classifier following paper architecture.
    """
    
    def __init__(self, 
                 embed_dim=128,
                 hidden_dim=128,
                 learning_rate=1e-3,
                 batch_size=128,
                 epochs=5,
                 max_len=200,
                 vocab_size=73738,
                 dropout_rate=0.5,
                 fc_sizes=(64, 64, 16),
                 random_state=42):
        """
        Initialize the LSTM classifier following paper specifications.
        
        Args:
            embed_dim: Embedding dimension (default: 128)
            hidden_dim: LSTM hidden dimension (128 per direction)
            learning_rate: Learning rate for Adam optimizer (default: 0.001)
            batch_size: Batch size for training (default: 128)
            epochs: Number of training epochs (default: 5)
            max_len: Maximum sequence length (200)
            vocab_size: Maximum vocabulary size (default: 73738)
            dropout_rate: Dropout rate (0.5)
            fc_sizes: Tuple of fully-connected layer sizes (64,64,16)
            random_state: Random seed for reproducibility
        """
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate
        self.fc_sizes = tuple(fc_sizes)
        self.random_state = random_state
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Initialize components
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = SimpleTokenizer()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.is_fitted = False
        
        print(f"✓ LSTM Classifier initialized (device: {self.device})")
        
    def fit(self, texts, labels):
        """
        Train the LSTM model on text data.
        
        Args:
            texts: List of review texts
            labels: List of sentiment labels
        """
        print("\n[1/4] Encoding labels...")
        y_encoded = self.label_encoder.fit_transform(labels)
        num_classes = len(self.label_encoder.classes_)
        print(f"  - Classes: {list(self.label_encoder.classes_)}")
        
        print(f"\n[2/4] Building vocabulary...")
        self.tokenizer.build_vocab(texts, max_size=self.vocab_size)
        vocab_size = len(self.tokenizer) + 1  # +1 for padding token
        
        print(f"\n[3/4] Creating datasets and dataloaders...")
        dataset = TextDataset(texts, labels, self.tokenizer, self.label_encoder, self.max_len)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        print(f"  - Dataset size: {len(dataset)}")
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - Number of batches: {len(dataloader)}")
        
        print(f"\n[4/4] Training LSTM model...")
        print(f"  - Vocab size: {vocab_size}")
        print(f"  - Embed dim: {self.embed_dim}")
        print(f"  - Hidden dim: {self.hidden_dim}")
        print(f"  - Dropout rate: {self.dropout_rate}")
        print(f"  - FC sizes: {self.fc_sizes}")
        print(f"  - Epochs: {self.epochs}")
        print(f"  - Device: {self.device}")
        
        # Initialize model with new architecture
        self.model = LSTMModel(
            vocab_size=vocab_size,
            embed_dim=self.embed_dim, 
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Use appropriate loss function based on number of classes
        if num_classes == 2:
            # Binary classification: use BCELoss (model outputs sigmoid)
            criterion = nn.BCELoss()
            print(f"  - Using Binary Cross Entropy Loss for binary classification")
        else:
            # Multi-class: use CrossEntropyLoss (need to modify model output)
            criterion = nn.CrossEntropyLoss() 
            print(f"  - Using Cross Entropy Loss for {num_classes}-class classification")
        
        # Training loop
        for epoch in range(1, self.epochs + 1):
            total_loss = 0
            self.model.train()
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{self.epochs}")
            for inputs, targets in progress_bar:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                
                if num_classes == 2:
                    # For binary classification, convert targets to float and reshape
                    targets = targets.float().unsqueeze(1)
                    loss = criterion(outputs, targets)
                else:
                    # For multi-class, use as-is
                    loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(dataloader)
            print(f"  Epoch {epoch}: Average Loss = {avg_loss:.4f}")
        
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
        
        dataset = TextDataset(texts, ['positive'] * len(texts), self.tokenizer, self.label_encoder, self.max_len)  # dummy labels
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        self.model.eval()
        predictions = []
        num_classes = len(self.label_encoder.classes_)
        
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                
                if num_classes == 2:
                    # Binary classification: threshold sigmoid output at 0.5
                    preds = (outputs > 0.5).float().squeeze().cpu().numpy()
                else:
                    # Multi-class: use argmax
                    preds = outputs.argmax(dim=1).cpu().numpy()
                    
                predictions.extend(preds)
        
        return self.label_encoder.inverse_transform(predictions.astype(int))
    
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
        
        dataset = TextDataset(texts, ['positive'] * len(texts), self.tokenizer, self.label_encoder, self.max_len)  # dummy labels
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        self.model.eval()
        probabilities = []
        num_classes = len(self.label_encoder.classes_)
        
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                
                if num_classes == 2:
                    # Binary classification: sigmoid outputs probabilities
                    pos_probs = outputs.squeeze().cpu().numpy()
                    neg_probs = 1 - pos_probs
                    probs = np.column_stack([neg_probs, pos_probs])  # [negative, positive]
                else:
                    # Multi-class: apply softmax
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    
                probabilities.extend(probs)
        
        return np.array(probabilities)
    
    def save(self, output_dir):
        """Save model, tokenizer, and label encoder."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state dict
        torch.save(self.model.state_dict(), output_dir / 'model.pt')
        
        # Save tokenizer
        with open(output_dir / 'tokenizer.pkl', 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        # Save label encoder
        with open(output_dir / 'label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save config
        config = {
            'embed_dim': self.embed_dim,
            'hidden_dim': self.hidden_dim,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'max_len': self.max_len,
            'vocab_size': self.vocab_size,
            'dropout_rate': self.dropout_rate,
            'fc_sizes': list(self.fc_sizes),
            'random_state': self.random_state,
            'num_classes': len(self.label_encoder.classes_)
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
        
        model = cls(**{k: v for k, v in config.items() if k != 'num_classes'})
        
        with open(output_dir / 'tokenizer.pkl', 'rb') as f:
            model.tokenizer = pickle.load(f)
        
        with open(output_dir / 'label_encoder.pkl', 'rb') as f:
            model.label_encoder = pickle.load(f)
        
        # Reconstruct model architecture
        vocab_size = len(model.tokenizer) + 1
        model.model = LSTMModel(
            vocab_size=vocab_size,
            embed_dim=model.embed_dim,
            hidden_dim=model.hidden_dim,
            dropout_rate=model.dropout_rate
        ).to(model.device)
        
        # Load model weights
        model.model.load_state_dict(torch.load(output_dir / 'model.pt', map_location=model.device))
        model.is_fitted = True
        
        print(f"✓ Model loaded from {output_dir}")
        return model


# ====================================================
# Simple Tokenizer (can be replaced by pretrained tokenizer)
# ====================================================
class SimpleTokenizer:
    def __init__(self, vocab=None):
        self.word2idx = vocab or {}
        self.idx2word = {i: w for w, i in self.word2idx.items()}

    def build_vocab(self, texts, max_size=20000):
        from collections import Counter
        counter = Counter()
        for text in texts:
            counter.update(text.lower().split())
        most_common = counter.most_common(max_size)
        self.word2idx = {w: i + 1 for i, (w, _) in enumerate(most_common)}  # 0 reserved for padding
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        print(f"✓ Vocabulary built: {len(self.word2idx)} words")

    def __call__(self, text):
        return [self.word2idx.get(w, 0) for w in text.lower().split()]

    def __len__(self):
        return len(self.word2idx)


def main(dataset_name, 
         embed_dim=128,
         hidden_dim=128,
         learning_rate=1e-3,
         batch_size=128,
         epochs=5,
         max_len=200,
         vocab_size=73738,
         dropout_rate=0.5,
         fc_sizes=(64,64,16),
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
        embed_dim: Embedding dimension (default: 128)
        hidden_dim: LSTM hidden dimension (128 per direction)
        learning_rate: Learning rate for Adam optimizer (default: 0.001)
        batch_size: Batch size for training (default: 128)
        epochs: Number of training epochs (default: 5)
        max_len: Maximum sequence length (200)
        vocab_size: Maximum vocabulary size (default: 73738)
        dropout_rate: Dropout rate (0.5)
        fc_sizes: Fully-connected layer sizes (64,64,16)
        subset: Fraction of data to use
        output_dir: Directory to save results
        use_wandb: Whether to use WandB for tracking
        n_jobs: Number of CPU cores to use (default: CPU count - 1)
        upload_to_hf: Whether to upload results to HuggingFace Hub
        hf_repo: HuggingFace repository name for results (default: auto-generated)
    """
    print("\n" + "="*60)
    print("LSTM + Word Embeddings Baseline")
    print("="*60)
    
    # Display device info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpu_count = torch.get_num_threads() if hasattr(torch, 'get_num_threads') else 1
    cores_to_use = max(1, cpu_count - 1) if n_jobs is None else n_jobs
    print(f"\nSystem Info:")
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    if n_jobs is not None:
        print(f"  CPU cores to use: {cores_to_use}")
    
    # Setup output directory
    output_dir = setup_output_directory(output_dir, model_name="lstm_baseline")
    
    # Initialize WandB if requested
    wandb_initialized = init_wandb_if_available(
        project_name="game-review-sentiment",
        experiment_name=f"lstm_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "model": "Bidirectional LSTM (Paper Architecture)",
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "max_len": max_len,
            "vocab_size": vocab_size,
            "dropout_rate": dropout_rate,
            "fc_sizes": fc_sizes,
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
    model = LSTMSentimentClassifier(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        max_len=max_len,
        vocab_size=vocab_size,
        dropout_rate=dropout_rate,
        fc_sizes=fc_sizes
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
    
    # Compile all results
    all_results = {
        'model_config': {
            'embed_dim': embed_dim,
            'hidden_dim': hidden_dim,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs,
            'max_len': max_len,
            'vocab_size': vocab_size,
            'dropout_rate': dropout_rate,
            'fc_sizes': fc_sizes,
            'subset': subset
        },
        'training_time': train_time,
        'dataset_info': {
            'train_size': len(train_data['text']),
            'val_size': len(val_data['text']),
            'test_size': len(test_data['text'])
        },
        'model_info': {
            'device': device,
            'vocab_size_actual': len(model.tokenizer),
            'total_parameters': sum(p.numel() for p in model.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.model.parameters() if p.requires_grad)
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
            model_name="lstm_baseline",
            hf_repo_name=hf_repo
        )
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train LSTM + Word Embeddings baseline for sentiment analysis'
    )
    parser.add_argument(
        '--dataset', 
        type=str, 
        default=os.getenv('HF_DATASET_NAME'),
        help='HuggingFace dataset name (default: from .env HF_DATASET_NAME)'
    )
    parser.add_argument(
        '--embed_dim',
        type=int,
        default=128,
        help='Word embedding dimension (default: 128)'
    )
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=128,
        help='LSTM hidden dimension (default: 128)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size for training (default: 128)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of training epochs (default: 5)'
    )
    parser.add_argument(
        '--max_len',
        type=int,
        default=200,
        help='Maximum sequence length (default: 200)'
    )
    parser.add_argument(
        '--vocab_size',
        type=int,
        default=73738,
        help='Maximum vocabulary size (default: 73738)'
    )
    parser.add_argument(
        '--dropout_rate',
        type=float,
        default=0.5,
        help='Dropout rate (default: 0.5)'
    )
    parser.add_argument(
        '--n_jobs',
        type=int,
        default=None,
        help='Number of CPU cores to use (default: CPU count - 1, leaving 1 for orchestration)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='Learning rate for Adam optimizer (default: 0.001)'
    )
    parser.add_argument(
        '--fc_sizes',
        type=str,
        default='64,64,16',
        help='Comma-separated fully-connected layer sizes, e.g. "64,64,16" (default)'
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
        help='HuggingFace repository name for results (default: auto-generated from username)'
    )
    
    args = parser.parse_args()
    
    # Validate dataset name
    if not args.dataset:
        parser.error("--dataset is required (or set HF_DATASET_NAME in .env)")
    
    # Determine upload setting
    upload_to_hf = args.upload_to_hf and not args.no_upload
    
    # parse fc_sizes string into tuple of ints
    fc_sizes_parsed = tuple(int(x) for x in args.fc_sizes.split(',')) if hasattr(args, 'fc_sizes') and args.fc_sizes else (64,64,16)

    main(
        dataset_name=args.dataset,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_len=args.max_len,
        vocab_size=args.vocab_size,
        dropout_rate=args.dropout_rate,
        fc_sizes=fc_sizes_parsed,
        subset=args.subset,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        n_jobs=args.n_jobs,
        upload_to_hf=upload_to_hf,
        hf_repo=args.hf_repo
    )
