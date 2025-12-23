"""
EmbeddingGemma-300m Embedding Generator with Checkpoint Support

This script generates embeddings using Google's embeddinggemma-300m model and saves them for downstream classifiers.
Supports checkpointing at every major stage:
1. Data loading
2. Embedding generation (train/val/test)

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
from sentence_transformers import SentenceTransformer

# Import utilities
from model_phase.utilities import (
    load_dataset_from_hf,
    init_wandb_if_available,
    finish_wandb
)

# Import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

MODEL_NAME = 'google/embeddinggemma-300m'


class CheckpointManager:
    """Manages checkpoints for resumable embedding generation."""
    
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
    
    def get_progress_summary(self):
        """Get a summary of embedding generation progress."""
        completed = self.state['completed_stages']
        all_stages = [
            'data_loaded',
            'train_embeddings',
            'validation_embeddings',
            'test_embeddings'
        ]
        
        summary = {
            'completed_stages': completed,
            'remaining_stages': [s for s in all_stages if s not in completed],
            'progress_percentage': len(completed) / len(all_stages) * 100
        }
        return summary


class GemmaEmbeddingGenerator:
    """Generate EmbeddingGemma-300m embeddings with checkpoint support."""
    
    def __init__(self, 
                 max_length=256,
                 batch_size=128,
                 checkpoint_manager=None):
        self.model_name = MODEL_NAME
        self.max_length = max_length
        self.batch_size = batch_size
        self.checkpoint_manager = checkpoint_manager
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {self.device}")
        
        # GPU info
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print(f"GPU: {gpu_name}")
            print(f"GPU Memory: {gpu_memory:.2f} GB")
            print(f"⚠️  Note: EmbeddingGemma does NOT support float16")
            print(f"    Using float32 for P100 GPU (optimal for this architecture)")
        else:
            print("⚠️  Warning: Running on CPU will be very slow. Use GPU for faster embeddings.")
        
        # Load EmbeddingGemma-300m model
        print(f"\nLoading EmbeddingGemma model: {self.model_name}")
        print(f"  ℹ️  Note: This model requires accepting Google's license on HuggingFace")
        print(f"  ℹ️  Run: huggingface-cli login (if you haven't already)")
        
        self.model = SentenceTransformer(self.model_name, device=str(self.device))
        
        # Set max sequence length (model supports up to 2048 tokens)
        self.model.max_seq_length = max_length
        print(f"  ✓ Model loaded successfully")
        print(f"  ✓ Max sequence length set to: {max_length} tokens (model max: 2048)")
        print(f"  ✓ Embedding dimension: 768")
        
        self.label2id = None
        self.id2label = None
        
    def generate_embeddings_with_checkpoint(self, texts, labels, stage_name, desc="Encoding"):
        """Generate embeddings for texts with checkpoint support."""
        
        # Check if embeddings already exist
        if self.checkpoint_manager and self.checkpoint_manager.is_stage_completed(stage_name):
            print(f"\n✓ Stage '{stage_name}' already completed, loading from checkpoint...")
            embeddings, saved_labels = self.checkpoint_manager.load_embeddings(stage_name)
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
        last_log_time = start_time
        
        # Pre-allocate array if resuming
        all_embeddings = None
        if start_idx > 0 and embeddings:
            partial_emb = embeddings[0]
            embedding_dim = partial_emb.shape[1]
            all_embeddings = np.empty((len(texts), embedding_dim), dtype=np.float32)
            all_embeddings[:len(partial_emb)] = partial_emb
            embeddings = []  # Clear list
        
        # Process in batches
        for batch_idx, i in enumerate(range(start_idx, len(texts), self.batch_size)):
            batch_texts = texts[i:i + self.batch_size]
            batch_end = i + len(batch_texts)
            
            # Profiling for first 10 batches
            if batch_idx < 10:
                t_start = time.time()
            
            # Generate embeddings using Sentence Transformers
            # encode_document() handles tokenization internally
            # Returns numpy array by default
            batch_embeddings = self.model.encode(
                batch_texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,  # Keep raw embeddings
                device=str(self.device)
            )
            
            # Profiling checkpoint after encoding
            if batch_idx < 10:
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                t_after_encoding = time.time()
            
            # Allocate array on first batch
            if all_embeddings is None:
                embedding_dim = batch_embeddings.shape[1]
                all_embeddings = np.empty((len(texts), embedding_dim), dtype=np.float32)
                print(f"  Detected embedding dimension: {embedding_dim}")
            
            # Store embeddings
            all_embeddings[i:batch_end] = batch_embeddings
            
            # Profiling for first 10 batches
            if batch_idx < 10:
                t_after_store = time.time()
                encoding_ms = (t_after_encoding - t_start) * 1000
                store_ms = (t_after_store - t_after_encoding) * 1000
                total_ms = (t_after_store - t_start) * 1000
                print(f"    [PROFILE Batch {batch_idx}] Total: {total_ms:.1f}ms | "
                      f"Encoding: {encoding_ms:.1f}ms | Store: {store_ms:.1f}ms")
            
            # Clear GPU cache periodically
            if self.device.type == 'cuda' and batch_idx % 50 == 0:
                torch.cuda.empty_cache()
            
            processed = batch_end
            current_time = time.time()
            
            # Progress logging - every 5 seconds or at completion
            if (current_time - last_log_time >= 5.0) or processed == len(texts):
                elapsed = current_time - start_time
                samples_per_sec = processed / elapsed if elapsed > 0 else 0
                eta = (len(texts) - processed) / samples_per_sec if samples_per_sec > 0 else 0
                print(f"  Progress: {processed}/{len(texts)} samples "
                      f"({processed/len(texts)*100:.1f}%) - "
                      f"{samples_per_sec:.1f} samples/s - "
                      f"ETA: {eta:.0f}s")
                last_log_time = current_time
            
            # INCREMENTAL CHECKPOINT: Save every 200 batches
            if self.checkpoint_manager and (i // self.batch_size + 1) % 200 == 0:
                np.savez_compressed(
                    partial_file,
                    embeddings=all_embeddings[:processed],
                    last_index=processed
                )
                print(f"  💾 Partial checkpoint saved at {processed}/{len(texts)} samples")
        
        embeddings = all_embeddings
        elapsed = time.time() - start_time
        
        print(f"  ✓ Embeddings generated: {embeddings.shape}")
        print(f"  ✓ Time: {elapsed:.2f}s ({len(texts)/elapsed:.1f} samples/s)")
        
        # Create label mapping
        if self.label2id is None:
            unique_labels = sorted(set(labels))
            self.label2id = {label: idx for idx, label in enumerate(unique_labels)}
            self.id2label = {idx: label for label, idx in self.label2id.items()}
        
        # Save final checkpoint and clean up partial
        if self.checkpoint_manager:
            # Save with string labels (original format for compatibility)
            self.checkpoint_manager.save_embeddings(stage_name, embeddings, labels=np.array(labels))
            self.checkpoint_manager.mark_stage_completed(
                stage_name, 
                {'shape': list(embeddings.shape), 'time': elapsed}
            )
            # Delete partial checkpoint
            if partial_file and partial_file.exists():
                partial_file.unlink()
                print(f"  🗑️  Cleaned up partial checkpoint")
        
        return embeddings
    
    def generate_all_embeddings(self, train_data, val_data, test_data, use_wandb=False):
        """Generate embeddings for all splits with checkpoint support."""
        print("\n" + "="*60)
        print("Generating EmbeddingGemma-300m Embeddings with Checkpoint Support")
        print("="*60)
        
        print(f"\nDataset Info:")
        print(f"  Training samples: {len(train_data['text'])}")
        print(f"  Validation samples: {len(val_data['text'])}")
        print(f"  Test samples: {len(test_data['text'])}")
        
        # Create label mapping from training data
        unique_labels = sorted(set(train_data['label']))
        self.label2id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        
        print(f"  Classes: {list(self.label2id.keys())}")
        
        # Generate train embeddings
        X_train = self.generate_embeddings_with_checkpoint(
            train_data['text'], train_data['label'], 'train_embeddings', "Training"
        )
        
        # Generate validation embeddings
        X_val = self.generate_embeddings_with_checkpoint(
            val_data['text'], val_data['label'], 'validation_embeddings', "Validation"
        )
        
        # Generate test embeddings
        X_test = self.generate_embeddings_with_checkpoint(
            test_data['text'], test_data['label'], 'test_embeddings', "Test"
        )
        
        print("\n" + "="*60)
        print("Embedding Generation Complete!")
        print("="*60)
        print(f"  Train embeddings: {X_train.shape}")
        print(f"  Validation embeddings: {X_val.shape}")
        print(f"  Test embeddings: {X_test.shape}")
        
        return {
            'train': X_train,
            'val': X_val,
            'test': X_test
        }


def main(dataset_name,
         max_length=256,
         batch_size=128,
         subset=1.0,
         output_dir=None,
         use_wandb=False,
         experiment_name=None,
         resume_from=None):
    """Main embedding generation pipeline with checkpoint support."""
    
    print("\n" + "="*60)
    print("EmbeddingGemma-300m Embedding Generation with Checkpoint Support")
    print("="*60)
    
    # Setup output directory (backward compatible with previous embeddings)
    if output_dir is None:
        if experiment_name:
            output_dir = f'model_phase/results/{experiment_name}'
        else:
            # Use fixed directory name for backward compatibility
            output_dir = 'model_phase/results/embeddings'
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # AUTO-DETECT checkpoint from Kaggle input
    checkpoint_dir = output_dir / 'checkpoints'
    kaggle_checkpoint_found = False
    
    # Check if running on Kaggle and has checkpoint dataset
    if Path('/kaggle/input').exists():
        print("\n🔍 Searching for checkpoints in /kaggle/input/...")
        
        # Find checkpoint in input datasets
        for input_subdir in Path('/kaggle/input').iterdir():
            if input_subdir.is_dir():
                potential_checkpoint = input_subdir / 'checkpoints'
                checkpoint_state_file = potential_checkpoint / 'checkpoint_state.json'
                
                if checkpoint_state_file.exists():
                    print(f"✅ Found checkpoint in: {potential_checkpoint}")
                    
                    # Copy checkpoint to working directory
                    import shutil
                    if checkpoint_dir.exists():
                        shutil.rmtree(checkpoint_dir)
                    shutil.copytree(potential_checkpoint, checkpoint_dir)
                    
                    kaggle_checkpoint_found = True
                    print(f"✅ Checkpoint copied to: {checkpoint_dir}")
                    break
        
        if not kaggle_checkpoint_found:
            print("ℹ️  No checkpoint found in /kaggle/input/")
            print("   Starting fresh embedding generation...")
    
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
            run_name = experiment_name or f"gemma_embeddings_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            wandb.init(
                project=wandb_project,
                name=run_name,
                config={
                    "model": MODEL_NAME,
                    "max_length": max_length,
                    "batch_size": batch_size,
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
    
    # Mark as completed for progress tracking
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
    
    # Initialize embedding generator
    print(f"\n{'='*60}")
    print("Initializing EmbeddingGemma-300m Embedding Generator")
    print(f"{'='*60}")

    generator = GemmaEmbeddingGenerator(
        max_length=max_length,
        batch_size=batch_size,
        checkpoint_manager=checkpoint_manager
    )
    
    # Generate embeddings (with checkpoint)
    embedding_start = time.time()
    embeddings = generator.generate_all_embeddings(
        train_data,
        val_data,
        test_data,
        use_wandb=wandb_initialized
    )
    total_time = time.time() - embedding_start
    
    # Save metadata
    metadata = {
        'model_config': {
            'model_name': MODEL_NAME,
            'max_length': max_length,
            'batch_size': batch_size,
            'subset': subset,
            'embedding_dimension': 768
        },
        'embedding_generation_time': total_time,
        'dataset_info': {
            'train_size': len(train_data['text']),
            'val_size': len(val_data['text']),
            'test_size': len(test_data['text'])
        },
        'embedding_shapes': {
            'train': list(embeddings['train'].shape),
            'val': list(embeddings['val'].shape),
            'test': list(embeddings['test'].shape)
        },
        'label_mapping': {
            'label2id': generator.label2id,
            'id2label': generator.id2label
        }
    }
    
    # Save metadata
    with open(output_dir / 'embedding_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n✓ Metadata saved to {output_dir / 'embedding_metadata.json'}")
    
    # Finish WandB
    if wandb_initialized:
        finish_wandb(use_wandb=True)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Embedding Generation Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"\nTotal time: {total_time/60:.2f} minutes")
    print(f"\nGenerated embeddings:")
    print(f"  - {checkpoint_dir}/train_embeddings_embeddings.npz")
    print(f"  - {checkpoint_dir}/validation_embeddings_embeddings.npz")
    print(f"  - {checkpoint_dir}/test_embeddings_embeddings.npz")
    print(f"\nUse these embeddings with main_xgboost.py:")
    print(f"  python model_phase/main_xgboost.py --checkpoint_dir {checkpoint_dir}")
    
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate EmbeddingGemma-300m embeddings with checkpoint support')
    parser.add_argument('--dataset', type=str, default=os.getenv('HF_DATASET_NAME'),
                        help='HuggingFace dataset name')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Maximum sequence length (256 recommended, model supports up to 2048 tokens)')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for embedding generation (optimized for 16GB GPU)')
    parser.add_argument('--subset', type=float, default=1.0,
                        help='Fraction of data to use')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use WandB for tracking')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Custom experiment name')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume from checkpoint directory')
    
    args = parser.parse_args()
    
    if not args.dataset:
        parser.error("--dataset is required (or set HF_DATASET_NAME in .env)")
    
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
        subset=args.subset,
        output_dir=output_dir,
        use_wandb=args.use_wandb,
        experiment_name=args.experiment_name,
        resume_from=args.resume_from
    )
