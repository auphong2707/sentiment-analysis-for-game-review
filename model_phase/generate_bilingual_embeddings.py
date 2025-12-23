"""
bilingual-embedding-base Embedding Generator with Checkpoint Support

This script generates embeddings using Lajavaness/bilingual-embedding-base model and saves them for downstream classifiers.
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
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer

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

MODEL_NAME = 'Lajavaness/bilingual-embedding-base'


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
            'val_embeddings',
            'test_embeddings'
        ]
        
        summary = {
            'completed_stages': completed,
            'remaining_stages': [s for s in all_stages if s not in completed],
            'progress_percentage': len(completed) / len(all_stages) * 100
        }
        return summary


class GameReviewDataset(Dataset):
    """PyTorch Dataset for game reviews."""
    
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


class BilingualEmbeddingGenerator:
    """Generate bilingual-embedding-base embeddings with checkpoint support."""
    
    def __init__(self, 
                 max_length=512,
                 batch_size=256,
                 checkpoint_manager=None):
        self.model_name = MODEL_NAME
        self.max_length = max_length
        self.batch_size = batch_size
        self.checkpoint_manager = checkpoint_manager
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {self.device}")
        
        # Detect GPU architecture for mixed precision optimization
        self.use_amp = False
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            compute_cap = torch.cuda.get_device_capability(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print(f"GPU: {gpu_name}")
            print(f"GPU Memory: {gpu_memory:.2f} GB")
            print(f"Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
            
            # Only enable mixed precision for Volta+ (V100, A100, etc.)
            # P100 (Pascal 6.0) runs SLOWER with FP16 autocast
            if compute_cap[0] >= 7:
                self.use_amp = True
                print(f"⚡ Mixed precision (FP16) ENABLED - Volta+ GPU detected")
            else:
                print(f"⚠️  Mixed precision DISABLED - {gpu_name} runs faster with FP32")
        else:
            print("⚠️  Warning: Running on CPU will be very slow. Use GPU for faster embeddings.")
        
        # Load bilingual-embedding-base model (frozen for embedding extraction)
        print(f"\nLoading bilingual-embedding-base model: {self.model_name}")
        # Use local_files_only=False to allow download, but don't check for optional features
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            clean_up_tokenization_spaces=True,
            use_fast=True,
            trust_remote_code=True
        )
        self.embedding_model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        self.embedding_model.to(self.device)
        self.embedding_model.eval()  # Freeze embedding model
        
        # Optimize with torch.compile for 20-30% speedup (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                print("  Optimizing model with torch.compile...")
                self.embedding_model = torch.compile(self.embedding_model, mode='max-autotune')
                print("  ✓ torch.compile enabled (expect 20-30% speedup after warmup)")
            except Exception as e:
                print(f"  ⚠️  torch.compile failed: {e}")
        else:
            print("  ℹ️  torch.compile not available (requires PyTorch 2.0+)")
        
        # Freeze all parameters
        for param in self.embedding_model.parameters():
            param.requires_grad = False
        
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
            print(f"  Max length: {self.max_length}")
            print(f"\n  ⚡ Performance tip: If speed is critical, reduce --max_length")
            print(f"     Attention is O(n²): 512→256 gives ~4x speedup, 512→128 gives ~16x speedup")
        
        start_time = time.time()
        last_log_time = start_time
        
        # Will detect embedding dimension from first batch and pre-allocate array
        all_embeddings = None
        
        # If resuming, load partial embeddings
        if start_idx > 0 and embeddings:
            partial_emb = embeddings[0]
            embedding_dim = partial_emb.shape[1]
            all_embeddings = np.empty((len(texts), embedding_dim), dtype=np.float32)
            all_embeddings[:len(partial_emb)] = partial_emb
            embeddings = []  # Clear list
        
        # Create batches
        for batch_idx, i in enumerate(range(start_idx, len(texts), self.batch_size)):
            batch_texts = texts[i:i + self.batch_size]
            batch_end = i + len(batch_texts)
            
            # Profiling for first 10 batches
            if batch_idx < 10:
                t_start = time.time()
            
            # Tokenize on CPU (inevitable, but pipelined with GPU via async transfer)
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move to GPU with non_blocking for async transfer (overlaps with previous GPU work)
            inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
            
            # Profiling checkpoint after tokenization
            if batch_idx < 10:
                t_after_tokenize = time.time()
            
            # Generate embeddings with conditional mixed precision
            with torch.inference_mode():
                # Only use autocast for Volta+ GPUs (V100, A100) - P100 runs faster without it
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.embedding_model(**inputs)
                    # Use CLS token embedding - detach to free computation graph immediately
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].float().detach()
            
            # Profiling checkpoint after inference (sync GPU to measure actual time)
            if batch_idx < 10:
                torch.cuda.synchronize()
                t_after_inference = time.time()
            
            # Detect embedding dimension from first batch and allocate array
            if all_embeddings is None:
                embedding_dim = batch_embeddings.shape[1]
                all_embeddings = np.empty((len(texts), embedding_dim), dtype=np.float32)
                print(f"  Detected embedding dimension: {embedding_dim}")
            
            # Immediate GPU-to-CPU transfer (batching didn't help - sync overhead was not the bottleneck)
            # Move to CPU immediately to free GPU memory for next batch
            all_embeddings[i:batch_end] = batch_embeddings.cpu().numpy()
            
            # Profiling checkpoint after CPU transfer
            if batch_idx < 10:
                t_after_transfer = time.time()
            
            # Explicit cleanup - free GPU memory immediately
            del batch_embeddings, outputs, inputs
            if batch_idx % 50 == 0:  # Clear cache every 50 batches to prevent fragmentation
                torch.cuda.empty_cache()
            
            # Profiling: Print detailed timing for first 10 batches
            if batch_idx < 10:
                t_after_cleanup = time.time()
                tokenize_ms = (t_after_tokenize - t_start) * 1000
                inference_ms = (t_after_inference - t_after_tokenize) * 1000
                transfer_ms = (t_after_transfer - t_after_inference) * 1000
                cleanup_ms = (t_after_cleanup - t_after_transfer) * 1000
                total_ms = (t_after_cleanup - t_start) * 1000
                print(f"    [PROFILE Batch {batch_idx}] Total: {total_ms:.1f}ms | "
                      f"Tokenize: {tokenize_ms:.1f}ms | Inference: {inference_ms:.1f}ms | "
                      f"Transfer: {transfer_ms:.1f}ms | Cleanup: {cleanup_ms:.1f}ms")
            
            processed = batch_end
            current_time = time.time()
            
            # Progress logging - only every 5 seconds or at completion to reduce overhead
            if (current_time - last_log_time >= 5.0) or processed == len(texts):
                elapsed = current_time - start_time
                samples_per_sec = processed / elapsed if elapsed > 0 else 0
                eta = (len(texts) - processed) / samples_per_sec if samples_per_sec > 0 else 0
                print(f"  Progress: {processed}/{len(texts)} samples "
                      f"({processed/len(texts)*100:.1f}%) - "
                      f"{samples_per_sec:.1f} samples/s - "
                      f"ETA: {eta:.0f}s")
                last_log_time = current_time
            
            # INCREMENTAL CHECKPOINT: Save every 200 batches to reduce I/O overhead
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
        
        # Convert labels for saving
        if self.label2id is None:
            unique_labels = sorted(set(labels))
            self.label2id = {label: idx for idx, label in enumerate(unique_labels)}
            self.id2label = {idx: label for label, idx in self.label2id.items()}
        
        label_ids = np.array([self.label2id[label] for label in labels])
        
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
        print("Generating bilingual-embedding-base Embeddings with Checkpoint Support")
        print("="*60)
        
        print(f"\nDataset Info:")
        print(f"  Training samples: {len(train_data['text'])}")
        print(f"  Validation samples: {len(val_data['text'])}")
        print(f"  Test samples: {len(test_data['text'])}")
        
        # Create label mapping from training data
        train_dataset = GameReviewDataset(train_data['text'], train_data['label'])
        self.label2id = train_dataset.label2id
        self.id2label = train_dataset.id2label
        
        print(f"  Classes: {list(self.label2id.keys())}")
        
        # Generate train embeddings
        X_train = self.generate_embeddings_with_checkpoint(
            train_data['text'], train_data['label'], 'train_embeddings', "Training"
        )
        
        # Generate validation embeddings
        X_val = self.generate_embeddings_with_checkpoint(
            val_data['text'], val_data['label'], 'val_embeddings', "Validation"
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
         max_length=512,
         batch_size=32,
         subset=1.0,
         output_dir=None,
         use_wandb=False,
         experiment_name=None,
         resume_from=None):
    """Main embedding generation pipeline with checkpoint support."""
    
    print("\n" + "="*60)
    print("bilingual-embedding-base Embedding Generation with Checkpoint Support")
    print("="*60)
    
    # Setup output directory
    if output_dir is None:
        if experiment_name:
            output_dir = f'model_phase/results/{experiment_name}'
        else:
            output_dir = f'model_phase/results/bilingual_embeddings_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
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
            run_name = experiment_name or f"bilingual_embeddings_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
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
    print("Initializing bilingual-embedding-base Embedding Generator")
    print(f"{'='*60}")

    generator = BilingualEmbeddingGenerator(
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
            'subset': subset
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
    print(f"\nUse these embeddings with train_xgboost.py:")
    print(f"  python model_phase/train_xgboost.py --checkpoint_dir {checkpoint_dir}")
    
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate bilingual-embedding-base embeddings with checkpoint support')
    parser.add_argument('--dataset', type=str, default=os.getenv('HF_DATASET_NAME'),
                        help='HuggingFace dataset name')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for embedding generation')
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
