"""
Add labels to existing checkpoint files that only contain embeddings.

This script loads embeddings from checkpoint, loads corresponding labels from
HuggingFace dataset, validates they match, and saves new checkpoint with both.
"""

import numpy as np
import argparse
from pathlib import Path
from datasets import load_dataset
import json
import os


def add_labels_to_checkpoint(checkpoint_dir, dataset_name, split_name):
    """
    Add labels to checkpoint for a specific split.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        dataset_name: Name of HuggingFace dataset
        split_name: Split name (train/validation/test)
    """
    checkpoint_dir = Path(checkpoint_dir)
    stage_name = f'{split_name}_embeddings'
    
    # 1. Load embeddings from checkpoint
    embed_file = checkpoint_dir / f'{stage_name}_embeddings.npz'
    if not embed_file.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embed_file}")
    
    print(f"\n{'='*70}")
    print(f"Processing: {split_name}")
    print(f"{'='*70}")
    print(f"📂 Loading embeddings from: {embed_file}")
    
    data = np.load(embed_file, allow_pickle=True)
    embeddings = data['embeddings']
    
    print(f"  ✓ Embeddings shape: {embeddings.shape}")
    print(f"  ✓ Number of samples: {len(embeddings)}")
    
    # Check if labels already exist
    if 'labels' in data:
        print(f"  ⚠️  Labels already exist in checkpoint!")
        response = input("  Do you want to overwrite? (yes/no): ")
        if response.lower() != 'yes':
            print(f"  ⏭️  Skipping {split_name}")
            return False
    
    # 2. Load labels from HuggingFace dataset
    print(f"\n📥 Loading labels from HuggingFace dataset...")
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        print("  ✓ Using HF_TOKEN")
        dataset = load_dataset(dataset_name, split=split_name, token=hf_token)
    else:
        print("  ℹ️  Loading public dataset (no HF_TOKEN)")
        dataset = load_dataset(dataset_name, split=split_name)
    
    labels = np.array(dataset['review_category'])
    
    print(f"  ✓ Labels shape: {labels.shape}")
    print(f"  ✓ Number of labels: {len(labels)}")
    print(f"  ✓ Unique labels: {sorted(np.unique(labels).tolist())}")
    
    # 3. Validate embeddings and labels match
    print(f"\n🔍 Validating...")
    if len(embeddings) != len(labels):
        raise ValueError(
            f"❌ MISMATCH!\n"
            f"  Embeddings: {len(embeddings)} samples\n"
            f"  Labels: {len(labels)} samples\n"
            f"  Checkpoint may have been created from a subset of dataset!"
        )
    
    print(f"  ✓ Validation passed: {len(embeddings)} samples match")
    
    # 4. Save new checkpoint with both embeddings and labels
    backup_file = checkpoint_dir / f'{stage_name}_embeddings.backup.npz'
    new_file = checkpoint_dir / f'{stage_name}_embeddings_with_labels.npz'
    
    print(f"\n💾 Saving new checkpoint...")
    
    # Create backup of original file
    print(f"  📦 Creating backup: {backup_file.name}")
    if not backup_file.exists():
        data_copy = {key: data[key] for key in data.files}
        np.savez_compressed(backup_file, **data_copy)
    
    # Save new file with embeddings + labels
    print(f"  💾 Saving with labels: {new_file.name}")
    np.savez_compressed(new_file, embeddings=embeddings, labels=labels)
    
    print(f"  ✓ Saved successfully!")
    
    return True


def replace_checkpoint_files(checkpoint_dir, dry_run=False):
    """
    Replace original checkpoint files with new ones that contain labels.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        dry_run: If True, only show what would be done without actually doing it
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    print(f"\n{'='*70}")
    print(f"{'DRY RUN: ' if dry_run else ''}Replacing checkpoint files...")
    print(f"{'='*70}")
    
    splits = ['train', 'validation', 'test']
    for split_name in splits:
        stage_name = f'{split_name}_embeddings'
        
        original_file = checkpoint_dir / f'{stage_name}_embeddings.npz'
        new_file = checkpoint_dir / f'{stage_name}_embeddings_with_labels.npz'
        
        if not new_file.exists():
            print(f"  ⏭️  Skipping {split_name}: new file not found")
            continue
        
        print(f"\n  📝 {split_name}:")
        print(f"    Original: {original_file.name}")
        print(f"    New: {new_file.name}")
        
        if dry_run:
            print(f"    [DRY RUN] Would replace original with new file")
        else:
            # Original already backed up, safe to replace
            import shutil
            shutil.move(str(new_file), str(original_file))
            print(f"    ✓ Replaced!")
    
    if not dry_run:
        print(f"\n✅ All checkpoint files updated successfully!")
        print(f"\n💡 Backup files are kept in case you need them:")
        for split_name in splits:
            stage_name = f'{split_name}_embeddings'
            backup_file = checkpoint_dir / f'{stage_name}_embeddings.backup.npz'
            if backup_file.exists():
                print(f"  - {backup_file.name}")


def verify_checkpoints(checkpoint_dir):
    """Verify all checkpoint files contain both embeddings and labels."""
    checkpoint_dir = Path(checkpoint_dir)
    
    print(f"\n{'='*70}")
    print(f"Verifying checkpoint files...")
    print(f"{'='*70}")
    
    splits = ['train', 'validation', 'test']
    all_good = True
    
    for split_name in splits:
        stage_name = f'{split_name}_embeddings'
        embed_file = checkpoint_dir / f'{stage_name}_embeddings.npz'
        
        if not embed_file.exists():
            print(f"\n❌ {split_name}: File not found")
            all_good = False
            continue
        
        data = np.load(embed_file, allow_pickle=True)
        
        has_embeddings = 'embeddings' in data
        has_labels = 'labels' in data
        
        print(f"\n✓ {split_name}:")
        print(f"  - Embeddings: {'✓ Yes' if has_embeddings else '❌ No'}")
        print(f"  - Labels: {'✓ Yes' if has_labels else '❌ No'}")
        
        if has_embeddings and has_labels:
            embeddings = data['embeddings']
            labels = data['labels']
            print(f"  - Embeddings shape: {embeddings.shape}")
            print(f"  - Labels shape: {labels.shape}")
            print(f"  - Samples match: {'✓ Yes' if len(embeddings) == len(labels) else '❌ No'}")
            
            if len(embeddings) != len(labels):
                all_good = False
        else:
            all_good = False
    
    if all_good:
        print(f"\n✅ All checkpoints verified successfully!")
    else:
        print(f"\n⚠️  Some checkpoints have issues!")
    
    return all_good


def main():
    parser = argparse.ArgumentParser(
        description='Add labels to checkpoint files that only contain embeddings'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        required=True,
        help='Directory containing checkpoint files'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='auphong2707/game-reviews-sentiment',
        help='HuggingFace dataset name'
    )
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['train', 'validation', 'test'],
        help='Splits to process (default: train validation test)'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify checkpoints, do not add labels'
    )
    parser.add_argument(
        '--no-replace',
        action='store_true',
        help='Do not replace original files, keep _with_labels suffix'
    )
    
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    print(f"\n{'='*70}")
    print(f"Add Labels to Checkpoint")
    print(f"{'='*70}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Dataset: {args.dataset}")
    print(f"Splits: {', '.join(args.splits)}")
    
    # Verify only mode
    if args.verify_only:
        verify_checkpoints(checkpoint_dir)
        return
    
    # Process each split
    success_count = 0
    for split_name in args.splits:
        try:
            if add_labels_to_checkpoint(checkpoint_dir, args.dataset, split_name):
                success_count += 1
        except Exception as e:
            print(f"\n❌ Error processing {split_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"Summary: {success_count}/{len(args.splits)} splits processed successfully")
    print(f"{'='*70}")
    
    # Replace original files unless --no-replace
    if success_count > 0 and not args.no_replace:
        response = input("\n❓ Replace original checkpoint files? (yes/no): ")
        if response.lower() == 'yes':
            replace_checkpoint_files(checkpoint_dir, dry_run=False)
        else:
            print(f"\n📁 New files saved with '_with_labels' suffix")
            print(f"   You can manually replace them later if needed")
    
    # Final verification
    print(f"\n{'='*70}")
    verify_checkpoints(checkpoint_dir)


if __name__ == '__main__':
    main()
