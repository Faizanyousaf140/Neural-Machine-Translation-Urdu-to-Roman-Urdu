import os
import math
import pickle
import random
import json
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from jiwer import cer
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Import enhanced model
from enhanced_model import create_model, create_experiment_models

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

# ================================
# Enhanced Dataset Class
# ================================

class UrduRomanDataset(Dataset):
    def __init__(self, src_seqs, tgt_seqs):
        self.src_seqs = src_seqs
        self.tgt_seqs = tgt_seqs

    def __len__(self):
        return len(self.src_seqs)

    def __getitem__(self, idx):
        return torch.tensor(self.src_seqs[idx]), torch.tensor(self.tgt_seqs[idx])

def collate_fn(batch):
    """Enhanced dynamic padding with better memory management."""
    src_seqs, tgt_seqs = zip(*batch)
    
    # Get lengths before padding
    src_lens = [len(s) for s in src_seqs]
    max_src_len = max(src_lens)
    max_tgt_len = max(len(t) for t in tgt_seqs)
    
    # Pad sequences
    src_padded = torch.zeros(len(src_seqs), max_src_len, dtype=torch.long)
    tgt_padded = torch.zeros(len(tgt_seqs), max_tgt_len, dtype=torch.long)
    
    for i, (src, tgt) in enumerate(zip(src_seqs, tgt_seqs)):
        src_padded[i, :len(src)] = src
        tgt_padded[i, :len(tgt)] = tgt
    
    src_lens = torch.tensor(src_lens)
    return src_padded, src_lens, tgt_padded

# ================================
# Enhanced Metrics
# ================================

def calculate_bleu(pred_list, target_list, idx2char):
    """Calculate average BLEU-4 score with better handling."""
    smooth = SmoothingFunction().method4
    bleu_scores = []
    
    for pred, target in zip(pred_list, target_list):
        # Convert indices to chars (skip special tokens)
        pred_chars = [idx2char.get(idx, '') for idx in pred if idx not in [0,1,2,3]]
        target_chars = [[idx2char.get(idx, '') for idx in target if idx not in [0,1,2,3]]]
        
        if not pred_chars or not target_chars[0]:
            bleu_scores.append(0.0)
            continue
            
        try:
            bleu = sentence_bleu(target_chars, pred_chars, smoothing_function=smooth)
            bleu_scores.append(bleu)
        except:
            bleu_scores.append(0.0)
    
    return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

def calculate_cer(pred_list, target_list, idx2char):
    """Calculate Character Error Rate with better error handling."""
    total_cer = 0.0
    count = 0
    
    for pred, target in zip(pred_list, target_list):
        pred_str = ''.join([idx2char.get(idx, '') for idx in pred if idx not in [0,1,2,3]])
        target_str = ''.join([idx2char.get(idx, '') for idx in target if idx not in [0,1,2,3]])
        
        if target_str:
            try:
                total_cer += cer(target_str, pred_str)
                count += 1
            except:
                count += 1  # Count as error
    
    return total_cer / count if count > 0 else 1.0

def calculate_accuracy(pred_list, target_list, idx2char):
    """Calculate exact match accuracy."""
    correct = 0
    total = 0
    
    for pred, target in zip(pred_list, target_list):
        pred_str = ''.join([idx2char.get(idx, '') for idx in pred if idx not in [0,1,2,3]])
        target_str = ''.join([idx2char.get(idx, '') for idx in target if idx not in [0,1,2,3]])
        
        if pred_str == target_str:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0.0

# ================================
# Enhanced Training & Evaluation
# ================================

def train_epoch(model, dataloader, optimizer, criterion, clip=1.0, teacher_forcing_ratio=0.5):
    """Enhanced training epoch with better monitoring."""
    model.train()
    epoch_loss = 0
    num_batches = 0
    
    for src, src_len, trg in tqdm(dataloader, desc="Training"):
        src, src_len, trg = src.to(device), src_len.to(device), trg.to(device)
        
        optimizer.zero_grad()
        output = model(src, src_len, trg, teacher_forcing_ratio=teacher_forcing_ratio)
        
        # Output: [B, trg_len, output_dim], trg: [B, trg_len]
        output = output[:, 1:].reshape(-1, output.shape[-1])  # skip <sos>
        trg = trg[:, 1:].reshape(-1)  # skip <sos>
        
        loss = criterion(output, trg)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
    
    return epoch_loss / num_batches

def evaluate(model, dataloader, criterion, idx2char, return_predictions=False):
    """Enhanced evaluation with more metrics."""
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_targets = []
    num_batches = 0
    
    with torch.no_grad():
        for src, src_len, trg in tqdm(dataloader, desc="Evaluating"):
            src, src_len, trg = src.to(device), src_len.to(device), trg.to(device)
            output = model(src, src_len, trg, teacher_forcing_ratio=0.0)  # no teacher forcing
            
            # Loss
            output_flat = output[:, 1:].reshape(-1, output.shape[-1])
            trg_flat = trg[:, 1:].reshape(-1)
            loss = criterion(output_flat, trg_flat)
            epoch_loss += loss.item()
            
            # For metrics: get predicted sequences
            preds = output.argmax(dim=-1)  # [B, trg_len]
            for i in range(preds.size(0)):
                all_preds.append(preds[i].cpu().tolist())
                all_targets.append(trg[i].cpu().tolist())
            
            num_batches += 1
    
    # Compute metrics
    bleu = calculate_bleu(all_preds, all_targets, idx2char)
    char_error = calculate_cer(all_preds, all_targets, idx2char)
    accuracy = calculate_accuracy(all_preds, all_targets, idx2char)
    perplexity = math.exp(epoch_loss / num_batches)
    
    results = {
        'loss': epoch_loss / num_batches,
        'bleu': bleu,
        'cer': char_error,
        'accuracy': accuracy,
        'perplexity': perplexity
    }
    
    if return_predictions:
        results['predictions'] = all_preds
        results['targets'] = all_targets
    
    return results

# ================================
# Training Utilities
# ================================

def save_checkpoint(model, optimizer, epoch, loss, metrics, filepath):
    """Save model checkpoint with metadata."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['metrics']

def plot_training_history(history, save_path=None):
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # BLEU
    axes[0, 1].plot(history['val_bleu'], label='Val BLEU')
    axes[0, 1].set_title('Validation BLEU Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('BLEU')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # CER
    axes[1, 0].plot(history['val_cer'], label='Val CER')
    axes[1, 0].set_title('Validation Character Error Rate')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('CER')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Perplexity
    axes[1, 1].plot(history['val_perplexity'], label='Val Perplexity')
    axes[1, 1].set_title('Validation Perplexity')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Perplexity')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# ================================
# Main Training Functions
# ================================

def train_single_experiment(config, experiment_name, data, tgt_idx2char, results_dir="experiments"):
    """Train a single experiment configuration."""
    
    print(f"\n🧪 Starting Experiment: {experiment_name}")
    print("=" * 60)
    print(f"Configuration: {config}")
    
    # Create results directory
    exp_dir = os.path.join(results_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Build model (only pass model parameters)
    model_params = {
        'input_dim': config['input_dim'],
        'output_dim': config['output_dim'],
        'emb_dim': config['emb_dim'],
        'enc_hid_dim': config['enc_hid_dim'],
        'dec_hid_dim': config['dec_hid_dim'],
        'enc_layers': config['enc_layers'],
        'dec_layers': config['dec_layers'],
        'dropout': config['dropout'],
        'attention': config['attention'],
        'device': device
    }
    model = create_model(**model_params).to(device)
    print(f"✅ Model created. Parameters: {model.count_parameters():,}")
    
    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore <pad>
    
    # Learning rate scheduler1
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    
    # DataLoaders
    train_dataset = UrduRomanDataset(*data['train'])
    val_dataset = UrduRomanDataset(*data['val'])
    test_dataset = UrduRomanDataset(*data['test'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [], 'val_bleu': [], 
        'val_cer': [], 'val_accuracy': [], 'val_perplexity': []
    }
    
    # Training loop
    best_valid_loss = float('inf')
    best_bleu = 0.0
    patience = 5
    patience_counter = 0
    
    print(f"\n🚀 Starting training for {config['epochs']} epochs...")
    
    for epoch in range(config['epochs']):
        start_time = time.time()
        
        # Training
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, 
            clip=config['grad_clip'], 
            teacher_forcing_ratio=config.get('teacher_forcing_ratio', 0.5)
        )
        
        # Validation
        val_results = evaluate(model, val_loader, criterion, tgt_idx2char)
        
        # Learning rate scheduling
        scheduler.step(val_results['loss'])
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_results['loss'])
        history['val_bleu'].append(val_results['bleu'])
        history['val_cer'].append(val_results['cer'])
        history['val_accuracy'].append(val_results['accuracy'])
        history['val_perplexity'].append(val_results['perplexity'])
        
        # Print progress
        epoch_time = time.time() - start_time
        print(f'Epoch: {epoch+1:02}/{config["epochs"]} | '
              f'Train Loss: {train_loss:.4f} | Val Loss: {val_results["loss"]:.4f} | '
              f'BLEU: {val_results["bleu"]:.4f} | CER: {val_results["cer"]:.4f} | '
              f'Acc: {val_results["accuracy"]:.4f} | Time: {epoch_time:.1f}s')
        
        # Save best model
        if val_results['loss'] < best_valid_loss:
            best_valid_loss = val_results['loss']
            best_bleu = val_results['bleu']
            patience_counter = 0
            
            # Save best model
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best_model.pt'))
            save_checkpoint(model, optimizer, epoch, val_results['loss'], val_results, 
                          os.path.join(exp_dir, 'best_checkpoint.pt'))
            print("✅ Saved new best model!")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"🛑 Early stopping at epoch {epoch+1}")
            break
    
    # Final test evaluation
    print(f"\n🧪 Final test evaluation...")
    model.load_state_dict(torch.load(os.path.join(exp_dir, 'best_model.pt')))
    test_results = evaluate(model, test_loader, criterion, tgt_idx2char)
    
    # Save results
    results = {
        'config': config,
        'history': history,
        'test_results': test_results,
        'best_val_loss': best_valid_loss,
        'best_val_bleu': best_bleu,
        'total_epochs': len(history['train_loss'])
    }
    
    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot training history
    plot_training_history(history, os.path.join(exp_dir, 'training_history.png'))
    
    print(f"\n✅ Experiment {experiment_name} completed!")
    print(f"📊 Test Results: BLEU={test_results['bleu']:.4f}, CER={test_results['cer']:.4f}, Acc={test_results['accuracy']:.4f}")
    
    return results

def run_hyperparameter_experiments():
    """Run multiple hyperparameter experiments."""
    
    print("🚀 Starting Hyperparameter Experiments")
    print("=" * 60)
    
    # Load data
    with open("processed_data/data.pkl", "rb") as f:
        data = pickle.load(f)
    with open("processed_data/tgt_idx2char.pkl", "rb") as f:
        tgt_idx2char = pickle.load(f)
    
    # Get experiment configurations
    experiments = create_experiment_models(device)
    
    # Add training parameters to each config
    for name, config in experiments.items():
        config.update({
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 20,
            'grad_clip': 1.0,
            'teacher_forcing_ratio': 0.5
        })
    
    # Run experiments
    all_results = {}
    
    for name, config in experiments.items():
        try:
            results = train_single_experiment(config, name, data, tgt_idx2char)
            all_results[name] = results
        except Exception as e:
            print(f"❌ Experiment {name} failed: {e}")
            continue
    
    # Compare results
    print("\n📊 EXPERIMENT COMPARISON:")
    print("=" * 60)
    print(f"{'Experiment':<15} {'BLEU':<8} {'CER':<8} {'Accuracy':<10} {'Parameters':<12}")
    print("-" * 60)
    
    for name, results in all_results.items():
        test = results['test_results']
        params = results['config']
        param_count = sum(p.numel() for p in create_model(**params).parameters() if p.requires_grad)
        print(f"{name:<15} {test['bleu']:<8.4f} {test['cer']:<8.4f} {test['accuracy']:<10.4f} {param_count:<12,}")
    
    # Save comparison
    with open("experiments/comparison.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ All experiments completed! Results saved to experiments/")

def main():
    """Main training function."""
    
    print("🚀 Enhanced Urdu-Roman Translation Training")
    print("=" * 60)
    
    # Load data
    with open("processed_data/data.pkl", "rb") as f:
        data = pickle.load(f)
    with open("processed_data/tgt_idx2char.pkl", "rb") as f:
        tgt_idx2char = pickle.load(f)
    
    print(f"📊 Dataset loaded:")
    print(f"  Train: {len(data['train'][0])} pairs")
    print(f"  Val:   {len(data['val'][0])} pairs")
    print(f"  Test:  {len(data['test'][0])} pairs")
    
    # Choose training mode
    print("\n🎯 Choose training mode:")
    print("1. Single experiment (recommended for testing)")
    print("2. Full hyperparameter experiments")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Single experiment with enhanced model
        config = {
            'input_dim': 60,
            'output_dim': 40,
            'emb_dim': 256,
            'enc_hid_dim': 512,
            'dec_hid_dim': 512,
            'enc_layers': 2,
            'dec_layers': 4,
            'dropout': 0.3,
            'attention': True,
            'device': device,
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 20,
            'grad_clip': 1.0,
            'teacher_forcing_ratio': 0.5
        }
        
        results = train_single_experiment(config, "enhanced_baseline", data, tgt_idx2char)
        
    elif choice == "2":
        # Run all hyperparameter experiments
        run_hyperparameter_experiments()
    
    else:
        print("❌ Invalid choice. Running single experiment...")
        main()

if __name__ == "__main__":
    main()
