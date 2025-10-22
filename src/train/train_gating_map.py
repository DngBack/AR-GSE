"""
Training Script cho Gating Network với MAP
============================================

Triển khai đầy đủ theo pipeline:
1. Load expert logits đã calibrated
2. Huấn luyện gating với Mixture NLL + Load-balancing
3. Validation và model selection
4. Export gating weights và mixture posteriors

Usage:
    python train_gating_map.py --routing dense
    python train_gating_map.py --routing top_k --top_k 2
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple

from src.models.gating_network_map import GatingNetwork, compute_uncertainty_for_map
from src.models.gating_losses import GatingLoss, compute_gating_metrics


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'dataset': {
        'name': 'cifar100_lt_if100',
        'splits_dir': './data/cifar100_lt_if100_splits_fixed',
        'num_classes': 100,
        'num_groups': 2,
    },
    'experts': {
        'names': ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline'],
        'logits_dir': './outputs/logits/cifar100_lt_if100/',
    },
    'gating': {
        # Architecture
        'hidden_dims': [256, 128],
        'dropout': 0.1,
        'activation': 'relu',
        
        # Routing
        'routing': 'dense',  # 'dense' or 'top_k'
        'top_k': 2,
        'noise_std': 1.0,
        
        # Training
        'epochs': 100,
        'batch_size': 128,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'warmup_epochs': 5,
        
        # Loss weights
        'lambda_lb': 1e-2,     # load-balancing
        'lambda_h': 0.01,      # entropy regularization
        'use_load_balancing': True,
        'use_entropy_reg': True,
        
        # Long-tail handling
        'use_class_weights': True,  # reweight loss theo tần suất
        
        # Validation
        'val_interval': 5,
    },
    'output': {
        'checkpoints_dir': './checkpoints/gating_map/',
        'results_dir': './results/gating_map/',
    },
    'seed': 42
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# DATA LOADING
# ============================================================================

def load_expert_logits(
    expert_names: List[str],
    logits_dir: str,
    split_name: str,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Load logits từ tất cả experts cho một split.
    
    Args:
        expert_names: danh sách tên experts
        logits_dir: thư mục chứa logits
        split_name: 'gating', 'val', 'test', etc.
        device: device để load
    
    Returns:
        logits: [N, E, C] tensor
    """
    logits_list = []
    
    for expert_name in expert_names:
        logits_path = Path(logits_dir) / expert_name / f"{split_name}_logits.pt"
        
        if not logits_path.exists():
            raise FileNotFoundError(f"Logits not found: {logits_path}")
        
        # Load logits (có thể là float16, convert về float32)
        logits_e = torch.load(logits_path, map_location=device).float()
        logits_list.append(logits_e)
    
    # Stack: [E, N, C] → transpose → [N, E, C]
    logits = torch.stack(logits_list, dim=0).transpose(0, 1)
    
    return logits


def load_labels(splits_dir: str, split_name: str, device: str = 'cpu') -> torch.Tensor:
    """
    Load labels cho một split.
    
    Args:
        splits_dir: thư mục chứa split files
        split_name: 'gating', 'val', 'test', etc.
        device: device để load
    
    Returns:
        labels: [N] tensor
    """
    import torchvision
    
    # Load indices
    indices_file = f"{split_name}_indices.json"
    indices_path = Path(splits_dir) / indices_file
    
    if not indices_path.exists():
        raise FileNotFoundError(f"Indices not found: {indices_path}")
    
    with open(indices_path, 'r') as f:
        indices = json.load(f)
    
    # Xác định dataset gốc (train hay test)
    # gating/expert/train → CIFAR-100 train
    # val/test/tunev → CIFAR-100 test
    if split_name in ['gating', 'expert', 'train']:
        cifar_train = True
    else:
        cifar_train = False
    
    # Load CIFAR-100
    dataset = torchvision.datasets.CIFAR100(
        root='./data',
        train=cifar_train,
        download=False
    )
    
    # Extract labels
    labels = torch.tensor([dataset.targets[i] for i in indices], device=device)
    
    return labels


def load_class_weights(splits_dir: str, device: str = 'cpu') -> torch.Tensor:
    """
    Load class weights (frequency-based) cho reweighting.
    
    Returns:
        weights: [C] tensor (normalized to sum=C)
    """
    weights_path = Path(splits_dir) / 'class_weights.json'
    
    if not weights_path.exists():
        print("⚠️  class_weights.json not found, using uniform weights")
        return torch.ones(100, device=device)
    
    with open(weights_path, 'r') as f:
        weights_data = json.load(f)
    
    # Convert to tensor
    if isinstance(weights_data, list):
        weights = torch.tensor(weights_data, device=device)
    elif isinstance(weights_data, dict):
        weights = torch.tensor([weights_data[str(i)] for i in range(100)], device=device)
    else:
        raise ValueError(f"Unexpected format: {type(weights_data)}")
    
    return weights


def create_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Tạo train và validation dataloaders.
    
    Returns:
        train_loader: DataLoader cho gating split
        val_loader: DataLoader cho val split
    """
    print("Loading expert logits and labels...")
    
    expert_names = config['experts']['names']
    logits_dir = config['experts']['logits_dir']
    splits_dir = config['dataset']['splits_dir']
    
    # Train: sử dụng 'gating' split (10% of train, cùng long-tail distribution với expert)
    # Lý do: Experts đã train trên 'expert' split (90%), 
    # nên cần split RIÊNG (10% còn lại) để train gating tránh overfitting
    print("  Loading gating split (10% of train with same long-tail)...")
    train_logits = load_expert_logits(expert_names, logits_dir, 'gating', DEVICE)
    train_labels = load_labels(splits_dir, 'gating', DEVICE)
    
    print(f"    Train: {train_logits.shape[0]:,} samples, "
          f"{train_logits.shape[1]} experts, {train_logits.shape[2]} classes")
    
    # Validation: sử dụng 'val' split (balanced)
    print("  Loading val split...")
    val_logits = load_expert_logits(expert_names, logits_dir, 'val', DEVICE)
    val_labels = load_labels(splits_dir, 'val', DEVICE)
    
    print(f"    Val: {val_logits.shape[0]:,} samples")
    
    # Create datasets
    train_dataset = TensorDataset(train_logits, train_labels)
    val_dataset = TensorDataset(val_logits, val_labels)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['gating']['batch_size'],
        shuffle=True,
        num_workers=0,  # already on GPU
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['gating']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    return train_loader, val_loader


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_one_epoch(
    model: GatingNetwork,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: GatingLoss,
    sample_weights: torch.Tensor = None,
    grad_clip: float = 1.0
) -> Dict[str, float]:
    """
    Train one epoch.
    
    Returns:
        metrics: dict với các metrics trung bình
    """
    model.train()
    
    total_loss = 0.0
    loss_components = {'nll': 0.0, 'load_balancing': 0.0, 'entropy': 0.0}
    
    all_weights = []
    all_posteriors = []
    all_targets = []
    
    for batch_idx, (logits, targets) in enumerate(train_loader):
        # logits: [B, E, C], targets: [B]
        logits = logits.to(DEVICE)
        targets = targets.to(DEVICE)
        
        # Convert logits to posteriors
        posteriors = torch.softmax(logits, dim=-1)  # [B, E, C]
        
        # Forward
        weights, aux = model(posteriors)  # [B, E]
        
        # Compute loss
        batch_sample_weights = None
        if sample_weights is not None:
            batch_sample_weights = sample_weights[targets]
        
        loss, components = loss_fn(
            posteriors, weights, targets,
            sample_weights=batch_sample_weights,
            return_components=True
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        for k, v in components.items():
            if k in loss_components:
                loss_components[k] += v
        
        # Collect for epoch-level metrics
        all_weights.append(weights.detach())
        all_posteriors.append(posteriors.detach())
        all_targets.append(targets.detach())
    
    # Compute epoch metrics
    num_batches = len(train_loader)
    metrics = {
        'loss': total_loss / num_batches,
        'nll': loss_components['nll'] / num_batches,
    }
    
    if loss_components['load_balancing'] > 0:
        metrics['load_balancing'] = loss_components['load_balancing'] / num_batches
    if loss_components['entropy'] > 0:
        metrics['entropy'] = loss_components['entropy'] / num_batches
    
    # Gating-specific metrics
    all_weights = torch.cat(all_weights, dim=0)
    all_posteriors = torch.cat(all_posteriors, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    gating_metrics = compute_gating_metrics(all_weights, all_posteriors, all_targets)
    metrics.update(gating_metrics)
    
    return metrics


@torch.no_grad()
def validate(
    model: GatingNetwork,
    val_loader: DataLoader,
    loss_fn: GatingLoss,
    class_weights: torch.Tensor = None
) -> Dict[str, float]:
    """
    Validate model.
    
    Returns:
        metrics: dict với các metrics
    """
    model.eval()
    
    total_loss = 0.0
    total_nll = 0.0
    all_weights = []
    all_posteriors = []
    all_targets = []
    
    for logits, targets in val_loader:
        logits = logits.to(DEVICE)
        targets = targets.to(DEVICE)
        
        posteriors = torch.softmax(logits, dim=-1)
        weights, _ = model(posteriors)
        
        # Loss (với components để debug)
        loss, components = loss_fn(posteriors, weights, targets, return_components=True)
        total_loss += loss.item()
        total_nll += components['nll']
        
        # Collect
        all_weights.append(weights)
        all_posteriors.append(posteriors)
        all_targets.append(targets)
    
    # Aggregate
    all_weights = torch.cat(all_weights, dim=0)
    all_posteriors = torch.cat(all_posteriors, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Metrics
    metrics = {
        'loss': total_loss / len(val_loader),
        'nll': total_nll / len(val_loader),
    }
    
    gating_metrics = compute_gating_metrics(all_weights, all_posteriors, all_targets)
    metrics.update(gating_metrics)
    
    # Group-wise accuracy
    metrics.update(compute_group_accuracies(all_posteriors, all_weights, all_targets))
    
    return metrics


def compute_group_accuracies(
    posteriors: torch.Tensor,
    weights: torch.Tensor,
    targets: torch.Tensor,
    num_groups: int = 2
) -> Dict[str, float]:
    """
    Compute group-wise accuracies (head/tail).
    
    Args:
        posteriors: [N, E, C]
        weights: [N, E]
        targets: [N]
        num_groups: 2 (head/tail)
    
    Returns:
        metrics: {'head_acc', 'tail_acc', 'balanced_acc'}
    """
    # Mixture predictions
    mixture_posterior = torch.sum(weights.unsqueeze(-1) * posteriors, dim=1)
    predictions = mixture_posterior.argmax(dim=-1)
    
    # Define groups (CIFAR-100-LT: head=0-49, tail=50-99)
    head_mask = targets < 50
    tail_mask = targets >= 50
    
    # Accuracies
    head_acc = (predictions[head_mask] == targets[head_mask]).float().mean().item()
    tail_acc = (predictions[tail_mask] == targets[tail_mask]).float().mean().item()
    balanced_acc = (head_acc + tail_acc) / 2
    
    return {
        'head_acc': head_acc,
        'tail_acc': tail_acc,
        'balanced_acc': balanced_acc
    }


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train_gating(config: Dict):
    """Main training function."""
    
    print("="*70)
    print("🚀 TRAINING GATING NETWORK FOR MAP")
    print("="*70)
    
    # Setup
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Data
    train_loader, val_loader = create_dataloaders(config)
    
    # Class weights (for loss reweighting)
    class_weights = None
    if config['gating']['use_class_weights']:
        class_weights = load_class_weights(config['dataset']['splits_dir'], DEVICE)
        print(f"✅ Loaded class weights (range: [{class_weights.min():.4f}, {class_weights.max():.4f}])")
    
    # Model
    num_experts = len(config['experts']['names'])
    num_classes = config['dataset']['num_classes']
    
    print(f"\n📦 Creating GatingNetwork:")
    print(f"   Experts: {num_experts}")
    print(f"   Classes: {num_classes}")
    print(f"   Routing: {config['gating']['routing']}")
    print(f"   Hidden: {config['gating']['hidden_dims']}")
    
    model = GatingNetwork(
        num_experts=num_experts,
        num_classes=num_classes,
        hidden_dims=config['gating']['hidden_dims'],
        dropout=config['gating']['dropout'],
        routing=config['gating']['routing'],
        top_k=config['gating']['top_k'],
        noise_std=config['gating']['noise_std'],
        activation=config['gating']['activation']
    ).to(DEVICE)
    
    print(f"   Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Feature dim: {model.feature_extractor.feature_dim}")
    
    # Loss
    use_lb = config['gating']['use_load_balancing'] and config['gating']['routing'] == 'top_k'
    print(f"\n⚙️  Loss Configuration:")
    print(f"   Mixture NLL: ✓")
    print(f"   Load-balancing: {'✓' if use_lb else '✗ (disabled for dense routing)'}")
    print(f"   Entropy reg: {'✓' if config['gating']['use_entropy_reg'] else '✗'}")
    if use_lb:
        print(f"   λ_LB: {config['gating']['lambda_lb']}")
    if config['gating']['use_entropy_reg']:
        print(f"   λ_H: {config['gating']['lambda_h']}")
    
    loss_fn = GatingLoss(
        lambda_lb=config['gating']['lambda_lb'],
        lambda_h=config['gating']['lambda_h'],
        use_load_balancing=use_lb,  # Chỉ dùng LB cho sparse routing
        use_entropy_reg=config['gating']['use_entropy_reg'],
        top_k=config['gating']['top_k'] if config['gating']['routing'] == 'top_k' else 1,
        num_experts=num_experts,
        entropy_mode='maximize'
    )
    
    # Optimizer
    if config['gating']['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['gating']['lr'],
            weight_decay=config['gating']['weight_decay']
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['gating']['lr'],
            momentum=0.9,
            weight_decay=config['gating']['weight_decay']
        )
    
    # Scheduler
    if config['gating']['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['gating']['epochs']
        )
    else:
        scheduler = None
    
    # Warmup
    warmup_epochs = config['gating']['warmup_epochs']
    
    # Training loop
    best_val_loss = float('inf')
    best_balanced_acc = 0.0
    results_history = []
    
    checkpoint_dir = Path(config['output']['checkpoints_dir']) / config['dataset']['name']
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🏋️  Starting training for {config['gating']['epochs']} epochs...")
    print(f"   Batch size: {config['gating']['batch_size']}")
    print(f"   Learning rate: {config['gating']['lr']}")
    print(f"   Warmup epochs: {warmup_epochs}")
    
    for epoch in range(config['gating']['epochs']):
        # Warmup LR
        if epoch < warmup_epochs:
            lr_scale = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = config['gating']['lr'] * lr_scale
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn,
            sample_weights=class_weights,
            grad_clip=1.0
        )
        
        # Scheduler
        if scheduler is not None and epoch >= warmup_epochs:
            scheduler.step()
        
        # Print
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1:3d}/{config['gating']['epochs']}:")
        print(f"  Train Loss: {train_metrics['loss']:.4f} "
              f"(NLL={train_metrics['nll']:.4f}), LR={current_lr:.6f}")
        print(f"  Mixture Acc: {train_metrics['mixture_acc']:.4f}, "
              f"Effective Experts: {train_metrics['effective_experts']:.2f}")
        
        # Validate
        if (epoch + 1) % config['gating']['val_interval'] == 0 or epoch == config['gating']['epochs'] - 1:
            val_metrics = validate(model, val_loader, loss_fn, class_weights)
            
            print(f"  Val Loss: {val_metrics['loss']:.4f} (NLL={val_metrics['nll']:.4f})")
            print(f"  Val Acc: Overall={val_metrics['mixture_acc']:.4f}, "
                  f"Head={val_metrics['head_acc']:.4f}, Tail={val_metrics['tail_acc']:.4f}, "
                  f"Balanced={val_metrics['balanced_acc']:.4f}")
            
            # Save best model
            if val_metrics['balanced_acc'] > best_balanced_acc:
                best_balanced_acc = val_metrics['balanced_acc']
                best_val_loss = val_metrics['loss']
                
                save_path = checkpoint_dir / 'best_gating.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': val_metrics,
                    'config': config
                }, save_path)
                
                print(f"  💾 Saved best model (balanced_acc={best_balanced_acc:.4f})")
            
            # Save history
            results_history.append({
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics
            })
    
    # Save final model
    final_path = checkpoint_dir / 'final_gating.pth'
    torch.save({
        'epoch': config['gating']['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }, final_path)
    
    # Save training history
    results_dir = Path(config['output']['results_dir']) / config['dataset']['name']
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'training_history.json', 'w') as f:
        # Convert tensors to python types
        history_serializable = []
        for item in results_history:
            serializable = {'epoch': item['epoch'], 'train': {}, 'val': {}}
            for split in ['train', 'val']:
                for k, v in item[split].items():
                    serializable[split][k] = float(v) if isinstance(v, (int, float, np.number)) else v
            history_serializable.append(serializable)
        json.dump(history_serializable, f, indent=2)
    
    print(f"\n✅ Training completed!")
    print(f"   Best balanced acc: {best_balanced_acc:.4f}")
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Checkpoints saved to: {checkpoint_dir}")
    
    return model, results_history


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Gating Network for MAP')
    parser.add_argument('--routing', type=str, default='dense', choices=['dense', 'top_k'],
                       help='Routing strategy')
    parser.add_argument('--top_k', type=int, default=2,
                       help='K for top-k routing')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--lambda_lb', type=float, default=1e-2,
                       help='Load-balancing weight')
    parser.add_argument('--lambda_h', type=float, default=0.01,
                       help='Entropy regularization weight')
    
    args = parser.parse_args()
    
    # Update config
    CONFIG['gating']['routing'] = args.routing
    CONFIG['gating']['top_k'] = args.top_k
    CONFIG['gating']['epochs'] = args.epochs
    CONFIG['gating']['batch_size'] = args.batch_size
    CONFIG['gating']['lr'] = args.lr
    CONFIG['gating']['lambda_lb'] = args.lambda_lb
    CONFIG['gating']['lambda_h'] = args.lambda_h
    
    # Train
    model, history = train_gating(CONFIG)
    
    print("\n🎉 Done!")


if __name__ == '__main__':
    main()
