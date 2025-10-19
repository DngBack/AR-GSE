"""
Demo: Reweighting Effect on CIFAR-100-LT
Minh họa sự khác biệt giữa standard accuracy và reweighted accuracy
"""

import torch
import json
import numpy as np
from pathlib import Path

def demo_reweighting():
    """Demo reweighting với ví dụ cụ thể."""
    
    print("=" * 80)
    print("DEMO: REWEIGHTING ON CIFAR-100-LT (IF=100)")
    print("=" * 80)
    
    # Load class weights from training distribution
    splits_dir = Path("./data/cifar100_lt_if100_splits_fixed")
    weights_path = splits_dir / "class_weights.json"
    
    print(f"\n1. Loading class weights from: {weights_path}")
    with open(weights_path, 'r') as f:
        weights_data = json.load(f)
    
    if isinstance(weights_data, list):
        class_weights = torch.tensor(weights_data, dtype=torch.float32)
    else:
        class_weights = torch.tensor([weights_data[str(i)] for i in range(100)])
    
    class_weights = class_weights / class_weights.sum()
    
    # Show distribution
    print(f"\nClass Weight Distribution:")
    print(f"  Head class 0:  {class_weights[0]:.6f} (most frequent)")
    print(f"  Mid class 50:  {class_weights[50]:.6f}")
    print(f"  Tail class 99: {class_weights[99]:.6f} (least frequent)")
    print(f"  Ratio (head/tail): {class_weights[0] / class_weights[99]:.2f}x")
    
    # Load training class counts to show actual numbers
    train_counts_path = splits_dir / "train_class_counts.json"
    with open(train_counts_path, 'r') as f:
        train_counts = json.load(f)
    
    print(f"\nTraining Set Distribution:")
    print(f"  Class 0:  {train_counts['0']} samples (head)")
    print(f"  Class 50: {train_counts['50']} samples (mid)")
    print(f"  Class 99: {train_counts['99']} samples (tail)")
    print(f"  Total: {sum(int(v) for v in train_counts.values())} samples")
    
    # Simulate validation predictions on balanced val set
    print("\n" + "=" * 80)
    print("2. SIMULATION: Model Predictions on Balanced Val Set (10 samples/class)")
    print("=" * 80)
    
    # Simulate a model with head bias
    # Head classes: 90% accuracy
    # Mid classes: 60% accuracy  
    # Tail classes: 30% accuracy
    
    num_classes = 100
    samples_per_class = 10  # balanced val
    
    # Create synthetic predictions
    np.random.seed(42)
    all_targets = []
    all_preds = []
    
    for c in range(num_classes):
        # Determine accuracy based on class position
        if c < 20:  # Head classes (0-19)
            acc = 0.90
        elif c < 80:  # Mid classes (20-79)
            acc = 0.60
        else:  # Tail classes (80-99)
            acc = 0.30
        
        # Generate predictions
        targets = np.full(samples_per_class, c)
        num_correct = int(acc * samples_per_class)
        preds = np.full(samples_per_class, c)
        
        # Make some predictions wrong
        if num_correct < samples_per_class:
            wrong_indices = np.random.choice(samples_per_class, 
                                            samples_per_class - num_correct, 
                                            replace=False)
            for idx in wrong_indices:
                # Predict a random wrong class
                wrong_class = np.random.choice([i for i in range(num_classes) if i != c])
                preds[idx] = wrong_class
        
        all_targets.append(targets)
        all_preds.append(preds)
    
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    
    print(f"\nModel Performance (simulated with head bias):")
    print(f"  Head classes (0-19):   90% accuracy")
    print(f"  Mid classes (20-79):   60% accuracy")
    print(f"  Tail classes (80-99):  30% accuracy")
    
    # Method 1: Standard Accuracy
    print("\n" + "=" * 80)
    print("3. METHOD 1: STANDARD ACCURACY (Each class weighted equally)")
    print("=" * 80)
    
    standard_acc = (all_preds == all_targets).mean()
    print(f"\nCalculation:")
    print(f"  Total correct: {(all_preds == all_targets).sum()}")
    print(f"  Total samples: {len(all_targets)}")
    print(f"  Standard Accuracy = {(all_preds == all_targets).sum()} / {len(all_targets)}")
    print(f"                    = {standard_acc:.4f} ({standard_acc*100:.2f}%)")
    
    # Breakdown by group
    head_mask = all_targets < 20
    mid_mask = (all_targets >= 20) & (all_targets < 80)
    tail_mask = all_targets >= 80
    
    head_acc = (all_preds[head_mask] == all_targets[head_mask]).mean()
    mid_acc = (all_preds[mid_mask] == all_targets[mid_mask]).mean()
    tail_acc = (all_preds[tail_mask] == all_targets[tail_mask]).mean()
    
    print(f"\nBreakdown by group:")
    print(f"  Head (0-19):   {head_acc:.4f} ({head_acc*100:.2f}%)")
    print(f"  Mid (20-79):   {mid_acc:.4f} ({mid_acc*100:.2f}%)")
    print(f"  Tail (80-99):  {tail_acc:.4f} ({tail_acc*100:.2f}%)")
    print(f"  Average:       {(head_acc + mid_acc + tail_acc)/3:.4f}")
    
    # Method 2: Reweighted Accuracy
    print("\n" + "=" * 80)
    print("4. METHOD 2: REWEIGHTED ACCURACY (Classes weighted by training frequency)")
    print("=" * 80)
    
    # Compute per-class accuracy
    class_acc = np.zeros(num_classes)
    for c in range(num_classes):
        mask = all_targets == c
        if mask.sum() > 0:
            class_acc[c] = (all_preds[mask] == c).mean()
    
    # Reweight by training distribution
    class_weights_np = class_weights.numpy()
    reweighted_acc = (class_acc * class_weights_np).sum()
    
    print(f"\nCalculation:")
    print(f"  Step 1: Compute per-class accuracy")
    print(f"    Class 0:  {class_acc[0]:.4f}")
    print(f"    Class 50: {class_acc[50]:.4f}")
    print(f"    Class 99: {class_acc[99]:.4f}")
    
    print(f"\n  Step 2: Weight by training distribution")
    print(f"    Class 0:  {class_acc[0]:.4f} × {class_weights_np[0]:.6f} = {class_acc[0]*class_weights_np[0]:.6f}")
    print(f"    Class 50: {class_acc[50]:.4f} × {class_weights_np[50]:.6f} = {class_acc[50]*class_weights_np[50]:.6f}")
    print(f"    Class 99: {class_acc[99]:.4f} × {class_weights_np[99]:.6f} = {class_acc[99]*class_weights_np[99]:.6f}")
    
    print(f"\n  Step 3: Sum weighted accuracies")
    print(f"    Reweighted Accuracy = Σ(class_acc[c] × class_weight[c])")
    print(f"                        = {reweighted_acc:.4f} ({reweighted_acc*100:.2f}%)")
    
    # Group-wise contribution
    head_weights_sum = class_weights_np[:20].sum()
    mid_weights_sum = class_weights_np[20:80].sum()
    tail_weights_sum = class_weights_np[80:].sum()
    
    head_contrib = (class_acc[:20] * class_weights_np[:20]).sum()
    mid_contrib = (class_acc[20:80] * class_weights_np[20:80]).sum()
    tail_contrib = (class_acc[80:] * class_weights_np[80:]).sum()
    
    print(f"\nContribution by group:")
    print(f"  Head (0-19):   {head_contrib:.6f} (weight sum: {head_weights_sum:.4f}, {head_weights_sum*100:.1f}%)")
    print(f"  Mid (20-79):   {mid_contrib:.6f} (weight sum: {mid_weights_sum:.4f}, {mid_weights_sum*100:.1f}%)")
    print(f"  Tail (80-99):  {tail_contrib:.6f} (weight sum: {tail_weights_sum:.4f}, {tail_weights_sum*100:.1f}%)")
    print(f"  Total:         {head_contrib + mid_contrib + tail_contrib:.6f}")
    
    # Comparison
    print("\n" + "=" * 80)
    print("5. COMPARISON & INTERPRETATION")
    print("=" * 80)
    
    print(f"\nStandard Accuracy:    {standard_acc*100:.2f}%")
    print(f"Reweighted Accuracy:  {reweighted_acc*100:.2f}%")
    print(f"Difference:           {abs(reweighted_acc - standard_acc)*100:.2f}%")
    
    print(f"\n✅ Reweighted accuracy is HIGHER because:")
    print(f"   - Model performs well on head classes (90% acc)")
    print(f"   - Head classes have high weights ({head_weights_sum*100:.1f}% of total)")
    print(f"   - Head contribution dominates: {head_contrib/reweighted_acc*100:.1f}%")
    
    print(f"\n✅ This reflects real-world performance:")
    print(f"   - In actual test set (long-tail), head classes appear more often")
    print(f"   - Getting head classes right matters more for overall accuracy")
    print(f"   - Reweighted accuracy predicts performance on long-tail test data")
    
    print(f"\n❌ Standard accuracy is misleading:")
    print(f"   - Treats all classes equally (unrealistic for long-tail)")
    print(f"   - Overweights tail classes (they're rare in reality)")
    print(f"   - Doesn't match actual test performance")
    
    # What if model was tail-biased instead?
    print("\n" + "=" * 80)
    print("6. COUNTER-EXAMPLE: Tail-biased Model")
    print("=" * 80)
    
    # Simulate tail-biased model
    all_preds_tail = []
    for c in range(num_classes):
        if c < 20:  # Head
            acc = 0.30
        elif c < 80:  # Mid
            acc = 0.60
        else:  # Tail
            acc = 0.90
        
        targets = np.full(samples_per_class, c)
        num_correct = int(acc * samples_per_class)
        preds = np.full(samples_per_class, c)
        
        if num_correct < samples_per_class:
            wrong_indices = np.random.choice(samples_per_class, 
                                            samples_per_class - num_correct, 
                                            replace=False)
            for idx in wrong_indices:
                wrong_class = np.random.choice([i for i in range(num_classes) if i != c])
                preds[idx] = wrong_class
        
        all_preds_tail.append(preds)
    
    all_preds_tail = np.concatenate(all_preds_tail)
    
    standard_acc_tail = (all_preds_tail == all_targets).mean()
    
    class_acc_tail = np.zeros(num_classes)
    for c in range(num_classes):
        mask = all_targets == c
        if mask.sum() > 0:
            class_acc_tail[c] = (all_preds_tail[mask] == c).mean()
    
    reweighted_acc_tail = (class_acc_tail * class_weights_np).sum()
    
    print(f"\nTail-biased Model Performance:")
    print(f"  Head classes (0-19):   30% accuracy")
    print(f"  Mid classes (20-79):   60% accuracy")
    print(f"  Tail classes (80-99):  90% accuracy")
    
    print(f"\nStandard Accuracy:    {standard_acc_tail*100:.2f}%")
    print(f"Reweighted Accuracy:  {reweighted_acc_tail*100:.2f}%")
    print(f"Difference:           {abs(reweighted_acc_tail - standard_acc_tail)*100:.2f}%")
    
    print(f"\n❌ Reweighted accuracy is LOWER because:")
    print(f"   - Model performs poorly on head classes (30% acc)")
    print(f"   - Head classes dominate the test set ({head_weights_sum*100:.1f}%)")
    print(f"   - Real-world performance would be poor!")
    
    print(f"\n✅ Standard accuracy is misleading (again):")
    print(f"   - Shows {standard_acc_tail*100:.2f}% (seems okay)")
    print(f"   - But actual performance on long-tail test would be {reweighted_acc_tail*100:.2f}%!")
    
    print("\n" + "=" * 80)
    print("7. KEY TAKEAWAYS")
    print("=" * 80)
    
    print(f"""
1. **Balanced val set** (10 samples/class) does NOT represent long-tail distribution
   
2. **Standard accuracy** on balanced val is misleading:
   - Treats rare classes (5 train samples) same as common classes (500 samples)
   - Does NOT predict actual test performance
   
3. **Reweighted accuracy** simulates long-tail performance:
   - Weights classes by their training frequency
   - Predicts performance on actual long-tail test set
   - Consistent with training objective
   
4. **In practice**:
   - Train on long-tail expert/gating splits
   - Evaluate on clean balanced val splits (no duplication)
   - Apply reweighting to get realistic metrics
   - Best of both worlds: clean data + accurate metrics

5. **Example from this demo**:
   Head-biased model:
   - Standard:   {standard_acc*100:.2f}% (misleading)
   - Reweighted: {reweighted_acc*100:.2f}% (accurate) ✅
   
   Tail-biased model:
   - Standard:   {standard_acc_tail*100:.2f}% (misleading)
   - Reweighted: {reweighted_acc_tail*100:.2f}% (accurate) ✅

   Reweighting reveals which model is actually better for long-tail!
""")
    
    print("=" * 80)
    print("Demo completed! See docs/reweighting_explained.md for more details.")
    print("=" * 80)

if __name__ == "__main__":
    demo_reweighting()
