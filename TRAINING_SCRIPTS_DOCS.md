# AR-GSE Training Scripts Documentation

Comprehensive documentation for the refactored AR-GSE training scripts.

## Overview

The AR-GSE training process has been refactored into dedicated Python scripts that replace the original module commands, providing better usability, error handling, and configuration options.

## Scripts Summary

| Script | Purpose | Replaces |
|--------|---------|----------|
| `create_splits.py` | Data preparation | `python -c "from src.data.enhanced_datasets import create_full_cifar100_lt_splits; create_full_cifar100_lt_splits()"` |
| `train_experts.py` | Expert models training | `python -m src.train.train_expert` |
| `train_gating.py` | Gating models training | `python -m src.train.train_gating_only` |
| `train_argse.py` | AR-GSE ensemble training | `python run_improved_eg_outer.py` |
| `evaluate_argse.py` | Model evaluation | `python -m src.train.eval_agse_plugin.py` |
| `run_pipeline.py` | Complete pipeline | Manual step-by-step execution |

## Detailed Usage

### 1. Data Preparation (`create_splits.py`)

Creates CIFAR-100-LT dataset splits with long-tail distribution.

```bash
# Basic usage
python create_splits.py

# With custom parameters (advanced script)
python create_splits_advanced.py --imb-factor 50 --val-ratio 0.25 --verbose
```

**Features:**
- Automatic CIFAR-100 download
- Configurable imbalance factors
- Detailed distribution analysis
- Progress tracking

### 2. Expert Models Training (`train_experts.py`)

Trains the three expert models: CE, LogitAdjust, and BalancedSoftmax.

```bash
# Train all experts
python train_experts.py

# Train specific expert
python train_experts.py --expert ce --verbose

# Dry run (show configuration)
python train_experts.py --dry-run

# Custom device
python train_experts.py --device cuda
```

**Arguments:**
- `--expert {ce|logitadjust|balsoftmax|all}`: Expert type to train
- `--epochs INT`: Override training epochs
- `--lr FLOAT`: Override learning rate
- `--device {cpu|cuda|auto}`: Training device
- `--verbose`: Detailed output
- `--dry-run`: Show config without training

**Output:** Models saved to `checkpoints/experts/cifar100_lt_if100/`

### 3. Gating Models Training (`train_gating.py`)

Trains gating models in two modes: pretrain and selective.

```bash
# Pretrain mode (warmup)
python train_gating.py --mode pretrain

# Selective mode (expert selection)
python train_gating.py --mode selective

# With verbose output
python train_gating.py --mode pretrain --verbose
```

**Arguments:**
- `--mode {pretrain|selective}`: Training mode (required)
- `--device {cpu|cuda|auto}`: Training device
- `--verbose`: Detailed output
- `--dry-run`: Show config without training

**Output:** Models saved to `checkpoints/gating_pretrained/cifar100_lt_if100/`

### 4. AR-GSE Ensemble Training (`train_argse.py`)

Trains the complete AR-GSE ensemble model.

```bash
# Basic training
python train_argse.py

# With verbose output and prerequisites check
python train_argse.py --verbose

# Dry run to check prerequisites
python train_argse.py --dry-run
```

**Arguments:**
- `--device {cpu|cuda|auto}`: Training device
- `--verbose`: Detailed output
- `--dry-run`: Check prerequisites without training
- `--resume`: Resume from checkpoint

**Prerequisites Checked:**
- Expert models exist
- Gating models exist
- Data splits are prepared

**Output:** Models saved to `checkpoints/argse_worst_eg_improved_v3_3/cifar100_lt_if100/`

### 5. Model Evaluation (`evaluate_argse.py`)

Evaluates the trained AR-GSE model.

```bash
# Basic evaluation
python evaluate_argse.py

# Evaluate on specific dataset
python evaluate_argse.py --dataset val --verbose

# Save detailed results
python evaluate_argse.py --save-results
```

**Arguments:**
- `--dataset {test|val|tunev}`: Dataset split for evaluation
- `--checkpoint PATH`: Specific checkpoint to evaluate
- `--device {cpu|cuda|auto}`: Evaluation device
- `--verbose`: Detailed output
- `--save-results`: Save detailed results

**Output:** Results saved to `results_worst_eg_improved/cifar100_lt_if100/`

### 6. Complete Pipeline (`run_pipeline.py`)

Runs the entire training pipeline automatically.

```bash
# Full pipeline
python run_pipeline.py

# Start from specific step
python run_pipeline.py --start-from experts --verbose

# Run partial pipeline
python run_pipeline.py --start-from data --stop-at gating

# Dry run (show what would be executed)
python run_pipeline.py --dry-run
```

**Arguments:**
- `--start-from {data|experts|gating|argse|eval}`: Starting step
- `--stop-at {data|experts|gating|argse|eval}`: Stopping step
- `--device {cpu|cuda|auto}`: Training device
- `--verbose`: Detailed output for all steps
- `--dry-run`: Show pipeline without execution

## Error Handling & Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   Solution: Run from project root directory
   Command: cd /path/to/AR-GSE && python script.py
   ```

2. **Missing Dependencies**
   ```
   Solution: Install requirements
   Command: pip install -r requirements.txt
   ```

3. **CUDA Out of Memory**
   ```
   Solution: Use CPU or reduce batch size
   Command: python script.py --device cpu
   ```

4. **Missing Prerequisites**
   ```
   Each script checks prerequisites and provides specific guidance
   ```

### Verbose Mode Benefits

- Detailed configuration display
- Progress tracking
- Error context and suggestions
- System information (GPU, PyTorch version)
- Full error tracebacks

### Dry Run Mode Benefits

- Check prerequisites without execution
- Validate configurations
- Estimate resource requirements
- Test pipeline steps

## Best Practices

### 1. Progressive Training
```bash
# Step-by-step approach
python create_splits.py
python train_experts.py --verbose
python train_gating.py --mode pretrain --verbose
python train_gating.py --mode selective --verbose
python train_argse.py --verbose
python evaluate_argse.py --verbose
```

### 2. Pipeline Training
```bash
# Automated approach
python run_pipeline.py --verbose

# Or with specific steps
python run_pipeline.py --start-from experts --verbose
```

### 3. Development Workflow
```bash
# Test configurations first
python train_experts.py --dry-run
python run_pipeline.py --dry-run

# Then execute
python run_pipeline.py --verbose
```

## Integration with Original Commands

The scripts maintain compatibility with the original workflow:

| Original Command | New Script | Advantage |
|------------------|------------|-----------|
| `python -c "from src.data..."` | `python create_splits.py` | Better error handling, progress tracking |
| `python -m src.train.train_expert` | `python train_experts.py` | Selective training, dry run, verbose mode |
| `python -m src.train.train_gating_only --mode X` | `python train_gating.py --mode X` | Better UX, error handling |
| `python run_improved_eg_outer.py` | `python train_argse.py` | Prerequisites check, better output |
| `python -m src.train.eval_agse_plugin.py` | `python evaluate_argse.py` | Flexible evaluation, result management |

## Output Structure

```
AR-GSE/
├── data/
│   └── cifar100_lt_if100_splits/     # Dataset splits
├── checkpoints/
│   ├── experts/cifar100_lt_if100/    # Expert models
│   ├── gating_pretrained/cifar100_lt_if100/  # Gating models
│   └── argse_worst_eg_improved_v3_3/cifar100_lt_if100/  # AR-GSE models
└── results_worst_eg_improved/
    └── cifar100_lt_if100/            # Evaluation results
```

## Performance Tips

1. **Use GPU when available**: All scripts auto-detect CUDA
2. **Monitor memory usage**: Use `--verbose` to track resource usage
3. **Save intermediate results**: Each step saves checkpoints
4. **Use pipeline for automation**: `run_pipeline.py` for unattended training
5. **Check prerequisites**: Use `--dry-run` to validate setup