# Quick Start Guide

## Step-by-Step Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download and Preprocess Data

```bash
python main.py --preprocess
```

This will:
- Automatically download the Duolingo dataset from Kaggle
- Process and clean the data
- Create features
- Split into train/val/test
- Save to `data/processed.pkl`

**Note:** You need `kagglehub` installed. If you get an error, install it:
```bash
pip install kagglehub
```

### 3. Train Models

**Train all models:**
```bash
python main.py --model all --episodes 50
```

**Train individual models:**
```bash
# SM-2 Baseline (fast)
python main.py --model sm2 --episodes 20

# Logistic Regression (fast)
python main.py --model logistic --episodes 30

# Bandit RL (medium)
python main.py --model bandit --episodes 50

# DQN RL (slow, but most powerful)
python main.py --model dqn --episodes 100 --use_gpu
```

### 4. View Results

After training, check:
- `results/model_comparison.csv` - Comparison table
- `results/training_curves_*.png` - Training progress plots
- `results/comparison_*.png` - Bar chart comparisons

## Example Workflow

```bash
# 1. Preprocess (one time)
python main.py --preprocess

# 2. Quick test with all models (few episodes)
python main.py --model all --episodes 10

# 3. Full training
python main.py --model all --episodes 50 --save_models

# 4. Compare specific models
python main.py --model sm2 --episodes 30
python main.py --model dqn --episodes 100 --use_gpu
```

## Troubleshooting

**"Processed data not found"**
- Run `python main.py --preprocess` first

**"kagglehub not found"**
- Install: `pip install kagglehub`

**Out of memory (DQN)**
- Reduce batch size: `--batch_size 16`
- Use CPU: remove `--use_gpu`
- Reduce episodes: `--episodes 20`

**Slow training**
- Start with fewer episodes: `--episodes 10`
- Train models individually instead of `--model all`

## Expected Output

When training completes, you should see:
```
MODEL COMPARISON
============================================================

SM-2:
  Recall Rate: 0.XXXX
  Intervention Efficiency: X.XXXX
  Cumulative Reward: XX.XX
  Episodes: XX

Logistic Regression:
  Recall Rate: 0.XXXX
  ...

Comparison table saved to results/model_comparison.csv
Training curve saved to results/training_curves_recall_rate.png
...
```

## Next Steps

1. Experiment with hyperparameters (see README.md)
2. Try different time budgets: `--time_budget 200.0`
3. Adjust forgetting rate: `--forgetting_rate 0.05`
4. Compare models on test set (modify code to use test data)

