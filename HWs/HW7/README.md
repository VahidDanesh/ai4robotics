# Behavior Cloning Training Scripts

This directory contains scripts to train, evaluate, and select the best behavior cloning model for the Pusher environment.

## Script Overview

1. `train_model.py` - Trains a single behavior cloning model with specified hyperparameters
2. `train_with_parameters.py` - Trains multiple models with different hyperparameter configurations
3. `evaluate_models.py` - Evaluates and compares trained models to select the best one
4. `train_optimal_model.py` - A comprehensive script that trains a model with optimized parameters

## Requirements

Make sure you have the required packages installed:

```bash
pip install mujoco==3.2.3
pip install "minari[hf,hdf5]"
pip install gymnasium[mujoco]
pip install torch numpy
```

## Environment Setup

These scripts are designed to work with both CPU and GPU acceleration. If you have a CUDA-capable GPU, the code will automatically use it for faster training.

For better performance, consider running this on Google Colab with a GPU:
1. Upload the scripts to Colab
2. Select Runtime > Change runtime type > Hardware accelerator > GPU
3. Run the scripts in Colab

## Easy Solution: Use the Optimal Model Script

If you just want to quickly train a good model, use the `train_optimal_model.py` script:

```bash
python HWs/HW7/train_optimal_model.py
```

This script uses optimized default parameters (512 hidden units, 2 layers, 1024 batch size, 50 epochs) that should work well for the Pusher environment. The trained model will be saved to `HWs/HW7/trained_pusher_policy.pt`.

You can customize parameters as needed:

```bash
python HWs/HW7/train_optimal_model.py --hidden_size 512 --num_layers 3 --epochs 100 --visualize
```

## Advanced Usage: Multiple Training Options

### Single Model Training

To train a single model with specific hyperparameters:

```bash
python HWs/HW7/train_model.py --hid_size 512 --num_layers 2 --batch_size 1024 --epochs 50
```

Parameters:
- `--hid_size`: Hidden layer size (default: 256)
- `--num_layers`: Number of hidden layers (default: 2)
- `--batch_size`: Batch size for training (default: 1024)
- `--epochs`: Number of training epochs (default: 50)
- `--model_path`: Path to save the trained model (default: 'trained_pusher_policy.pt')
- `--visualize`: Enable visualization after training (flag)

### Training Multiple Models

To train multiple models with different hyperparameter configurations:

```bash
python HWs/HW7/train_with_parameters.py --output_dir HWs/HW7/models
```

Parameters:
- `--output_dir`: Directory to save trained models (default: 'HWs/HW7/models')
- `--visualize`: Enable visualization for each model (flag)
- `--resume`: Skip configurations that have already been trained (flag)
- `--continue_on_error`: Continue training other models if one fails (flag)

### Hyperparameter Selection Guide

Based on the homework instructions, here are some guidelines for hyperparameter selection:

- **Hidden layers**: The assignment recommends no more than a handful of layers. Good values to try are 2-3.
- **Layer sizes**: Layer sizes between 128-1024 typically work well. The assignment recommends not going beyond a few thousand.
- **Batch size**: For efficiency, use powers of 2 (e.g., 512, 1024, 2048).
- **Epochs**: Typically, 50-100 epochs is sufficient. The assignment suggests that more than a few hundred is probably too many.

## Evaluating Models

To evaluate models and select the best one:

```bash
python HWs/HW7/evaluate_models.py --model_dir HWs/HW7/models --num_trials 5
```

Parameters:
- `--model_dir`: Directory containing trained models (default: 'HWs/HW7/models')
- `--num_trials`: Number of evaluation trials per model (default: 5)
- `--output_path`: Path to save the best model (default: 'HWs/HW7/trained_pusher_policy.pt')

## Complete Workflow Example

```bash
# Create model directory
mkdir -p HWs/HW7/models

# Train multiple models with different hyperparameter configurations
python HWs/HW7/train_with_parameters.py --output_dir HWs/HW7/models

# Evaluate all models and select the best one
python HWs/HW7/evaluate_models.py --model_dir HWs/HW7/models --output_path HWs/HW7/trained_pusher_policy.pt
```

The best model will be copied to `HWs/HW7/trained_pusher_policy.pt` for submission.

## Troubleshooting

If you encounter issues:

1. **Package installation problems**: Make sure you have installed all required packages with the exact versions specified.

2. **GPU memory errors**: If you run out of GPU memory, try reducing the batch size or model size.

3. **Path issues**: Make sure the paths to scripts and model files are correct. Use absolute paths if needed.

4. **Visualization issues**: If visualization fails, try running without the `--visualize` flag.

5. **MuJoCo/Gymnasium errors**: Some environments require special setup. Check the Gymnasium documentation for additional dependencies. 