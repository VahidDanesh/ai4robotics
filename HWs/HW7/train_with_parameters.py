import subprocess
import argparse
import os

# Define hyperparameter configurations to try
configurations = [
    # Configuration format: (hidden_size, num_layers, batch_size, epochs)
    (256, 2, 1024, 100),      # Baseline configuration
    (512, 2, 1024, 100),      # Larger hidden size
    (1024, 4, 2048, 100),
    (1024, 4, 4096, 100),
    (2048, 4, 4096, 100),\
]

def main(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    for i, (hid_size, num_layers, batch_size, epochs) in enumerate(configurations):
        # Skip configurations already tested if resume flag is set
        model_path = os.path.join(args.output_dir, f"model_h{hid_size}_l{num_layers}_b{batch_size}_e{epochs}.pt")
        if args.resume and os.path.exists(model_path):
            print(f"Skipping configuration {i+1}/{len(configurations)} as model already exists at {model_path}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Training configuration {i+1}/{len(configurations)}:")
        print(f"Hidden size: {hid_size}, Layers: {num_layers}, Batch size: {batch_size}, Epochs: {epochs}")
        print(f"{'='*80}\n")
        
        # Construct command
        cmd = [
            "python", "HWs/HW7/train_model.py",
            "--hid_size", str(hid_size),
            "--num_layers", str(num_layers),
            "--batch_size", str(batch_size),
            "--epochs", str(epochs),
            "--model_path", model_path
        ]
        
        if args.visualize:
            cmd.append("--visualize")
        
        # Run the training script
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"Error training configuration {i+1}")
            if not args.continue_on_error:
                break
    
    print("\nTraining completed for all configurations")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train multiple behavior cloning models with different hyperparameters')
    parser.add_argument('--output_dir', type=str, default='HWs/HW7/models', help='Directory to save trained models')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization for each model')
    parser.add_argument('--resume', action='store_true', help='Skip configurations that have already been trained')
    parser.add_argument('--continue_on_error', action='store_true', help='Continue training other models if one fails')
    
    args = parser.parse_args()
    main(args) 