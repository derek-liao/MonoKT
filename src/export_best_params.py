import argparse
import wandb
import yaml
import os
from datetime import datetime

def export_best_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g. routerkt)")
    parser.add_argument("--sweep_id", type=str, required=True, help="Sweep ID to export from")
    parser.add_argument("--metric", type=str, default="test_auc", help="Metric to optimize for")
    parser.add_argument("--output_dir", type=str, default="configs/best_params", help="Directory to save best parameters")
    args = parser.parse_args()

    # Initialize wandb API
    api = wandb.Api()
    
    # Get the sweep
    sweep = api.sweep(f"kt-{args.model}/{args.sweep_id}")
    
    # Get all runs in the sweep
    runs = api.runs(f"kt-{args.model}", 
                   {"sweep": sweep.id},
                   order=f"-{args.metric}")
    
    # Get the best run
    best_run = runs[0]
    
    # Get the best parameters
    best_params = {
        "model_name": args.model,
        "best_metric": {
            args.metric: best_run.summary.get(args.metric)
        },
        "parameters": {}
    }
    
    # Add all config parameters
    for key, value in best_run.config.items():
        if key not in ["_wandb", "wandb_version"]:
            best_params["parameters"][key] = value
            
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"best_{args.model}_{timestamp}.yaml")
    
    # Save to yaml file
    with open(output_file, "w") as f:
        yaml.dump(best_params, f, default_flow_style=False)
        
    print(f"\nBest parameters exported to: {output_file}")
    print(f"\nBest {args.metric}: {best_run.summary.get(args.metric)}")
    print("\nBest parameters:")
    for key, value in best_params["parameters"].items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    export_best_params() 