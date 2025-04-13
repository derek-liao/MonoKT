import os
import argparse
import wandb
import yaml
from main import main
from utils.config import ConfigNode as CN
from export_best_params import export_best_params

def update_config_with_sweep(config, sweep_config):
    """Update config with wandb sweep parameters."""
    # First ensure model_name is set
    if "model_name" in sweep_config:
        config.model_name = sweep_config["model_name"]
    
    # Then handle other parameters
    for key, value in sweep_config.items():
        if key == "model_name":
            continue  # already handled
        elif key == "data_name":
            config.data_name = value
        elif key in ["num_attn_heads", "num_shared_heads", "num_selected_heads", 
                    "balance_loss_weight", "l2", "routing_mode", "separate_qr"]:
            if not hasattr(config, f"{config.model_name}_config"):
                config[f"{config.model_name}_config"] = CN()
            config[f"{config.model_name}_config"][key] = value
    return config

def train_with_wandb():
    """Train model with wandb sweep configuration."""
    # Initialize wandb
    run = wandb.init()
    
    # Load base config
    base_cfg_file = open("configs/example.yaml", "r")
    base_cfg = yaml.safe_load(base_cfg_file)
    cfg = CN(base_cfg)
    cfg.set_new_allowed(True)
    
    # Update config with sweep parameters
    cfg = update_config_with_sweep(cfg, dict(run.config))
    
    # Train model and get results
    test_auc, test_acc, test_rmse = main(cfg)
    
    # Log metrics
    wandb.log({
        "test_auc": test_auc,
        "test_acc": test_acc,
        "test_rmse": test_rmse
    })

def run_sweep():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name to sweep (e.g. routerkt)")
    parser.add_argument("--sweep_id", type=str, help="Existing sweep ID to continue")
    parser.add_argument("--count", type=int, default=-1, help="Number of runs per dataset")
    parser.add_argument("--data", type=str, nargs="+", help="Datasets to sweep (e.g. spanish statics). If not specified, use all datasets from config")
    parser.add_argument("--metric", type=str, default="test_auc", help="Metric to optimize for")
    parser.add_argument("--output_dir", type=str, default="configs/best_params", help="Directory to save best parameters")
    parser.add_argument("--project", type=str, default="kt-sweep", help="W&B project name")
    args = parser.parse_args()
    
    if args.sweep_id is None:
        # Load sweep config
        sweep_config_path = f"configs/sweep_{args.model}.yaml"
        if not os.path.exists(sweep_config_path):
            raise ValueError(f"Sweep config not found: {sweep_config_path}")
            
        with open(sweep_config_path) as f:
            base_sweep_config = yaml.safe_load(f)
            
        # Ensure model name is set in config
        if "model_name" not in base_sweep_config["parameters"]:
            base_sweep_config["parameters"]["model_name"] = {"value": args.model}
        
        # Get datasets to sweep
        if args.data:
            datasets = args.data
        else:
            datasets = base_sweep_config["parameters"]["data_name"]["values"]
            
        print(f"\nStarting sweeps for datasets: {datasets}")
        
        # Create separate sweep for each dataset
        for dataset in datasets:
            # Create a copy of base config for this dataset
            sweep_config = base_sweep_config.copy()
            sweep_config["parameters"] = base_sweep_config["parameters"].copy()
            
            # Set fixed dataset
            sweep_config["parameters"]["data_name"] = {"value": dataset}
            
            # Set sweep name to include model and dataset
            sweep_config["name"] = f"{args.model}-{dataset}"
            
            # Initialize sweep in the project
            sweep_id = wandb.sweep(sweep_config, project=args.project)
            
            print(f"\nStarting sweep for dataset {dataset}")
            print(f"Project: {args.project}")
            print(f"Sweep name: {sweep_config['name']}")
            print(f"Sweep ID: {sweep_id}")
            print(f"Running {args.count} trials...")
            
            # Run sweep for this dataset
            if args.count == -1:
                wandb.agent(sweep_id, function=train_with_wandb)
            else:
                wandb.agent(sweep_id, function=train_with_wandb, count=args.count)
            
            
            print(f"Completed sweep for dataset {dataset}")
            
            # Export best parameters after sweep completion
            print(f"\nExporting best parameters for dataset {dataset}...")
            api = wandb.Api()
            sweep = api.sweep(f"{args.project}/{sweep_id}")
            
            # Create output directory if it doesn't exist
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Export best parameters
            try:
                best_run = sweep.best_run()
                print(best_run.config)
                best_params = {
                    "model_name": args.model,
                    "data_name": dataset,
                    "best_metric": {
                        args.metric: best_run.summary.get(args.metric)
                    },
                    "parameters": {}
                }
                
                # Add all config parameters
                for key, value in best_run.config.items():
                    if key not in ["_wandb", "wandb_version"]:
                        best_params["parameters"][key] = value
                
                # Save to yaml file
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(args.output_dir, f"best_{args.model}_{dataset}_{timestamp}.yaml")
                
                with open(output_file, "w") as f:
                    yaml.dump(best_params, f, default_flow_style=False)
                
                print(f"Best parameters exported to: {output_file}")
                print(f"Best {args.metric}: {best_run.summary.get(args.metric)}")
            except Exception as e:
                print(f"Failed to export best parameters: {e}")
    else:
        # If continuing existing sweep, just run the agent
        sweep_id = args.sweep_id
        try:
            sweep = wandb.api.sweep(f"{args.project}/{sweep_id}")
            print(f"\nContinuing sweep {sweep_id} in project {args.project}")
            print(f"Sweep name: {sweep.name}")
        except:
            print(f"\nContinuing sweep {sweep_id}")
            
        wandb.agent(sweep_id, function=train_with_wandb, count=args.count)
        
        # Export best parameters after sweep completion
        print("\nExporting best parameters...")
        try:
            api = wandb.Api()
            sweep = api.sweep(f"{args.project}/{sweep_id}")
            # Extract dataset from sweep name
            dataset = sweep.name.split("-")[-1]
            
            # Create output directory if it doesn't exist
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Export best parameters
            try:
                best_run = sweep.best_run()
                print(best_run.config)
                best_params = {
                    "model_name": args.model,
                    "data_name": dataset,
                    "best_metric": {
                        args.metric: best_run.summary.get(args.metric)
                    },
                    "parameters": {}
                }
                
                # Add all config parameters
                for key, value in best_run.config.items():
                    if key not in ["_wandb", "wandb_version"]:
                        best_params["parameters"][key] = value
                
                # Save to yaml file
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(args.output_dir, f"best_{args.model}_{dataset}_{timestamp}.yaml")
                
                with open(output_file, "w") as f:
                    yaml.dump(best_params, f, default_flow_style=False)
                
                print(f"Best parameters exported to: {output_file}")
                print(f"Best {args.metric}: {best_run.summary.get(args.metric)}")
            except Exception as e:
                print(f"Failed to export best parameters: {e}")
        except Exception as e:
            print(f"Failed to export best parameters: {e}")

if __name__ == "__main__":
    run_sweep() 