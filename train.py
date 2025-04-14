# Log test metrics to wandb
if wandb.run is None and config["train_config"]["log_wandb"]:
    wandb.log({
        "epoch_test_auc": test_auc,
        "epoch_test_acc": test_acc,
        "epoch_test_rmse": test_rmse
    }) 