import wandb

wandb.init(project="intersection-v0-dqn")

wandb.config = {
  "learning_rate": 5e-4,
  "buffer_size": 15000,
  "epochs": 2e4,
  "batch_size": 32,
  "gamma": 0.8,
  "exploration_fraction": 0.7,
}

wandb.log({"loss": loss})

