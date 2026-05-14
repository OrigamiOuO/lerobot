import wandb
import os

# 恢复到你刚才同步的同一个 Run 中
run = wandb.init(
    project="lerobot", 
    entity="origamiouo-shanghaitech-university",
    id="xiyr6niw", 
    resume="must"
)
print("Wandb initialized. Creating artifact...")

checkpoint_dir = "checkpoints/hao_proj/black_board_one/no_tac/checkpoints"
artifact = wandb.Artifact(name="no_tac_checkpoints", type="model")
artifact.add_dir(checkpoint_dir)

print("Uploading checkpoints to wandb... This may take a while.")
run.log_artifact(artifact)
run.finish()
print("Upload complete!")
