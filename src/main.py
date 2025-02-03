import os
import hydra
import wandb
from omegaconf import OmegaConf
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

from src.model.model_wrapper import ModelWrapper
from src.dataset.data_module import DataModule


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="main",
)
def main(cfg):
    torch.set_float32_matmul_precision("medium")
    
    output_dir = hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    print("output_dir:", output_dir)

    logger = WandbLogger(
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            name=cfg.wandb.name + " (" + output_dir.split("/")[-1] + ")",
            save_dir=output_dir,
        )

    callbacks = []
    callbacks.append(LearningRateMonitor("step"))
    callbacks.append(
        ModelCheckpoint(
            os.path.join(output_dir, "ckpts"),
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            monitor="info/global_step",
            mode="max",
        )
    )

    trainer = Trainer(
        max_epochs=-1,
        num_nodes=cfg.trainer.num_nodes,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        strategy=(
            "ddp_find_unused_parameters_true"
            if torch.cuda.device_count() > 1
            else "auto"
        ),
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=None,
        enable_progress_bar=True,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        precision="bf16-mixed",
        max_steps=cfg.trainer.max_steps,
        inference_mode=False if cfg.mode == "test" else True,
    )
    torch.manual_seed(cfg.seed + trainer.global_rank)

    model_wrapper = ModelWrapper(cfg)
    data_module = DataModule(cfg, global_rank=trainer.global_rank)

    if cfg.mode == "train":
        trainer.fit(model_wrapper, datamodule=data_module)
    else:
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=cfg.checkpointing.load,
        )
        #print(trainer.callback_metrics)

if __name__ == "__main__":
    main()
    