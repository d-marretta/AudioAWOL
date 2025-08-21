import yaml
import lightning as L
import torch
from torch.utils.data import DataLoader, random_split
from src.training.dataset import AudioAWOLDataset
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import wandb
from src.training.RealNVP import RealNVPLightning
import os

def main():
    wandb.login()
    with open("./configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open("./configs/training_config.yaml", "r") as cf:
        training_conf = yaml.safe_load(cf)

    model_path = config['REALNVP_PATH']
    os.makedirs(f"{model_path}/checkpoints", exist_ok=True)

    # Get data
    dataset = AudioAWOLDataset(config['DATASET_DIR'], config['PCA_PATH'])
    train_data, val_data, test_data = random_split(dataset, [0.8, 0.1, 0.1])
    train_dataloader = DataLoader(dataset=train_data, batch_size=training_conf['batch_size'], shuffle=True)
    val_dataloader = DataLoader(dataset=val_data, batch_size=training_conf['batch_size'], shuffle=False)
    test_dataloader = DataLoader(dataset=test_data, batch_size=training_conf['batch_size'], shuffle=False)
    
    model = RealNVPLightning(training_conf)

    wandb_logger = WandbLogger(
        project='AudioAWOL',
        name='AudioAWOL_training1',
        offline=False,
        log_model='all'
    )
    wandb_logger.experiment.config.update(config)

    early_stopping_cb = EarlyStopping(monitor='val_loss', patience=5)
    model_checkpoint_cb = ModelCheckpoint(dirpath=f'{model_path}/checkpoints', 
                                          filename='{epoch}-{val_loss:.2f}',
                                          monitor='val_loss',
                                          mode='min')
    trainer = L.Trainer(
        max_epochs=config["epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[config["gpu_id"]] if torch.cuda.is_available() else "auto",
        logger=wandb_logger,
        callbacks=[early_stopping_cb,model_checkpoint_cb]
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, dataloaders=test_dataloader)


if __name__ == '__main__':
    main()
