import os
from argparse import ArgumentParser

import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from configs.factory import (
    MainConfig,
    ShapeEmbeddingConfig,
    FTNetConfig,
    MiscellaneusConfig,
)
from src.models.baseline import Baseline


def train():
    parsers = ArgumentParser()
    parsers.add_argument(
        "--config",
        dest="config",
        type=str,
        default="configs/main_config.yaml",
    )

    parsers.add_argument("--num-workers", dest="num_workers", type=int, default=4)
    parsers.add_argument("--device", dest="device", type=str, default="gpu")
    args = parsers.parse_args()

    with open(args.config, "r") as f:
        payload = yaml.load(f, Loader=yaml.FullLoader)
    main_config = MainConfig(
        **payload["main"], device=args.device, num_workers=args.num_workers
    )
    shape_embedding_config = ShapeEmbeddingConfig(**payload["shape_embedding"])
    ftnet_config = FTNetConfig(**payload["ftnet"])
    miscellaneous_config = MiscellaneusConfig(**payload["miscellaneous"])

    net = Baseline(
        main_config=main_config,
        ftnet_config=ftnet_config,
        shape_embedding_config=shape_embedding_config,
        miscs_config=miscellaneous_config,
    )

    model_checkpoint = ModelCheckpoint(every_n_epochs=5)
    early_stopping = EarlyStopping(monitor="epoch_loss", patience=20, mode="min")

    trainer = Trainer(
        accelerator=args.device,
        max_epochs=main_config.epochs,
        callbacks=[model_checkpoint, early_stopping],
        logger=None,
    )

    if main_config.train_from_ckpt:
        ckpt_path = main_config.ckpt_path
        trainer.fit(model=net, ckpt_path=ckpt_path)
    else:
        trainer.fit(model=net)

    torch.save(
        obj=net.state_dict(),
        f=os.path.join(main_config.save_path, main_config.model_name),
    )


if __name__ == "__main__":
    train()
