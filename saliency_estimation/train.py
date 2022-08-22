from argparse import ArgumentParser

import dataloaders
import models
import pytorch_lightning as pl
import utils


def main(args):
    """
    main function for training
    Args:
        args (argparse.Namespace): Arguments for training
    """
    # seeding
    if args.seed != "None":
        pl.seed_everything(args.seed)

    # DataLoaders
    if args.dataloader == "cat2000":
        train_dataloader, val_dataloader = dataloaders.cat2000.build_cat2000(...)
    elif args.dataloader == "mit1003":
        train_dataloader, val_dataloader = dataloaders.mit1003.build_mit1003(...)

    # System
    if args.model == "DeepGaze1":
        model = models.DeepGaze1(args)
    elif args.model == "DeepGaze2":
        model = models.DeepGaze2(args)
    elif args.model == "DeepGaze2E":
        model = models.DeepGaze2E(args)
    elif args.model == "DeepGaze3":
        model = models.DeepGaze3(args)
    else:
        raise NotImplementedError(f"{args.model} model not implemented!")

    # Logger
    # TODO: Add option for non-wandb logger
    logger_dir = f"logs/{args.model}"
    utils.paths.create_folders(f"{logger_dir}/wandb")
    # TODO: Add arg for project name
    logger = pl.loggers.WandbLogger(
        project=f"saliency_estimation_{args.model}", log_model="all", save_dir=logger_dir
    )
    experiment_dir = "/".join(logger.experiment.dir.split("/")[:-1])
    logger.watch(model, log="all")

    # Profiler
    profiler = pl.profiler.PyTorchProfiler()

    # Callbacks
    # device_stats = pl.callbacks.DeviceStatsMonitor()
    model_summary = pl.callbacks.ModelSummary()
    timer = pl.callbacks.Timer()
    model_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=f"{experiment_dir}/checkpoints",
        filename="{epoch}-{val_loss:.2f}-{val_accuracy:.2f}",
        monitor="val_loss",
        save_last=True,
        save_top_k=5,
        mode="min",
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step", log_momentum=True)
    callbacks = [model_summary, timer, model_checkpoint, lr_monitor]

    # Trainer
    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.epochs,
        log_every_n_steps=1,
        enable_checkpointing=True,
        check_val_every_n_epoch=3,
        enable_progress_bar=True,
        logger=logger,
        profiler=profiler,
        callbacks=callbacks,
    )

    # Validate at the start
    trainer.validate(model, val_dataloader)

    # Training
    trainer.fit(model, train_dataloader, val_dataloader, args.ckpt_path)

    # Evaluation
    trainer.test(dataloaders=val_dataloader, ckpt_path="best")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--ckpt_path", type=utils.str2none)
    parser.add_argument("--val_ratio", type=float)
    parser.add_argument("--csv_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--labels", type=utils.str2list)
    parser.add_argument("--seed")
    parser.add_argument("--model", type=str)
    arguments = parser.parse_args()

    print(arguments)

    main(arguments)
