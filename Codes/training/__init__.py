"""
Training pipeline module for cell segmentation.

Provides loss functions, trainer, and training utilities.

Usage:
    from training import Trainer, get_loss_function, create_optimizer, create_scheduler

    # Create loss
    loss_fn = get_loss_function("bce_dice")

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, name="adamw", lr=1e-4)
    scheduler = create_scheduler(optimizer, name="cosine", num_epochs=100)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cuda"
    )

    # Train
    history = trainer.fit(num_epochs=100)
"""

from .losses import (
    DiceLoss,
    FocalLoss,
    BCEDiceLoss,
    CrossEntropyLoss,
    get_loss_function,
    list_available_losses,
    compute_class_weights,
)

from .trainer import (
    Trainer,
    create_optimizer,
    create_scheduler,
    compute_iou,
    compute_dice,
)

__all__ = [
    # Trainer
    "Trainer",
    "create_optimizer",
    "create_scheduler",
    # Metrics
    "compute_iou",
    "compute_dice",
    # Loss functions
    "DiceLoss",
    "FocalLoss",
    "BCEDiceLoss",
    "CrossEntropyLoss",
    # Factory
    "get_loss_function",
    "list_available_losses",
    # Utilities
    "compute_class_weights",
]
