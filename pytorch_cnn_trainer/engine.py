"""
CNN Training Script. 
Improvement from Imagenet script is taken from ross wightman
One of the orignal script is here
https://github.com/rwightman/pytorch-image-models/blob/master/train.py
https://github.com/pytorch/examples/tree/master/imagenet
This is just simplified version of both of them, trying to provide flexibility and modularity.

1. Support mixed precision training with PyTorch 1.6 (soon).
2. Early Stopping. (Done)
3. Add torchvision support. (Done)
4. fit() function (Done)
5. Keras like history object. (TODO)
6. Sanity Checker. (Done)
7. Gradient Penalty. (Done)
8. Mixed Precision Training. (Done)
"""

import torch
from torch.cuda import amp
from pytorch_cnn_trainer import utils
from tqdm import tqdm
import time

# import model
from collections import OrderedDict

__all__ = [
    "train_step",
    "val_step",
    "fit",
    "train_sanity_fit",
    "val_sanity_fit",
    "sanity_fit",
]


def train_step(
    model,
    train_loader,
    criterion,
    device,
    optimizer,
    scheduler=None,
    num_batches: int = None,
    log_interval: int = 100,
    grad_penalty: bool = False,
    fp16_scaler=None,
):
    """
    Performs one step of training. Calculates loss, forward pass, computes gradient and returns metrics.
    Args:
        model : A pytorch CNN Model.
        train_loader : Train loader.
        criterion : Loss function to be optimized.
        device : "cuda" or "cpu"
        optimizer : Torch optimizer to train.
        scheduler : Learning rate scheduler.
        num_batches : (optional) Integer To limit training to certain number of batches.
        log_interval : (optional) Defualt 100. Integer to Log after specified batch ids in every batch.
        grad_penalty : (optional) To penalize with l2 norm for big gradients.
        fp16_scaler: (optional)  Pass torch.cuda.amp.GradScaler() for fp16 precision Training.
    """

    start_train_step = time.time()

    model.train()
    last_idx = len(train_loader) - 1
    batch_time_m = utils.AverageMeter()
    # data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()
    cnt = 0
    batch_start = time.time()
    # num_updates = epoch * len(loader)

    for batch_idx, (inputs, target) in enumerate(train_loader):
        last_batch = batch_idx == last_idx
        # data_time_m.update(time.time() - batch_start)
        inputs = inputs.to(device)
        target = target.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        if fp16_scaler is not None:
            with amp.autocast():
                output = model(inputs)
                loss = criterion(output, target)
                # Scale the loss using Grad Scaler

            if grad_penalty is True:
                # Scales the loss for autograd.grad's backward pass, resulting in scaled grad_params
                scaled_grad_params = torch.autograd.grad(
                    fp16_scaler.scale(loss), model.parameters(), create_graph=True
                )
                # Creates unscaled grad_params before computing the penalty. scaled_grad_params are
                # not owned by any optimizer, so ordinary division is used instead of fp16_scaler.unscale_:
                inv_scale = 1.0 / fp16_scaler.get_scale()
                grad_params = [p * inv_scale for p in scaled_grad_params]
                # Computes the penalty term and adds it to the loss
                with amp.autocast():
                    grad_norm = 0
                    for grad in grad_params:
                        grad_norm += grad.pow(2).sum()

                    grad_norm = grad_norm.sqrt()
                    loss = loss + grad_norm

            fp16_scaler.scale(loss).backward()
            # Step using fp16_scaler.step()
            fp16_scaler.step(optimizer)
            # Update for next iteration
            fp16_scaler.update()

        else:
            output = model(inputs)
            loss = criterion(output, target)

            if grad_penalty is True:
                # Create gradients
                grad_params = torch.autograd.grad(
                    loss, model.parameters(), create_graph=True
                )
                # Compute the L2 Norm as penalty and add that to loss
                grad_norm = 0
                for grad in grad_params:
                    grad_norm += grad.pow(2).sum()
                grad_norm = grad_norm.sqrt()
                loss = loss + grad_norm

            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        cnt += 1
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

        top1_m.update(acc1.item(), output.size(0))
        top5_m.update(acc5.item(), output.size(0))
        losses_m.update(loss.item(), inputs.size(0))

        batch_time_m.update(time.time() - batch_start)
        batch_start = time.time()
        if last_batch or batch_idx % log_interval == 0:  # If we reach the log intervel
            print(
                "Batch Train Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                "Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  "
                "Top 1 Accuracy: {top1.val:>7.4f} ({top1.avg:>7.4f})  "
                "Top 5 Accuracy: {top5.val:>7.4f} ({top5.avg:>7.4f})".format(
                    batch_time=batch_time_m, loss=losses_m, top1=top1_m, top5=top5_m
                )
            )

        if num_batches is not None:
            if cnt >= num_batches:
                end_train_step = time.time()
                metrics = OrderedDict(
                    [("loss", losses_m.avg), ("top1", top1_m.avg), ("top5", top5_m.avg)]
                )
                print(f"Done till {num_batches} train batches")
                print(
                    f"Time taken for train step = {end_train_step - start_train_step} sec"
                )
                return metrics

    metrics = OrderedDict(
        [("loss", losses_m.avg), ("top1", top1_m.avg), ("top5", top5_m.avg)]
    )
    end_train_step = time.time()
    print(f"Time taken for train step = {end_train_step - start_train_step} sec")
    return metrics


def val_step(
    model, val_loader, criterion, device, num_batches=None, log_interval: int = 100
):
    """
    Performs one step of validation. Calculates loss, forward pass and returns metrics.
    Args:
        model : A pytorch CNN Model.
        val_loader : Validation loader.
        criterion : Loss function to be optimized.
        device : "cuda" or "cpu"
        num_batches : (optional) Integer To limit validation to certain number of batches.
        log_interval : (optional) Defualt 100. Integer to Log after specified batch ids in every batch.
    """
    start_test_step = time.time()
    last_idx = len(val_loader) - 1
    batch_time_m = utils.AverageMeter()
    # data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()
    cnt = 0
    model.eval()
    batch_start = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(val_loader):
            last_batch = batch_idx == last_idx
            inputs = inputs.to(device)
            target = target.to(device)

            output = model(inputs)
            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            reduced_loss = loss.data

            losses_m.update(reduced_loss.item(), inputs.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))
            batch_time_m.update(time.time() - batch_start)

            batch_start = time.time()

            if (
                last_batch or batch_idx % log_interval == 0
            ):  # If we reach the log intervel
                print(
                    "Batch Inference Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                    "Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  "
                    "Top 1 Accuracy: {top1.val:>7.4f} ({top1.avg:>7.4f})  "
                    "Top 5 Accuracy: {top5.val:>7.4f} ({top5.avg:>7.4f})".format(
                        batch_time=batch_time_m, loss=losses_m, top1=top1_m, top5=top5_m
                    )
                )

            if num_batches is not None:
                if cnt >= num_batches:
                    end_test_step = time.time()
                    metrics = OrderedDict(
                        [
                            ("loss", losses_m.avg),
                            ("top1", top1_m.avg),
                            ("top5", top5_m.avg),
                        ]
                    )
                    print(f"Done till {num_batches} validation batches")
                    print(
                        f"Time taken for validation step = {end_test_step - start_test_step} sec"
                    )
                    return metrics

        metrics = OrderedDict(
            [("loss", losses_m.avg), ("top1", top1_m.avg), ("top5", top5_m.avg)]
        )
        print("Finished the validation epoch")

    end_test_step = time.time()
    print(f"Time taken for validation step = {end_test_step - start_test_step} sec")
    return metrics


def fit(
    epochs,
    model,
    train_loader,
    valid_loader,
    criterion,
    device,
    optimizer,
    scheduler=None,
    early_stopper=None,
    num_batches: int = None,
    log_interval: int = 100,
    grad_penalty: bool = False,
    use_fp16: bool = False,
    swa_start: int = None,
    swa_scheduler=None,
):
    """
    A fit function that performs training for certain number of epochs.
    Args:
        epochs: Number of epochs to train.
        model : A pytorch CNN Model.
        train_loader : Train loader.
        val_loader : Validation loader.
        criterion : Loss function to be optimized.
        device : "cuda" or "cpu"
        optimizer : PyTorch optimizer.
        scheduler : (optional) Learning Rate scheduler.
        early_stopper: (optional) A utils provied early stopper, based on validation loss.
        num_batches : (optional) Integer To limit validation to certain number of batches.
        log_interval : (optional) Defualt 100. Integer to Log after specified batch ids in every batch.
        use_fp16 : (optional) To use Mixed Precision Training using float16 dtype.
        swa_start : (optional) To use Stochastic Weighted Averaging while Training
        swa_scheduler : (optional) A torch.optim.swa_utils.scheduler to be used during SWA Training epochs.
    """
    # Declaring necessary variables required to add to keras like history object
    history = {}
    loss = []
    top1_acc = []
    top5_acc = []
    loss_v = []
    top1_acc_v = []
    top5_acc_v = []

    if swa_start is not None:
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        print("Training with Stochastic Weighted averaging (SWA) Scheduler")

    if use_fp16 is True:
        print("Training with Mixed precision fp16 scaler")
        scaler = amp.GradScaler()
    else:
        scaler = None

    for epoch in tqdm(range(epochs)):
        print()
        print(f"Training Epoch = {epoch}")
        if swa_start is not None:
            if epoch > swa_start:
                train_metrics = train_step(
                    model,
                    train_loader,
                    criterion,
                    device,
                    optimizer,
                    scheduler=None,
                    num_batches=num_batches,
                    log_interval=log_interval,
                    grad_penalty=grad_penalty,
                    fp16_scaler=scaler,
                )

                swa_model.update_parameters(model)
                swa_scheduler.step()

            else:
                train_metrics = train_step(
                    model,
                    train_loader,
                    criterion,
                    device,
                    optimizer,
                    scheduler,
                    num_batches,
                    log_interval,
                    grad_penalty,
                    fp16_scaler=scaler,
                )

        else:
            train_metrics = train_step(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
                scheduler,
                num_batches,
                log_interval,
                grad_penalty,
                fp16_scaler=scaler,
            )

        loss_f, top1_acc_f, top5_acc_f = [
            train_metrics[i] for i in ("loss", "top1", "top5")
        ]
        loss.append(loss_f)
        top1_acc.append(top1_acc_f)
        top5_acc.append(top5_acc_f)

        print()
        print(f"Validating Epoch = {epoch}")
        valid_metrics = val_step(
            model, valid_loader, criterion, device, num_batches, log_interval
        )

        loss_f, top1_acc_f, top5_acc_f = [
            valid_metrics[i] for i in ("loss", "top1", "top5")
        ]
        loss_v.append(loss_f)
        top1_acc_v.append(top1_acc_f)
        top5_acc_v.append(top5_acc_f)

        validation_loss = valid_metrics["loss"]
        if early_stopper is not None:
            early_stopper(validation_loss, model=model)

            if early_stopper.early_stop:
                print("Saving Model and Early Stopping")
                print("Early Stopping. Ran out of Patience for validation loss")
                break

        print("Done Training, Model Saved to Disk")

    history = {
        "train": {"top1_acc": top1_acc, "top5_acc": top5_acc, "loss": loss},
        "val": {"top1_acc":top1_acc_v, "top5_acc": top5_acc_v, "loss":loss_v }
    }

    return history  # For now, we should probably return an history object like keras.


def train_sanity_fit(
    model,
    train_loader,
    criterion,
    device,
    num_batches: int = None,
    log_interval: int = 100,
    grad_penalty: bool = False,
    use_fp16: bool = False,
):
    """
    Performs Sanity fit over train loader.
    Use this to dummy check your fit function. It does not calculate metrics, timing, or does checkpointing.
    It iterates over both train_loader and valid_loader for given batches.
    Note: - It does not to loss.backward().
    Args:
        model : A pytorch CNN Model.
        train_loader : Train loader.
        criterion : Loss function to be optimized.
        device : "cuda" or "cpu"
        num_batches : (optional) Integer To limit sanity fit over certain batches. Useful is data is too big even for sanity check.
        log_interval : (optional) Defualt 100. Integer to Log after specified batch ids in every batch.
        use_fp16: : (optional) If True uses PyTorch native mixed precision Training.
    """
    model.train()
    cnt = 0
    last_idx = len(train_loader) - 1
    train_sanity_start = time.time()

    if use_fp16 is True:
        scaler = amp.GradScaler()

    for batch_idx, (inputs, target) in enumerate(train_loader):
        last_batch = batch_idx == last_idx
        # data_time_m.update(time.time() - batch_start)
        inputs = inputs.to(device)
        target = target.to(device)
        output = model(inputs)

        if use_fp16 is True:
            with amp.autocast():
                output = model(inputs)
                loss = criterion(output, target)

        else:
            loss = criterion(output, target)
            if grad_penalty is True:
                # Create gradients
                grad_params = torch.autograd.grad(
                    loss, model.parameters(), create_graph=True
                )
                # Compute the L2 Norm as penalty and add that to loss
                grad_norm = 0
                for grad in grad_params:
                    grad_norm += grad.pow(2).sum()
                grad_norm = grad_norm.sqrt()
                loss = loss + grad_norm

        cnt += 1

        if last_batch or batch_idx % log_interval == 0:
            print(f"Train Sanity check passed for batch till {batch_idx} batches")

        if num_batches is not None:
            if cnt >= num_batches:
                print(f"Sanity check passed till {cnt} train batches")
                print("All specicied batches done.")
                train_sanity_end = time.time()
                print(
                    f"Training Sanity check passed in time {train_sanity_end - train_sanity_start} !!"
                )
                return True

    train_sanity_end = time.time()
    print(
        f"Training Sanity check passed in time {train_sanity_end - train_sanity_start} !!"
    )
    return True


def val_sanity_fit(
    model,
    valid_loader,
    criterion,
    device,
    num_batches: int = None,
    log_interval: int = 100,
):
    """
    Performs Sanity fit over valid loader.
    Use this to dummy check your fit function. It does not calculate metrics, timing, or does checkpointing.
    It iterates over both train_loader and valid_loader for given batches.
    Note: - It does not to loss.backward().
    Args:
        model : A pytorch CNN Model.
        val_loader : Validation loader.
        criterion : Loss function to be optimized.
        device : "cuda" or "cpu"
        num_batches : (optional) Integer To limit sanity fit over certain batches. Useful is data is too big even for sanity check.
        log_interval : (optional) Defualt 100. Integer to Log after specified batch ids in every batch.
    """
    model.eval()
    cnt = 0
    last_idx = len(valid_loader) - 1
    val_sanity_start = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(valid_loader):
            last_batch = batch_idx == last_idx
            # data_time_m.update(time.time() - batch_start)
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)

            loss = criterion(output, target)
            cnt += 1

            if last_batch or batch_idx % log_interval == 0:
                print(
                    f"Validation Sanity check passed for batch till {batch_idx} batches"
                )

            if num_batches is not None:
                if cnt >= num_batches:
                    print(f"Sanity check passed till {cnt} validation batches")
                    print("All specicied batches done.")
                    val_sanity_end = time.time()
                    print(
                        f"Training Sanity check passed in time {val_sanity_end - val_sanity_start} !!"
                    )
                    return True

    val_sanity_end = time.time()
    print("Training Sanity check passed in time {val_sanity_end - val_sanity_start} !!")
    return True


def sanity_fit(
    model,
    train_loader,
    valid_loader,
    criterion,
    device,
    num_batches: int = None,
    log_interval: int = 100,
    grad_penalty: bool = False,
    use_fp16: bool = False,
):
    """
    Performs Sanity fit over train loader and valid loader.
    Use this to dummy check your fit function. It does not calculate metrics, timing, or does checkpointing.
    It iterates over both train_loader and valid_loader for given batches.
    Note: - It does not to loss.backward().
    Args:
        model : A pytorch CNN Model.
        train_loader : Training loader.
        val_loader : Validation loader.
        criterion : Loss function to be optimized.
        device : "cuda" or "cpu"
        num_batches : (optional) Integer To limit sanity fit over certain batches. Useful is data is too big even for sanity check.
        log_interval : (optional) Defualt 100. Integer to Log after specified batch ids in every batch.
    """
    # Train sanity check
    ts = train_sanity_fit(
        model,
        train_loader,
        criterion,
        device,
        num_batches,
        log_interval,
        grad_penalty,
        use_fp16,
    )
    vs = val_sanity_fit(
        model, valid_loader, criterion, device, num_batches, log_interval
    )
    # This line would run in case both pass otherwise it will create error naturally from torch.
    return True
