import os

import torch
from tqdm import tqdm

import happier.lib as lib


def _calculate_loss_and_backward(
    config,
    net,
    batch,
    relevance_fn,
    criterion,
    optimizer,
    scaler,
    epoch,
):
    with torch.cuda.amp.autocast(enabled=(scaler is not None)):
        di = net(batch["image"].cuda())
        labels = batch["label"].cuda()

        logs = {}
        losses = []
        for crit, weight in criterion:
            loss = crit(
                di,
                labels,
                relevance_fn=relevance_fn,
                indexes=batch["index"].cuda()
            )

            loss = loss.mean()
            losses.append(weight * loss)

            logs[f"{crit.__class__.__name__}_l{crit.hierarchy_level}"] = loss.item()

    total_loss = sum(losses)
    if scaler is None:
        total_loss.backward()
    else:
        scaler.scale(total_loss).backward()

    logs["total_loss"] = total_loss.item()
    _ = [loss.detach_() for loss in losses]
    total_loss.detach_()
    return logs


def base_training_loop(
    config,
    net,
    loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    epoch,
):
    meter = lib.DictAverage()
    net.train()
    net.zero_grad()

    iterator = tqdm(loader, disable=os.getenv('TQDM_DISABLE'))
    for i, batch in enumerate(iterator):
        logs = _calculate_loss_and_backward(
            config,
            net,
            batch,
            loader.dataset.compute_relevance_on_the_fly,
            criterion,
            optimizer,
            scaler,
            epoch,
        )

        if config.experience.record_gradient:
            if scaler is not None:
                for opt in optimizer.values():
                    scaler.unscale_(opt)

            logs["gradient_norm"] = lib.get_gradient_norm(net)

        if config.experience.gradient_clipping_norm is not None:
            if (scaler is not None) and (not config.experience.record_gradient):
                for opt in optimizer.values():
                    scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), config.experience.gradient_clipping_norm)

        for key, opt in optimizer.items():
            if (
                (config.experience.warmup_step is not None)
                and (config.experience.warmup_step >= epoch)
                and (key in config.experience.warmup_keys)
            ):
                if i == 0:
                    lib.LOGGER.warning("Warmimg UP")
                continue
            if scaler is None:
                opt.step()
            else:
                scaler.step(opt)

        for crit, _ in criterion:
            if hasattr(crit, 'update'):
                crit.update(scaler)

        net.zero_grad()
        _ = [crit.zero_grad() for crit, w in criterion]

        for sch in scheduler["on_step"]:
            sch.step()

        if scaler is not None:
            scaler.update()

        meter.update(logs)
        if not os.getenv('TQDM_DISABLE'):
            iterator.set_postfix(meter.avg)
        else:
            if (i + 1) % config.experience.print_freq == 0:
                lib.LOGGER.info(f'Iteration : {i}/{len(loader)}')
                for k, v in logs.items():
                    lib.LOGGER.info(f'Loss: {k}: {v} ')

        if config.experience.DEBUG:
            if (i+1) > int(config.experience.DEBUG):
                break

    for crit, _ in criterion:
        if hasattr(crit, 'optimize_proxies'):
            crit.optimize_proxies(loader.dataset.compute_relevance_on_the_fly)

    return meter.avg
