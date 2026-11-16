"""Batch progress for the learned Kinematic MeanFlow split generator."""

import torch

from source.common.mean_flow_schedule import fm_only_epoch_count


def set_kmf_progress(model, ema_model, epoch, batch_idx, num_epochs,
                     batches_per_epoch, flow_schedule=None):
    """Set absolute mixed-phase progress before a training batch.

    Validation reads the stored buffer and never calls this function. Computing
    from the restored epoch also reproduces the schedule after a checkpoint.
    """
    if getattr(model, 'mf_target_type', 'meanflow') != 'kmf_learnable':
        return {}
    if batches_per_epoch <= 0 or not 0 <= batch_idx < batches_per_epoch:
        raise ValueError('Expected a training batch within a nonempty epoch')
    if not 0 <= epoch < num_epochs:
        raise ValueError('Training epoch is outside the KMF schedule')
    first_mixed_epoch = fm_only_epoch_count(num_epochs, flow_schedule)
    mixed_batches = (num_epochs - first_mixed_epoch) * batches_per_epoch
    progress = 0.0
    if epoch >= first_mixed_epoch and model.flow_ratio < 1.0 and mixed_batches > 1:
        mixed_batch = (epoch - first_mixed_epoch) * batches_per_epoch + batch_idx
        progress = mixed_batch / (mixed_batches - 1)
    with torch.no_grad():
        model.kmf_generator.progress.fill_(progress)
        if ema_model is not None:
            # EMAModel updates parameters, so its progress buffer needs a copy.
            ema_model.kmf_generator.progress.copy_(model.kmf_generator.progress)
    return {'kmf_progress': progress, 'kmf_gamma': model.kmf_gamma_max * (2 * progress - 1)}
