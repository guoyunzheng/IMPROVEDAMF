"""Epoch-based FM/MF sampling schedule shared by Mean Flow workspaces."""

import math


def scheduled_flow_ratio(epoch, num_epochs, flow_ratio, schedule=None):
    """Return the FM sample fraction for a zero-based absolute epoch.

    With the schedule enabled, fm_only_epochs sets an exact pure-FM length.
    Otherwise the first ceil(num_epochs * fm_only_fraction) epochs use pure FM.
    Remaining epochs use flow_ratio. A missing or disabled
    schedule preserves fixed-ratio training, including older saved configs.
    Recomputing from the absolute epoch also preserves the phase on resume.
    """
    if num_epochs <= 0 or epoch < 0:
        raise ValueError("num_epochs must be positive and epoch must be nonnegative")
    if not 0 <= flow_ratio <= 1:
        raise ValueError("flow_ratio must lie in [0, 1]")
    if schedule is None or not schedule.get("enabled", True):
        return flow_ratio

    fm_only_epochs = schedule.get("fm_only_epochs")
    if fm_only_epochs is not None:
        if type(fm_only_epochs) is not int or not 0 <= fm_only_epochs <= num_epochs:
            raise ValueError("fm_only_epochs must be an integer between 0 and num_epochs")
    else:
        fm_only_fraction = schedule.get("fm_only_fraction", 0.8)
        if not 0 <= fm_only_fraction <= 1:
            raise ValueError("fm_only_fraction must lie in [0, 1]")
        fm_only_epochs = math.ceil(num_epochs * fm_only_fraction)
    return 1.0 if epoch < fm_only_epochs else flow_ratio
