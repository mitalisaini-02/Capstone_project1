"""Snapshot power calculation using the repository's TR 37.885 equations."""
import numpy as np


def dbm_to_mw(dbm):
    return np.power(10.0, np.asarray(dbm) / 10)


def snapshot(model, tx_xy, rx_xy, channels, states):
    """Calculate every receiver/transmitter pair, summing same-channel interferers."""
    tx_xy, rx_xy = np.asarray(tx_xy), np.asarray(rx_xy)
    channels, states = np.asarray(channels), np.asarray(states)
    shape = (len(rx_xy), len(tx_xy))
    if (tx_xy.shape != (shape[1], 2) or rx_xy.shape != (shape[0], 2)
            or channels.shape != (shape[1],) or states.shape != shape
            or not np.isfinite(tx_xy).all() or not np.isfinite(rx_xy).all()):
        raise ValueError("Invalid radio snapshot")
    distance = np.linalg.norm(rx_xy[:, None] - tx_xy[None, :], axis=2)
    loss = np.full(shape, np.nan)
    for state in ("LOS", "NLOS", "NLOSv"):
        mask = states == state
        loss[mask] = model.path_loss_db(distance[mask], state)
        if state == "NLOSv":
            loss[mask] += np.maximum(0, model.blockage_params_db(distance[mask])[0])
    if not np.isfinite(loss).all():
        raise ValueError("Invalid propagation state")
    rx = dbm_to_mw(model.eirp_gain_db - loss)
    same_channel = channels[:, None] == channels[None, :]
    np.fill_diagonal(same_channel, False)
    interference = rx @ same_channel.astype(float)
    noise = float(dbm_to_mw(model.noise_dbm))
    linear = rx / (interference + noise)
    return distance, loss, rx, interference, noise, linear, 10 * np.log10(linear)
