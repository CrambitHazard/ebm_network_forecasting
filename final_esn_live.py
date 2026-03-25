import io
import os
import time
from collections import deque

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(_SCRIPT_DIR, "data", "dataset1", "live_packets.csv")

PRIME_ON_FIRST_READ = True

MAX_ROWS_PER_STEP = 400
STEPS_PER_LOOP = 25

input_size = 1
reservoir_size = 100
output_size = 1

np.random.seed(42)

Win = np.random.rand(reservoir_size, input_size) - 0.5
W = np.random.rand(reservoir_size, reservoir_size) - 0.5

eigvals = np.linalg.eigvals(W)
W *= 0.9 / max(abs(eigvals))

Wout = np.random.randn(output_size, reservoir_size) * 0.03
Wout_bias = np.zeros((output_size, 1))

state = np.zeros((reservoir_size, 1))

leak_rate = 0.58

NLMS_MU = 0.35
NLMS_EPS = 1e-3
NLMS_MU_MAX = 0.9
WOUT_NORM_CAP = 4.0
BIAS_CLIP = 4.0

REFINE_GAMMA = 0.08
REFINE_NOISE_STD = 0.002
ENERGY_LAM = 0.01

_scale_ema = 120.0
SCALE_EMA = 0.12

window = 50
actual = deque(maxlen=window)
predicted = deque(maxlen=window)

last_len = 0
last_energy = 0.0
_csv_primed = False

plt.ion()
fig, ax = plt.subplots(figsize=(10, 5))
plt.show(block=False)


def energy(y, state, Wout, bias, lam=ENERGY_LAM):
    readout = Wout @ state + bias
    if readout.shape != y.shape:
        raise ValueError("y shape must match readout shape.")
    return float(np.sum((y - readout) ** 2) + lam * np.sum(state**2))


def draw_chart():
    """Redraw axes from deques (same colors/labels as reference figure)."""
    ax.clear()
    n = len(actual)
    x = range(n)
    point_kw = {"marker": "o", "markersize": 5} if n < 2 else {}
    ax.plot(
        x,
        list(actual),
        color="tab:blue",
        linewidth=1.2,
        label="Actual Packets/sec",
        **point_kw,
    )
    ax.plot(
        x,
        list(predicted),
        color="tab:orange",
        linewidth=1.2,
        label="Predicted (ESN)",
        **point_kw,
    )
    ax.set_title(f"Live Network Load | Energy = {last_energy:.2f}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Packets/sec")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, window - 1)
    ax.autoscale_view(scalex=False, scaley=True)
    ax.margins(y=0.08)
    fig.canvas.draw()
    fig.canvas.flush_events()


while True:
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) < 10:
            time.sleep(1)
            continue

        blob = b""
        for _attempt in range(8):
            try:
                with open(file_path, "rb") as fh:
                    blob = fh.read()
                break
            except PermissionError:
                time.sleep(0.05)
        if not blob.strip():
            time.sleep(0.2)
            continue

        try:
            data = pd.read_csv(
                io.BytesIO(blob),
                encoding="latin1",
                on_bad_lines="skip",
            )
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            time.sleep(0.2)
            continue
        current_len = len(data)

        if PRIME_ON_FIRST_READ and not _csv_primed:
            last_len = current_len
            _csv_primed = True
            time.sleep(1)
            continue

        steps = 0
        while current_len > last_len and steps < STEPS_PER_LOOP:
            delta = current_len - last_len
            packets = min(MAX_ROWS_PER_STEP, delta)
            last_len += packets

            _scale_ema = (1.0 - SCALE_EMA) * _scale_ema + SCALE_EMA * float(max(packets, 1.0))
            norm = max(_scale_ema, 25.0)
            u = np.array([[packets / norm]])
            target = np.array([[packets / norm]])

            state = (1 - leak_rate) * state + leak_rate * np.tanh(Win @ u + W @ state)
            y_readout = Wout @ state + Wout_bias
            y = y_readout.copy()
            y = y + np.random.normal(0.0, REFINE_NOISE_STD, size=y.shape)
            y = (1.0 - 2.0 * REFINE_GAMMA) * y + 2.0 * REFINE_GAMMA * y_readout

            error = target - y
            den = float(state.T @ state) + NLMS_EPS
            mu = min(NLMS_MU / den, NLMS_MU_MAX)
            Wout += mu * (error @ state.T)
            Wout_bias += mu * error

            wn = float(np.linalg.norm(Wout.reshape(-1)))
            if wn > WOUT_NORM_CAP and wn > 0.0:
                Wout *= WOUT_NORM_CAP / wn
            Wout_bias = np.clip(Wout_bias, -BIAS_CLIP, BIAS_CLIP)

            last_energy = energy(y, state, Wout, Wout_bias)

            actual.append(packets)
            pred_pkt = float(np.asarray(y).reshape(-1)[0] * norm)
            pred_pkt = float(np.clip(pred_pkt, -5.0 * norm, 5.0 * norm))
            predicted.append(pred_pkt)
            steps += 1

        if len(actual) > 0:
            draw_chart()
            plt.pause(0.05)
        else:
            ax.clear()
            ax.set_title("Live Network Load")
            ax.text(
                0.5,
                0.5,
                "Waiting for data…\n"
                "Live: set PRIME_ON_FIRST_READ True and append rows (see scripts/capture-live-packets.ps1).\n"
                "Or set it False and keep a non-empty CSV; this script will replay it in chunks.",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
            )
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.05)

        time.sleep(1)

    except Exception as err:
        print("Retrying...", err)
        time.sleep(1)
