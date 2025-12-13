import queue
import numpy as np
import sounddevice as sd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# Si PyCharm te fait une fenêtre blanche, force un backend GUI
# Essaie d'abord TkAgg (le plus commun). Si erreur, commente.
matplotlib.use("TkAgg")


def rms_db(x: np.ndarray, eps: float = 1e-12) -> float:
    x = x.reshape(-1).astype(np.float32)
    rms = float(np.sqrt(np.mean(x * x) + eps))
    return 20.0 * np.log10(rms + eps)


def main(sample_rate: int = 16000, frame_ms: int = 20, seconds_window: int = 20, input_device: int | None = None):
    frame_len = int(sample_rate * frame_ms / 1000)
    n = int(seconds_window * 1000 / frame_ms)

    if input_device is not None:
        sd.default.device = (input_device, None)

    # Debug device
    in_dev = sd.default.device[0]
    print("Input device id:", in_dev)
    try:
        print(sd.query_devices(in_dev, "input"))
    except Exception:
        print("Impossible de query le device, liste complète:")
        print(sd.query_devices())

    q: "queue.Queue[float]" = queue.Queue()
    hist = deque([-100.0] * n, maxlen=n)

    def callback(indata, frames, time_info, status):
        # indata shape: (frames, channels)
        db = rms_db(indata)
        # Si tu veux inspecter vite fait
        # print(f"db={db:.1f}")
        q.put(db)

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=frame_len,
        callback=callback,
    )
    stream.start()

    fig, ax = plt.subplots()
    x = np.arange(n)
    y = np.array(hist, dtype=np.float32)
    (line,) = ax.plot(x, y)

    ax.set_title("Mic level (RMS dBFS)")
    ax.set_xlabel(f"frames (window {seconds_window}s)")
    ax.set_ylabel("dBFS")
    ax.set_ylim(-100, 0)
    ax.grid(True)

    txt = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")

    def update(_):
        # vide la queue
        updated = False
        while True:
            try:
                db = q.get_nowait()
            except queue.Empty:
                break
            hist.append(db)
            updated = True

        if not updated:
            return (line, txt)

        y = np.array(hist, dtype=np.float32)
        line.set_ydata(y)

        p10 = float(np.percentile(y, 10))
        p50 = float(np.percentile(y, 50))
        p95 = float(np.percentile(y, 95))
        cur = float(y[-1])

        suggested = max(p95 + 3.0, -40.0)
        txt.set_text(
            f"cur={cur:.1f} dBFS\np10={p10:.1f}  p50={p50:.1f}  p95={p95:.1f}\n"
            f"suggested_threshold≈{suggested:.1f} dBFS"
        )

        # autoscale doux
        ymin = min(-100.0, p10 - 10.0)
        ymax = min(0.0, p95 + 15.0)
        ax.set_ylim(ymin, ymax)

        return (line, txt)

    ani = FuncAnimation(fig, update, interval=50, blit=False)

    try:
        plt.show()
    finally:
        stream.stop()
        stream.close()


if __name__ == "__main__":
    # Si besoin, décommente pour choisir ton micro:
    # print(sd.query_devices())
    # main(input_device=TON_ID)
    main()
