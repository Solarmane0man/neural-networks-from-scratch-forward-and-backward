"""
Neural Networks From Scratch: Forward and Backward scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""End-to-end demo: NumPy neural net from scratch on a nonlinear dataset."""
import numpy as np


def _nonlinear_dataset(n_samples=256, seed=0, label_noise=0.0):
    """Binary labels on noisy concentric rings (not linearly separable)."""
    rng = np.random.RandomState(seed)
    half = n_samples // 2
    t = rng.uniform(0.0, 2.0 * np.pi, size=half)
    r0 = 0.45 + rng.normal(0.0, 0.06, size=half)
    r1 = 1.15 + rng.normal(0.0, 0.06, size=half)
    x0 = np.column_stack([r0 * np.cos(t), r0 * np.sin(t)])
    x1 = np.column_stack([r1 * np.cos(t), r1 * np.sin(t)])
    x = np.vstack([x0, x1]).astype(np.float64)
    y = np.array([0] * half + [1] * half, dtype=np.int64)
    if label_noise > 0.0:
        flip = rng.rand(n_samples) < float(label_noise)
        y = y.copy()
        y[flip] = 1 - y[flip]
    idx = rng.permutation(n_samples)
    return x[idx], y[idx]


def _fresh_mlp(in_dim, n_classes, hidden=32, seed=1):
    """Return a freshly initialized untrained sequential MLP."""
    np.random.seed(int(seed))

    def init_fn(i, o):
        return initialize_weights(i, o, scheme="he")

    layers = [
        make_dense(in_dim, hidden, init_fn),
        make_activation("relu"),
        make_dense(hidden, hidden, init_fn),
        make_activation("relu"),
        make_dense(hidden, n_classes, init_fn),
    ]
    return make_sequential(layers)


def main():
    np.random.seed(0)
    x, y = _nonlinear_dataset(256, seed=0, label_noise=0.0)
    n_train = 192
    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train:], y[n_train:]
    in_dim, n_classes = int(x.shape[1]), 2

    designed, design_metrics = design_network(in_dim, n_classes, seed=0)
    print("design_network_accuracy:", round(float(design_metrics["accuracy"]), 4))

    # Fresh untrained net on the rings data: one train_step must drop loss.
    model = _fresh_mlp(in_dim, n_classes, hidden=32, seed=1)
    loss_fn = make_loss("cross_entropy")
    xb, yb = x_train[:32], y_train[:32]
    init_loss, _ = forward_backward(model, loss_fn, xb, yb)
    print("initial_batch_loss:", float(init_loss))

    optimizer = make_optimizer(model["params"], lr=0.1, kind="sgd")
    pre_loss = train_step(model, loss_fn, optimizer, xb, yb)
    post_loss, _ = forward_backward(model, loss_fn, xb, yb)
    print("train_step_loss:", float(pre_loss))
    print("after_train_step_loss:", float(post_loss))

    history = train(
        model, loss_fn, optimizer, x_train, y_train,
        epochs=50, batch_size=32, seed=0,
    )
    if isinstance(history, (list, tuple, np.ndarray)) and len(history) > 0:
        print("overfit_loss_start:", float(history[0]))
        print("overfit_loss_end:", float(history[-1]))
    else:
        print("train_history:", history)

    # Regularization demo: wide net + noisy train labels so the unregularized
    # baseline cannot already sit at 100% val accuracy.
    x_noisy, y_noisy = _nonlinear_dataset(192, seed=2, label_noise=0.22)
    x_hold, y_hold = _nonlinear_dataset(64, seed=3, label_noise=0.0)

    def baseline_model_fn():
        return _fresh_mlp(in_dim, n_classes, hidden=56, seed=12345)

    gen_result = improve_generalization(
        baseline_model_fn, x_noisy, y_noisy, x_hold, y_hold, seed=0,
    )
    print("baseline_val_accuracy:", round(float(gen_result["baseline_val_accuracy"]), 4))
    print("improved_val_accuracy:", round(float(gen_result["val_accuracy"]), 4))


if __name__ == "__main__":
    main()

