import numpy as np


class MLP:
    """Fully connected net, ReLU on the hidden layers. Forward and backward written by hand."""

    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        # Adam state
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    def forward(self, x):
        self.cache = [x]  # input of every layer, needed for the backward pass
        last = len(self.W) - 1
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if i < last:
                x = np.maximum(0, x)
            self.cache.append(x)
        return x

    def backward(self, dout):
        """dout = gradient of the loss w.r.t. the output. Returns gradients for W and b."""
        gW = [None] * len(self.W)
        gb = [None] * len(self.W)
        d = dout
        for i in reversed(range(len(self.W))):
            gW[i] = self.cache[i].T @ d
            gb[i] = d.sum(axis=0)
            if i > 0:
                d = d @ self.W[i].T
                d = d * (self.cache[i] > 0)  # relu: gradient only passes where the unit was active
        return gW, gb

    def adam_step(self, gW, gb, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for i in range(len(self.W)):
            for p, g, m, v in ((self.W, gW, self.mW, self.vW), (self.b, gb, self.mb, self.vb)):
                m[i] = b1 * m[i] + (1 - b1) * g[i]
                v[i] = b2 * v[i] + (1 - b2) * g[i] ** 2
                m_hat = m[i] / (1 - b1 ** self.t)
                v_hat = v[i] / (1 - b2 ** self.t)
                p[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def copy_from(self, other):
        self.W = [w.copy() for w in other.W]
        self.b = [b.copy() for b in other.b]

    def save(self, path):
        np.savez(path, *self.W, *self.b)

    def load(self, path):
        data = np.load(path)
        n = len(self.W)
        self.W = [data[f"arr_{i}"] for i in range(n)]
        self.b = [data[f"arr_{i}"] for i in range(n, 2 * n)]


if __name__ == "__main__":
    # gradient check: compare the hand-written backward pass with numerical gradients
    rng = np.random.default_rng(1)
    net = MLP([5, 8, 6, 3], seed=1)
    net.W = [w.astype(np.float64) for w in net.W]
    x = rng.normal(size=(4, 5))
    y = rng.normal(size=(4, 3))

    def loss():
        return 0.5 * np.sum((net.forward(x) - y) ** 2) / len(x)

    out = net.forward(x)
    gW, gb = net.backward((out - y) / len(x))

    worst = 0.0
    h = 1e-6
    for params, grads, name in ((net.W, gW, "W"), (net.b, gb, "b")):
        for i, (p, g) in enumerate(zip(params, grads)):
            for idx in np.ndindex(p.shape):
                old = p[idx]
                p[idx] = old + h
                up = loss()
                p[idx] = old - h
                down = loss()
                p[idx] = old
                num = (up - down) / (2 * h)
                rel = abs(num - g[idx]) / max(1e-8, abs(num) + abs(g[idx]))
                worst = max(worst, rel)
    msg = f"largest relative difference between backprop and numerical gradient: {worst:.2e}\n"
    msg += "OK" if worst < 1e-5 else "something is wrong"
    print(msg)
    with open("results/gradcheck.txt", "w", encoding="utf-8") as f:
        f.write(msg + "\n")