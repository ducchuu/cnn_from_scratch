import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def simple_training(num_epochs=100):
    # 模拟一个简单的训练过程： y = w * x, 目标值 y_true = 2.0, 实际的 true_w = 2.0
    w = 0.0  # 初始化权重
    lr = 0.1  # 学习率
    weights = []
    losses = []
    x = 1.0  # 固定输入
    y_true = 2.0  # 目标输出

    for epoch in range(num_epochs):
        y_pred = w * x
        loss = 0.5 * (y_pred - y_true) ** 2
        gradient = (y_pred - y_true) * x
        w = w - lr * gradient  # 参数更新

        weights.append(w)
        losses.append(loss)

    return weights, losses


weights, losses = simple_training(num_epochs=100)

# 设置图形
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
ax1.set_xlim(0, len(weights))
ax1.set_ylim(min(weights) - 0.5, max(weights) + 0.5)
(line_w,) = ax1.plot([], [], lw=2)
ax1.set_title("权重变化")

ax2.set_xlim(0, len(losses))
ax2.set_ylim(0, max(losses) * 1.1)
(line_loss,) = ax2.plot([], [], lw=2)
ax2.set_title("损失变化")


def init():
    line_w.set_data([], [])
    line_loss.set_data([], [])
    return line_w, line_loss


def update(frame):
    xdata = list(range(frame + 1))
    ydata_w = weights[: frame + 1]
    ydata_loss = losses[: frame + 1]
    line_w.set_data(xdata, ydata_w)
    line_loss.set_data(xdata, ydata_loss)
    return line_w, line_loss


ani = FuncAnimation(
    fig, update, frames=len(weights), init_func=init, blit=True, interval=100
)
plt.tight_layout()
plt.show()
