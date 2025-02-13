import pygame
import sys

# 初始化 pygame
pygame.init()

# 屏幕设置
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("神经网络训练过程交互式可视化")

# 设置字体
font = pygame.font.SysFont(None, 24)

# 定义各层神经元的位置
input_pos = (100, screen_height // 2)
hidden1_pos = (400, screen_height // 3)
hidden2_pos = (400, screen_height * 2 // 3)
output_pos = (700, screen_height // 2)

# 定义"下一步"按钮的位置和大小
button_rect = pygame.Rect(screen_width // 2 - 50, screen_height - 50, 100, 40)

# 预定义各步的说明文字和对应的详细信息
steps = [
    {
        "desc": "Initial state: Network structure and parameters",
        "details": "Input: x=1.0; Hidden: w1=[1.0, 2.0], b1=[0.5, 0.5]; Output: w2=[1.0, 2.0], b2=0.0",
        "highlight_paths": [],  # 初始状态不高亮任何路径
    },
    {
        "desc": "Forward pass: Calculate hidden layer weighted input",
        "details": "z1 = [1*1.0+0.5, 1*2.0+0.5] = [1.5, 2.5]",
        "highlight_paths": [
            (input_pos, hidden1_pos),
            (input_pos, hidden2_pos),
        ],  # 高亮输入到隐藏层的路径
    },
    {
        "desc": "Forward pass: Apply ReLU activation function",
        "details": "h = ReLU(z1) = [1.5, 2.5]",
    },
    {
        "desc": "Forward pass: Calculate output layer output",
        "details": "y_pred = h•w2 + b2 = 1.5*1.0 + 2.5*2.0 = 6.5",
    },
    {
        "desc": "Calculate loss: Use Mean Squared Error (MSE)",
        "details": "L = 0.5*(6.5-7.0)^2 = 0.125",
    },
    {
        "desc": "Backward pass: Calculate output layer gradient",
        "details": "dL/dy_pred = y_pred - target = -0.5",
    },
    {
        "desc": "Backward pass: Calculate output layer weight gradient",
        "details": "grad_w2 = dL/dy_pred * h = -0.5 * [1.5, 2.5] = [-0.75, -1.25]",
    },
    {
        "desc": "Backward pass: Gradient propagation to hidden layer",
        "details": "dL/dh = dL/dy_pred * w2 = [-0.5, -1.0]",
    },
    {
        "desc": "Backward pass: Calculate hidden layer weight gradient",
        "details": "grad_w1 = dL/dz1 * x = [-0.5, -1.0]",
    },
    {
        "desc": "Parameter update: Update all layer parameters",
        "details": "new_w1=[1.05, 2.1], new_b1=[0.55, 0.6]; new_w2=[1.075, 2.125], new_b2=0.05",
    },
]

# 定义不同类型数据的颜色
COLORS = {
    "input_value": (0, 0, 255),  # 蓝色
    "weights": (100, 100, 100),  # 灰色
    "activation": (0, 200, 0),  # 绿色
    "gradients": (255, 0, 0),  # 红色
    "loss": (255, 100, 0),  # 橙色
    "updated_params": (128, 0, 128),  # 紫色
}

# 当前步骤变量（初始为 0）
step = 0


def draw_network(screen, step):
    # 清空屏幕（填充白色）
    screen.fill((255, 255, 255))

    # 画出各层神经元：用圆圈表示
    # 输入层（蓝色）
    pygame.draw.circle(screen, (0, 0, 255), input_pos, 20)
    # 隐藏层（绿色）
    pygame.draw.circle(screen, (0, 255, 0), hidden1_pos, 20)
    pygame.draw.circle(screen, (0, 255, 0), hidden2_pos, 20)
    # 输出层（红色）
    pygame.draw.circle(screen, (255, 0, 0), output_pos, 20)

    # 画出各层之间的连线
    pygame.draw.line(screen, (0, 0, 0), input_pos, hidden1_pos, 2)
    pygame.draw.line(screen, (0, 0, 0), input_pos, hidden2_pos, 2)
    pygame.draw.line(screen, (0, 0, 0), hidden1_pos, output_pos, 2)
    pygame.draw.line(screen, (0, 0, 0), hidden2_pos, output_pos, 2)

    # 在输入层节点旁显示 x=1.0
    text = font.render("x=1.0", True, (0, 0, 0))
    screen.blit(text, (input_pos[0] - 20, input_pos[1] - 40))

    # 在隐藏层节点旁显示神经元标签
    text = font.render("Neuron1", True, (0, 0, 0))
    screen.blit(text, (hidden1_pos[0] - 40, hidden1_pos[1] - 40))
    text = font.render("Neuron2", True, (0, 0, 0))
    screen.blit(text, (hidden2_pos[0] - 40, hidden2_pos[1] - 40))

    # 在输出层节点旁显示 y_pred 标签
    text = font.render("y_pred", True, (0, 0, 0))
    screen.blit(text, (output_pos[0] - 30, output_pos[1] - 40))

    # 绘制高亮路径
    current_step = steps[step]
    for path in current_step["highlight_paths"]:
        pygame.draw.line(screen, (255, 255, 0), path[0], path[1], 4)  # 黄色高亮

    # 在各个位置显示数值，使用不同颜色
    if step >= 1:
        # 隐藏层加权输入（绿色）
        text = font.render(f"z={z1[0]:.1f}", True, COLORS["activation"])
        screen.blit(text, (hidden1_pos[0] + 25, hidden1_pos[1] - 30))
        text = font.render(f"z={z1[1]:.1f}", True, COLORS["activation"])
        screen.blit(text, (hidden2_pos[0] + 25, hidden2_pos[1] - 30))

    if step >= 2:
        # 激活值（绿色）
        text = font.render(f"h={h[0]:.1f}", True, COLORS["activation"])
        screen.blit(text, (hidden1_pos[0] + 25, hidden1_pos[1] + 10))
        text = font.render(f"h={h[1]:.1f}", True, COLORS["activation"])
        screen.blit(text, (hidden2_pos[0] + 25, hidden2_pos[1] + 10))

    if step >= 3:
        # 显示输出层计算结果 y_pred
        text = font.render("y_pred=6.5", True, (255, 0, 0))
        screen.blit(text, (output_pos[0] - 70, output_pos[1] + 30))

    if step >= 4:
        # 显示损失 L
        text = font.render("L = 0.125", True, (255, 0, 0))
        screen.blit(text, (output_pos[0] - 70, output_pos[1] + 60))

    if step >= 5:
        # 显示输出层梯度 dL/dy_pred（在输出节点上方显示）
        text = font.render("dL/dy_pred = -0.5", True, (255, 0, 0))
        screen.blit(text, (output_pos[0] - 70, output_pos[1] - 60))

    if step >= 6:
        # 显示输出层权重梯度 grad_w2（显示在连接隐藏层到输出层之间）
        text = font.render("grad_w2 = [-0.75, -1.25]", True, (255, 0, 0))
        screen.blit(text, (output_pos[0] - 120, output_pos[1] + 80))

    if step >= 7:
        # 显示传递到隐藏层的梯度 dL/dh（在隐藏层节点旁上方显示）
        text = font.render("dL/dh = [-0.5, -1.0]", True, (255, 0, 0))
        screen.blit(text, (hidden1_pos[0] - 80, hidden1_pos[1] - 60))
        screen.blit(text, (hidden2_pos[0] - 80, hidden2_pos[1] - 60))

    if step >= 8:
        # 显示隐藏层权重梯度 grad_w1（在输入层与隐藏层之间显示）
        text = font.render("grad_w1 = [-0.5, -1.0]", True, (255, 0, 0))
        screen.blit(text, (input_pos[0] + 30, input_pos[1] - 30))

    if step >= 9:
        # 显示参数更新后的结果（在屏幕底部显示更新后的权重与偏置）
        text = font.render("Parameter update completed", True, (255, 0, 0))
        screen.blit(text, (50, screen_height - 80))
        text = font.render("new_w1=[1.05,2.1], new_b1=[0.55,0.6]", True, (255, 0, 0))
        screen.blit(text, (hidden1_pos[0] - 100, hidden1_pos[1] + 40))
        text = font.render("new_w2=[1.075,2.125], new_b2=0.05", True, (255, 0, 0))
        screen.blit(text, (output_pos[0] - 140, output_pos[1] + 80))

    # 在屏幕顶部显示当前步骤的说明
    current_step = steps[step]
    step_text = font.render(current_step["desc"], True, (0, 0, 0))
    screen.blit(step_text, (50, 20))
    detail_text = font.render(current_step["details"], True, (0, 0, 0))
    screen.blit(detail_text, (50, 50))

    # 调整"Next Step"按钮样式
    pygame.draw.rect(screen, (200, 200, 200), button_rect)
    button_text = font.render("Next Step", True, (0, 0, 0))
    button_text_rect = button_text.get_rect(center=button_rect.center)
    screen.blit(button_text, button_text_rect)


# 主循环
clock = pygame.time.Clock()
running = True

while running:
    clock.tick(30)  # 控制帧率 30 FPS

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break

        # 点击鼠标按钮（仅响应按钮范围内的点击）
        if event.type == pygame.MOUSEBUTTONDOWN:
            if button_rect.collidepoint(event.pos):
                if step < len(steps) - 1:
                    step += 1
                else:
                    # 如果已完成所有步骤，则重置回初始状态
                    step = 0

    # 绘制网络和步骤信息
    draw_network(screen, step)

    # 更新屏幕
    pygame.display.flip()

pygame.quit()
sys.exit()
