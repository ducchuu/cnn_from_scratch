import pygame
import sys

# 初始化 pygame
pygame.init()

# 屏幕设置
screen_width = 1000
screen_height = 700
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("神经网络训练过程交互式可视化")

# 设置字体（使用更小的默认字体）
font = pygame.font.SysFont(None, 18)  # 字号调整为18

# 定义各层神经元的位置
input_pos = (150, screen_height // 2)
hidden1_pos = (450, screen_height // 3)
hidden2_pos = (450, screen_height * 2 // 3)
output_pos = (750, screen_height // 2)

# 定义"下一步"按钮的位置和大小
button_rect = pygame.Rect(screen_width // 2 - 50, screen_height - 80, 100, 40)

# 预定义各步的说明文字和对应的详细信息
steps = [
    {
        "desc": "Initial state: Network structure and parameters",
        "details": "Input: x=1.0; Hidden: w1=[1.0, 2.0], b1=[0.5, 0.5]; Output: w2=[1.0, 2.0], b2=0.0",
        "highlight_paths": [],
    },
    {
        "desc": "Forward pass: Calculate hidden layer weighted input",
        "details": "z1 = [1*1.0+0.5, 1*2.0+0.5] = [1.5, 2.5]",
        "highlight_paths": [(input_pos, hidden1_pos), (input_pos, hidden2_pos)],
    },
    {
        "desc": "Forward pass: Apply ReLU activation function",
        "details": "h = ReLU(z1) = [1.5, 2.5]",
        "highlight_paths": [(hidden1_pos, hidden1_pos), (hidden2_pos, hidden2_pos)],
    },
    {
        "desc": "Forward pass: Calculate output layer output",
        "details": "y_pred = h•w2 + b2 = 1.5*1.0 + 2.5*2.0 = 6.5",
        "highlight_paths": [(hidden1_pos, output_pos), (hidden2_pos, output_pos)],
    },
    {
        "desc": "Calculate loss: Use Mean Squared Error (MSE)",
        "details": "L = 0.5*(6.5-7.0)^2 = 0.125",
        "highlight_paths": [(output_pos, output_pos)],
    },
    {
        "desc": "Backward pass: Calculate output layer gradient",
        "details": "dL/dy_pred = y_pred - target = -0.5",
        "highlight_paths": [(output_pos, output_pos)],
    },
    {
        "desc": "Backward pass: Calculate output layer weight gradient",
        "details": "grad_w2 = dL/dy_pred * h = -0.5 * [1.5, 2.5] = [-0.75, -1.25]",
        "highlight_paths": [(hidden1_pos, output_pos), (hidden2_pos, output_pos)],
    },
    {
        "desc": "Backward pass: Gradient propagation to hidden layer",
        "details": "dL/dh = dL/dy_pred * w2 = [-0.5, -1.0]",
        "highlight_paths": [(output_pos, hidden1_pos), (output_pos, hidden2_pos)],
    },
    {
        "desc": "Backward pass: Calculate hidden layer weight gradient",
        "details": "grad_w1 = dL/dz1 * x = [-0.5, -1.0]",
        "highlight_paths": [(input_pos, hidden1_pos), (input_pos, hidden2_pos)],
    },
    {
        "desc": "Parameter update: Update all layer parameters",
        "details": "new_w1=[1.05,2.1], new_b1=[0.55,0.6]; new_w2=[1.075,2.125], new_b2=0.05",
        "highlight_paths": [
            (input_pos, hidden1_pos),
            (input_pos, hidden2_pos),
            (hidden1_pos, output_pos),
            (hidden2_pos, output_pos),
        ],
    },
]

# 修改颜色定义，使用更鲜明的配色
COLORS = {
    "input_value": (0, 120, 255),  # 深蓝色
    "weights": (50, 50, 50),  # 深灰色
    "activation": (34, 139, 34),  # 森林绿
    "gradients": (220, 20, 60),  # 猩红色
    "loss": (255, 140, 0),  # 深橙色
    "updated_params": (148, 0, 211),  # 深紫色
    "neuron_label": (0, 0, 0),  # 黑色
}

# 当前步骤变量（初始为 0）
step = 0


def get_edge_text_positions(start_pos, end_pos):
    # 计算边的中点
    mid_x = (start_pos[0] + end_pos[0]) // 2
    mid_y = (start_pos[1] + end_pos[1]) // 2

    # 计算边的方向向量
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]

    # 计算垂直于边的方向
    length = (dx**2 + dy**2) ** 0.5
    if length != 0:
        normal_x = -dy / length
        normal_y = dx / length
    else:
        normal_x, normal_y = 0, 1

    # 增加层级间距
    offset = 50
    return {
        "top3": (mid_x + normal_x * offset * 3, mid_y + normal_y * offset * 3),
        "top2": (mid_x + normal_x * offset * 2, mid_y + normal_y * offset * 2),
        "top1": (mid_x + normal_x * offset, mid_y + normal_y * offset),
        "mid": (mid_x, mid_y),
        "bottom1": (mid_x - normal_x * offset, mid_y - normal_y * offset),
        "bottom2": (mid_x - normal_x * offset * 2, mid_y - normal_y * offset * 2),
        "bottom3": (mid_x - normal_x * offset * 3, mid_y - normal_y * offset * 3),
    }


def draw_network(screen, step):
    screen.fill((255, 255, 255))
    screen_rect = screen.get_rect()

    # 根据前向传播过程定义相关变量
    # 定义输入层的输入值
    x = 1.0
    # 计算隐藏层的加权输入 z1，根据步骤1的示例计算
    z1 = [x * 1.0 + 0.5, x * 2.0 + 0.5]  # 得到 [1.5, 2.5]
    # 使用 ReLU 激活函数计算隐藏层输出 h
    h = [max(0, val) for val in z1]  # 结果为 [1.5, 2.5]

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
    text = font.render("Neuron1", True, COLORS["neuron_label"])
    screen.blit(text, (hidden1_pos[0] - 40, hidden1_pos[1] - 60))
    text = font.render("Neuron2", True, COLORS["neuron_label"])
    screen.blit(text, (hidden2_pos[0] - 40, hidden2_pos[1] - 60))

    # 在输出层节点旁显示 y_pred 标签
    text = font.render("y_pred", True, (0, 0, 0))
    screen.blit(text, (output_pos[0] - 30, output_pos[1] - 40))

    # 获取边的文本位置
    input_hidden1_pos = get_edge_text_positions(input_pos, hidden1_pos)
    input_hidden2_pos = get_edge_text_positions(input_pos, hidden2_pos)
    hidden1_output_pos = get_edge_text_positions(hidden1_pos, output_pos)
    hidden2_output_pos = get_edge_text_positions(hidden2_pos, output_pos)

    # 统一文本渲染方法
    def render_text(text, color, position, align="topleft"):
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect()
        setattr(text_rect, align, position)
        text_rect.clamp_ip(screen_rect)
        screen.blit(text_surf, text_rect)

    # 权重显示
    weight_positions = {
        "input_hidden1": get_edge_text_positions(input_pos, hidden1_pos)["mid"],
        "input_hidden2": get_edge_text_positions(input_pos, hidden2_pos)["mid"],
        "hidden1_output": get_edge_text_positions(hidden1_pos, output_pos)["mid"],
        "hidden2_output": get_edge_text_positions(hidden2_pos, output_pos)["mid"],
    }

    # 动态布局系统
    info_zones = {
        "input": {"base": (input_pos[0] - 50, input_pos[1] - 100), "offset": 0},
        "hidden1": {"base": (hidden1_pos[0] + 30, hidden1_pos[1] - 50), "offset": 0},
        "hidden2": {"base": (hidden2_pos[0] + 30, hidden2_pos[1] - 50), "offset": 0},
        "output": {"base": (output_pos[0] + 30, output_pos[1] - 50), "offset": 0},
        "edges": {"base": (screen_width // 2, 100), "offset": 0},
    }

    # 步骤相关显示
    if step >= 1:
        render_text(
            f"z1: {z1[0]:.1f}",
            COLORS["activation"],
            (hidden1_pos[0] + 30, hidden1_pos[1] - 30),
        )
        render_text(
            f"z1: {z1[1]:.1f}",
            COLORS["activation"],
            (hidden2_pos[0] + 30, hidden2_pos[1] - 30),
        )

    if step >= 2:
        render_text(
            f"h: {h[0]:.1f}",
            COLORS["activation"],
            (hidden1_pos[0] + 30, hidden1_pos[1] + 20),
        )
        render_text(
            f"h: {h[1]:.1f}",
            COLORS["activation"],
            (hidden2_pos[0] + 30, hidden2_pos[1] + 20),
        )

    # 梯度信息显示（使用分层布局）
    grad_layers = {
        6: ("top1", "grad_w2 = [-0.75, -1.25]"),
        7: ("bottom1", "dL/dh = [-0.5, -1.0]"),
        8: ("top2", "grad_w1 = [-0.5, -1.0]"),
    }

    for s, (layer, text) in grad_layers.items():
        if step >= s:
            pos1 = get_edge_text_positions(hidden1_pos, output_pos)[layer]
            pos2 = get_edge_text_positions(hidden2_pos, output_pos)[layer]
            render_text(text, COLORS["gradients"], pos1, "center")
            render_text(text, COLORS["gradients"], pos2, "center")

    # 其他信息显示
    if step >= 3:
        render_text(
            "y_pred: 6.5", COLORS["gradients"], (output_pos[0] + 40, output_pos[1] - 30)
        )

    if step >= 4:
        render_text(
            "Loss: 0.125", COLORS["loss"], (output_pos[0] + 40, output_pos[1] + 10)
        )

    if step >= 5:
        render_text(
            "dL/dy: -0.5", COLORS["gradients"], (output_pos[0] + 40, output_pos[1] + 50)
        )

    # 在屏幕顶部显示当前步骤的说明
    current_step = steps[step]
    step_text = font.render(current_step["desc"], True, (0, 0, 0))
    screen.blit(step_text, (50, 40))
    detail_text = font.render(current_step["details"], True, (0, 0, 0))
    screen.blit(detail_text, (50, 70))

    # 调整"Next Step"按钮样式
    pygame.draw.rect(screen, (200, 200, 200), button_rect)
    button_text = font.render("Next Step", True, (0, 0, 0))
    button_text_rect = button_text.get_rect(center=button_rect.center)
    screen.blit(button_text, button_text_rect)

    # 绘制高亮路径
    current_step = steps[step]
    for path in current_step["highlight_paths"]:
        pygame.draw.line(screen, (255, 255, 0), path[0], path[1], 4)  # 黄色高亮


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
