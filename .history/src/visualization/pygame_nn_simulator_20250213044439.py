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
        "desc": "初始状态：展示网络结构和参数",
        "details": "输入: x=1.0; 隐藏层: w1=[1.0, 2.0], b1=[0.5, 0.5]; 输出层: w2=[1.0, 2.0], b2=0.0",
    },
    {
        "desc": "前向传播：计算隐藏层加权输入 z = x*w + b",
        "details": "z1 = [1*1.0+0.5, 1*2.0+0.5] = [1.5, 2.5]",
    },
    {"desc": "前向传播：应用ReLU激活函数", "details": "h = ReLU(z1) = [1.5, 2.5]"},
    {
        "desc": "前向传播：计算输出层输出",
        "details": "y_pred = h•w2 + b2 = 1.5*1.0 + 2.5*2.0 = 6.5",
    },
    {"desc": "计算损失：采用均方误差 (MSE)", "details": "L = 0.5*(6.5-7.0)^2 = 0.125"},
    {
        "desc": "反向传播：计算输出层梯度",
        "details": "dL/dy_pred = y_pred - target = -0.5",
    },
    {
        "desc": "反向传播：计算输出层权重梯度",
        "details": "grad_w2 = dL/dy_pred * h = -0.5 * [1.5, 2.5] = [-0.75, -1.25]",
    },
    {
        "desc": "反向传播：梯度传递到隐藏层",
        "details": "dL/dh = dL/dy_pred * w2 = [-0.5, -1.0]",
    },
    {
        "desc": "反向传播：计算隐藏层权重梯度",
        "details": "grad_w1 = dL/dz1 * x = [-0.5, -1.0]",
    },
    {
        "desc": "参数更新：更新各层参数",
        "details": "new_w1=[1.05, 2.1], new_b1=[0.55, 0.6]; new_w2=[1.075, 2.125], new_b2=0.05",
    },
]

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

    # 根据当前 step 显示相应的结果：
    if step >= 1:
        # 显示隐藏层加权输入 z1（用红色显示，显示在隐藏节点右侧）
        text = font.render("z=1.5", True, (255, 0, 0))
        screen.blit(text, (hidden1_pos[0] + 25, hidden1_pos[1] - 10))
        text = font.render("z=2.5", True, (255, 0, 0))
        screen.blit(text, (hidden2_pos[0] + 25, hidden2_pos[1] - 10))

    if step >= 2:
        # 显示经过 ReLU 激活后的隐藏层输出 h
        text = font.render("h=1.5", True, (255, 0, 0))
        screen.blit(text, (hidden1_pos[0] + 25, hidden1_pos[1] + 10))
        text = font.render("h=2.5", True, (255, 0, 0))
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
        text = font.render("参数更新完成", True, (255, 0, 0))
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

    # 画出"下一步"按钮
    pygame.draw.rect(screen, (200, 200, 200), button_rect)
    button_text = font.render("下一步", True, (0, 0, 0))
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
