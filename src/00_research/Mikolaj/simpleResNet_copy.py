import torch  # Import PyTorch main library for tensor computation and automatic differentiation
import torch.nn as nn  # Import neural network module containing layers and activation functions
import torch.optim as optim  # Import optimizer module providing SGD, Adam, etc.
import torchvision  # Import computer vision library containing common datasets and models
import torchvision.transforms as transforms  # Import image transformation tools for data preprocessing
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from datetime import datetime

matplotlib.use("Agg")

# Define basic ResidualBlock
# This is the core component of ResNet, implementing residual learning: H(x) = F(x) + x
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Initialize residual block
        Parameters:
            in_channels: Number of input feature map channels
            out_channels: Number of output feature map channels
            stride: Convolution stride, used for downsampling (when stride=2)
        """
        super(ResidualBlock, self).__init__()
        # First convolution layer: may change channels and feature map size
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,  # 3x3 convolution kernel
            stride=stride,  # Stride controls feature map size change
            padding=1,  # Padding to maintain feature map size
            bias=False,  # No bias as BatchNorm follows
        )
        self.bn1 = nn.BatchNorm2d(
            out_channels
        )  # Batch normalization to stabilize training

        # Second convolution layer: maintains channels and feature map size
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)  # Second batch normalization layer
        self.relu = nn.ReLU(inplace=True)  # ReLU activation, inplace=True saves memory

        # Shortcut connection
        # Default is identity mapping (no operation)
        self.shortcut = nn.Sequential()

        # When input and output dimensions don't match, use projection shortcut
        if stride != 1 or in_channels != out_channels:
            # Use 1x1 convolution to adjust dimensions, implementing Ws projection matrix from paper
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

        # Add name for parameter tracking
        self.name = f"ResBlock_{in_channels}_{out_channels}_{stride}"

    def forward(self, x):
        """
        Forward propagation function
        Implements residual learning: H(x) = F(x) + x
        """
        identity = x  # Save input for shortcut connection

        # Calculate residual function F(x)
        out = self.conv1(x)  # First convolution
        out = self.bn1(out)  # Batch normalization
        out = self.relu(out)  # ReLU activation

        out = self.conv2(out)  # Second convolution
        out = self.bn2(out)  # Batch normalization

        # Implement residual connection: F(x) + x
        # If dimensions don't match, use shortcut projection
        out += self.shortcut(identity)

        out = self.relu(out)  # Final ReLU activation

        return out


# Define simplified ResNet
# This is a modular ResNet implementation that can create networks of different depths by adjusting num_blocks
class SimpleResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        """
        Initialize ResNet
        Parameters:
            num_blocks: List specifying number of residual blocks in each stage
            num_classes: Number of classification classes
        """
        super(SimpleResNet, self).__init__()
        self.in_channels = 64  # Initial number of channels

        # Initial convolution layer to convert input image to feature map
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)  # Batch normalization
        self.relu = nn.ReLU(inplace=True)  # ReLU activation

        # Create 4 stages of residual layers, each stage doubles channels and halves feature map size
        self.layer1 = self._make_layer(
            64, num_blocks[0], stride=1
        )  # Output: [B, 64, H, W]
        self.layer2 = self._make_layer(
            128, num_blocks[1], stride=2
        )  # Output: [B, 128, H/2, W/2]
        self.layer3 = self._make_layer(
            256, num_blocks[2], stride=2
        )  # Output: [B, 256, H/4, W/4]
        self.layer4 = self._make_layer(
            512, num_blocks[3], stride=2
        )  # Output: [B, 512, H/8, W/8]

        # Global average pooling to convert feature map to feature vector
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Output: [B, 512, 1, 1]

        # Fully connected layer for final classification
        self.fc = nn.Linear(512, num_classes)  # Output: [B, num_classes]

        # Initialize parameter statistics dictionary
        self.param_stats = {}
        self._init_param_stats()

    def _make_layer(self, out_channels, num_blocks, stride):
        """
        Create layer containing multiple residual blocks
        Parameters:
            out_channels: Number of output channels
            num_blocks: Number of residual blocks
            stride: Stride of first residual block, used for downsampling
        """
        layers = []

        # First residual block may change dimensions (channels and feature map size)
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))

        # Update current number of channels
        self.in_channels = out_channels

        # Add remaining residual blocks, maintaining dimensions
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        # Combine all residual blocks into a Sequential module
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward propagation function
        Defines complete path of data through network
        """
        # Initial processing
        x = self.conv1(x)  # Initial convolution
        x = self.bn1(x)  # Batch normalization
        x = self.relu(x)  # ReLU activation

        # Through 4 residual layers
        x = self.layer1(x)  # First layer residual block group
        x = self.layer2(x)  # Second layer residual block group
        x = self.layer3(x)  # Third layer residual block group
        x = self.layer4(x)  # Fourth layer residual block group

        # Global average pooling
        x = self.avgpool(x)

        # Flatten feature map to vector
        x = x.view(x.size(0), -1)

        # Fully connected layer classification
        x = self.fc(x)

        return x

    def _init_param_stats(self):
        """Initialize parameter statistics dictionary for tracking layer parameter changes"""
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.param_stats[name] = {
                    "mean": [],
                    "std": [],
                    "min": [],
                    "max": [],
                    "grad_mean": [],
                    "grad_std": [],
                    "grad_min": [],
                    "grad_max": [],
                }

    def update_param_stats(self):
        """Update parameter statistics information"""
        for name, param in self.named_parameters():
            if param.requires_grad:
                # Parameter statistics
                self.param_stats[name]["mean"].append(param.data.mean().item())
                self.param_stats[name]["std"].append(param.data.std().item())
                self.param_stats[name]["min"].append(param.data.min().item())
                self.param_stats[name]["max"].append(param.data.max().item())

                # Gradient statistics (if gradient exists)
                if param.grad is not None:
                    self.param_stats[name]["grad_mean"].append(param.grad.mean().item())
                    self.param_stats[name]["grad_std"].append(param.grad.std().item())
                    self.param_stats[name]["grad_min"].append(param.grad.min().item())
                    self.param_stats[name]["grad_max"].append(param.grad.max().item())
                else:
                    self.param_stats[name]["grad_mean"].append(0)
                    self.param_stats[name]["grad_std"].append(0)
                    self.param_stats[name]["grad_min"].append(0)
                    self.param_stats[name]["grad_max"].append(0)


# Create ResNet-18 model
# ResNet-18 contains 4 stages, each with 2 residual blocks, totaling 8 residual blocks (each containing 2 layers, totaling 16 layers)
# Plus initial convolution and final fully connected layer, totaling 18 layers
def resnet18():
    """
    Create ResNet-18 model instance
    [2,2,2,2] indicates 4 stages each with 2 residual blocks
    """
    return SimpleResNet([2, 2, 2, 2])


# Load CIFAR-10 dataset
# CIFAR-10 contains 32x32 color images in 10 classes
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        ),  # Normalize: (pixel value - mean) / std
    ]
)

# Download and load training set
trainset = torchvision.datasets.CIFAR10(
    root="./data",  # Data storage path
    train=True,  # Use training set
    download=True,  # Download if data doesn't exist
    transform=transform,  # Apply defined transformations
)

# Create data loader
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=128,  # 128 images per batch
    shuffle=True,  # Randomly shuffle data
    num_workers=2,  # Use 2 subprocesses for data loading
)

# Create validation set
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2
)

# Class names
classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# Initialize model, loss function, and optimizer
# Check if MPS (Apple M1 GPU acceleration) is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
model = resnet18().to(device)  # Create model and move to device

# Use cross-entropy loss function (suitable for classification)
criterion = nn.CrossEntropyLoss()

# Use SGD optimizer with momentum
optimizer = optim.SGD(
    model.parameters(),  # Optimize all model parameters
    lr=0.01,  # Learning rate
    momentum=0.9,  # Momentum factor for faster convergence
    weight_decay=5e-4,  # L2 regularization coefficient to prevent overfitting
)

# Create log directory for storing results
log_dir = f"logs/resnet18_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(f"{log_dir}/plots", exist_ok=True)

# Learning rate scheduler for adaptive learning rate adjustment
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", patience=3, factor=0.1
)


# Progress bar function (alternative to tqdm)
def progress_bar(current, total, msg=None):
    """Simple progress bar implementation"""
    bar_length = 40
    filled_length = int(bar_length * current / total)
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    print(f'\r[{bar}] {current}/{total} {msg or ""}', end="")
    if current == total:
        print()


# Training model function
def train(epochs=10):
    """
    Train model
    Parameters:
        epochs: Number of training rounds
    """
    # Initialize training log
    train_log = {
        "epoch": [],
        "batch": [],
        "loss": [],
        "accuracy": [],
        "lr": [],
        "time_per_batch": [],
        "time_per_epoch": [],
    }

    # Initialize validation log
    val_log = {
        "epoch": [],
        "loss": [],
        "accuracy": [],
    }

    # Print model structure and parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 80)
    print(f"Model structure:\n{model}")
    print("-" * 80)
    print(f"Total parameter count: {total_params:,}")
    print(f"Trainable parameter count: {trainable_params:,}")
    print("=" * 80)

    # Record initial parameter statistics
    print("Recording initial parameter statistics...")
    model.update_param_stats()

    # Record training start time
    total_start_time = time.time()

    for epoch in range(epochs):
        print(f"\nStarting Epoch {epoch+1}/{epochs}")
        epoch_start_time = time.time()

        # Training mode
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        batch_times = []

        # Training loop
        for i, data in enumerate(trainloader):
            batch_start_time = time.time()

            # Get input and labels
            inputs, labels = data[0].to(device), data[1].to(device)

            # Print batch information
            if i == 0:
                print(f"\nBatch size: {inputs.size(0)}")
                print(f"Input shape: {inputs.shape}")
                print(f"Label shape: {labels.shape}")
                print(f"Label example: {labels[:5].cpu().numpy()}")

            # Zero gradients (prevent gradient accumulation)
            optimizer.zero_grad()

            # Forward propagation
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward propagation
            loss.backward()

            # Record gradient statistics before parameter update
            if i % 50 == 0:
                model.update_param_stats()

            # Parameter update
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total

            # Record loss
            running_loss += loss.item()

            # Calculate batch time
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)

            # Display progress
            msg = f"loss: {loss.item():.4f} | acc: {accuracy:.2f}% | time: {batch_time:.3f}s"
            progress_bar(i + 1, len(trainloader), msg)

            # Record training log
            train_log["epoch"].append(epoch + 1)
            train_log["batch"].append(i + 1)
            train_log["loss"].append(loss.item())
            train_log["accuracy"].append(accuracy)
            train_log["lr"].append(optimizer.param_groups[0]["lr"])
            train_log["time_per_batch"].append(batch_time)

            # Print detailed statistics every 100 batches
            if i % 100 == 99:
                avg_loss = running_loss / 100
                print(f"\n[Epoch {epoch+1}, Batch {i+1}]")
                print(f"  Loss: {avg_loss:.4f}")
                print(f"  Accuracy: {accuracy:.2f}%")
                print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
                print(f"  Batch Time: {sum(batch_times[-100:]) / 100:.4f}s")

                # Plot loss curve for last 100 batches
                plt.figure(figsize=(10, 5))
                plt.plot(
                    range(max(0, len(train_log["loss"]) - 100), len(train_log["loss"])),
                    train_log["loss"][-100:],
                )
                plt.title(f"Training Loss (Epoch {epoch+1}, Batch {i+1})")
                plt.xlabel("Batch")
                plt.ylabel("Loss")
                plt.savefig(f"{log_dir}/plots/loss_epoch{epoch+1}_batch{i+1}.png")
                plt.close()

                running_loss = 0.0

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        train_log["time_per_epoch"].append(epoch_time)

        # Print epoch statistics
        print(f"\nEpoch {epoch+1} completed:")
        print(
            f"   Average loss: {sum(train_log['loss'][-len(trainloader):]) / len(trainloader):.4f}"
        )
        print(f"   Final accuracy: {accuracy:.2f}%")
        print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Epoch time: {epoch_time:.2f}s")

        # Validation mode
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        print("\nStarting validation...")
        with torch.no_grad():
            for i, data in enumerate(testloader):
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                # Display validation progress
                progress_bar(i + 1, len(testloader), f"loss: {loss.item():.4f}")

        # Calculate validation set statistics
        avg_val_loss = val_loss / len(testloader)
        val_accuracy = 100 * val_correct / val_total

        # Record validation log
        val_log["epoch"].append(epoch + 1)
        val_log["loss"].append(avg_val_loss)
        val_log["accuracy"].append(val_accuracy)

        # Print validation results
        print(f"\nValidation results:")
        print(f"   Validation loss: {avg_val_loss:.4f}")
        print(f"   Validation accuracy: {val_accuracy:.2f}%")

        # Learning rate scheduler
        scheduler.step(avg_val_loss)

        # Print accuracy for each class
        class_correct = list(0.0 for i in range(10))
        class_total = list(0.0 for i in range(10))

        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(labels.size(0)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        print("\nClass accuracies:")
        for i in range(10):
            print(f"  {classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}%")

        # Save model
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_log["loss"][-1],
                "val_loss": avg_val_loss,
                "train_acc": accuracy,
                "val_acc": val_accuracy,
            },
            f"{log_dir}/model_epoch{epoch+1}.pth",
        )

        # Plot parameter statistics
        if (epoch + 1) % 2 == 0 or epoch == epochs - 1:
            print("\nPlotting parameter statistics...")
            # Select some key layers for visualization
            key_layers = [
                "conv1.weight",
                "layer1.0.conv1.weight",
                "layer2.0.conv1.weight",
                "layer3.0.conv1.weight",
                "layer4.0.conv1.weight",
                "fc.weight",
            ]

            for layer_name in key_layers:
                if layer_name in model.param_stats:
                    plt.figure(figsize=(15, 10))

                    # Plot parameter mean and standard deviation
                    plt.subplot(2, 2, 1)
                    plt.plot(model.param_stats[layer_name]["mean"], label="Mean")
                    plt.plot(model.param_stats[layer_name]["std"], label="Std")
                    plt.title(f"{layer_name} - Parameters")
                    plt.legend()

                    # Plot parameter maximum and minimum values
                    plt.subplot(2, 2, 2)
                    plt.plot(model.param_stats[layer_name]["max"], label="Max")
                    plt.plot(model.param_stats[layer_name]["min"], label="Min")
                    plt.title(f"{layer_name} - Parameter Range")
                    plt.legend()

                    # Plot gradient mean and standard deviation
                    plt.subplot(2, 2, 3)
                    plt.plot(
                        model.param_stats[layer_name]["grad_mean"], label="Grad Mean"
                    )
                    plt.plot(
                        model.param_stats[layer_name]["grad_std"], label="Grad Std"
                    )
                    plt.title(f"{layer_name} - Gradients")
                    plt.legend()

                    # Plot gradient maximum and minimum values
                    plt.subplot(2, 2, 4)
                    plt.plot(
                        model.param_stats[layer_name]["grad_max"], label="Grad Max"
                    )
                    plt.plot(
                        model.param_stats[layer_name]["grad_min"], label="Grad Min"
                    )
                    plt.title(f"{layer_name} - Gradient Range")
                    plt.legend()

                    plt.tight_layout()
                    plt.savefig(
                        f"{log_dir}/plots/{layer_name.replace('.', '_')}_epoch{epoch+1}.png"
                    )
                    plt.close()

    # Calculate total training time
    total_time = time.time() - total_start_time

    # Print summary
    print("\n" + "=" * 80)
    print(f"Training completed! Total time: {total_time:.2f}s ({total_time/60:.2f}min)")
    print(f"Final training accuracy: {train_log['accuracy'][-1]:.2f}%")
    print(f"Final validation accuracy: {val_log['accuracy'][-1]:.2f}%")
    print("=" * 80)

    # Plot training and validation loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(
        range(1, epochs + 1),
        [
            sum(train_log["loss"][i * len(trainloader) : (i + 1) * len(trainloader)])
            / len(trainloader)
            for i in range(epochs)
        ],
        label="Train",
    )
    plt.plot(val_log["epoch"], val_log["loss"], label="Validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(
        range(1, epochs + 1),
        [train_log["accuracy"][i * len(trainloader) - 1] for i in range(1, epochs + 1)],
        label="Train",
    )
    plt.plot(val_log["epoch"], val_log["accuracy"], label="Validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{log_dir}/training_summary.png")
    plt.close()

    # Save training log
    import json

    with open(f"{log_dir}/train_log.json", "w") as f:
        json.dump(train_log, f)
    with open(f"{log_dir}/val_log.json", "w") as f:
        json.dump(val_log, f)

    print(f"\nTraining log and plots saved to: {log_dir}")

    return train_log, val_log


# Run training
# Training only executed when this script is run directly
if __name__ == "__main__":
    train(10)  # Train for 10 epochs
