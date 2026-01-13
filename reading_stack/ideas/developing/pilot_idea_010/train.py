"""
Training functions for IDEA-010 pilot experiment.

Provides training loops for:
- Hard label training (standard cross-entropy)
- Soft label training (knowledge distillation from teacher)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional


def train_hard_labels(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 50,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    device: str = 'cuda',
    verbose: bool = True,
) -> Dict:
    """
    Train model with hard labels (standard cross-entropy).

    Args:
        model: Neural network to train
        train_loader: DataLoader for training data
        epochs: Number of training epochs
        lr: Learning rate
        momentum: SGD momentum
        weight_decay: L2 regularization
        device: Device to train on
        verbose: Show progress bar

    Returns:
        history: Dict with 'loss' (per-epoch avg loss), 'accuracy' (per-epoch)
    """
    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )

    # Learning rate scheduler (cosine annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        'loss': [],
        'accuracy': [],
    }

    epoch_iter = tqdm(range(epochs), desc='Training') if verbose else range(epochs)

    for epoch in epoch_iter:
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

        scheduler.step()

        avg_loss = total_loss / total
        accuracy = correct / total

        history['loss'].append(avg_loss)
        history['accuracy'].append(accuracy)

        if verbose:
            epoch_iter.set_postfix(loss=f'{avg_loss:.4f}', acc=f'{accuracy:.4f}')

    return history


def train_soft_labels(
    model: nn.Module,
    train_loader: DataLoader,
    teacher: nn.Module,
    epochs: int = 50,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    temperature: float = 4.0,
    device: str = 'cuda',
    verbose: bool = True,
) -> Dict:
    """
    Train model with soft labels from teacher (knowledge distillation).

    Uses KL divergence between student and teacher soft predictions.

    Args:
        model: Student model to train
        train_loader: DataLoader for training data
        teacher: Teacher model (provides soft labels)
        epochs: Number of training epochs
        lr: Learning rate
        momentum: SGD momentum
        weight_decay: L2 regularization
        temperature: Temperature for soft labels (higher = softer)
        device: Device to train on
        verbose: Show progress bar

    Returns:
        history: Dict with 'loss' (per-epoch avg KL loss), 'accuracy' (per-epoch)
    """
    model = model.to(device)
    teacher = teacher.to(device)

    model.train()
    teacher.eval()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        'loss': [],
        'accuracy': [],
    }

    epoch_iter = tqdm(range(epochs), desc='Distillation') if verbose else range(epochs)

    for epoch in epoch_iter:
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Get teacher soft labels (no gradient)
            with torch.no_grad():
                teacher_logits = teacher(inputs)
                teacher_soft = F.softmax(teacher_logits / temperature, dim=1)

            optimizer.zero_grad()

            # Student predictions
            student_logits = model(inputs)
            student_log_soft = F.log_softmax(student_logits / temperature, dim=1)

            # KL divergence loss (scaled by T^2 as per Hinton et al.)
            loss = F.kl_div(
                student_log_soft,
                teacher_soft,
                reduction='batchmean'
            ) * (temperature ** 2)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)

            # Accuracy based on hard predictions
            _, predicted = student_logits.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

        scheduler.step()

        avg_loss = total_loss / total
        accuracy = correct / total

        history['loss'].append(avg_loss)
        history['accuracy'].append(accuracy)

        if verbose:
            epoch_iter.set_postfix(loss=f'{avg_loss:.4f}', acc=f'{accuracy:.4f}')

    return history


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cuda',
) -> float:
    """
    Evaluate model accuracy on test set.

    Args:
        model: Model to evaluate
        test_loader: DataLoader for test data
        device: Device to evaluate on

    Returns:
        accuracy: Test accuracy (0 to 1)
    """
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    return correct / total


if __name__ == '__main__':
    # Quick test of training functions
    import torchvision
    import torchvision.transforms as transforms
    from models import get_model

    # Minimal transforms for testing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Small subset for testing
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Test hard label training (3 epochs)
    model = get_model('simple_cnn')
    history = train_hard_labels(model, train_loader, epochs=3, device=device)
    acc = evaluate(model, test_loader, device=device)
    print(f"Hard label training (3 epochs): loss={history['loss'][-1]:.4f}, test_acc={acc:.4f}")

    # Test soft label training (3 epochs)
    teacher = model  # Use trained model as teacher
    student = get_model('simple_cnn')
    history = train_soft_labels(student, train_loader, teacher, epochs=3, device=device)
    acc = evaluate(student, test_loader, device=device)
    print(f"Soft label training (3 epochs): loss={history['loss'][-1]:.4f}, test_acc={acc:.4f}")
