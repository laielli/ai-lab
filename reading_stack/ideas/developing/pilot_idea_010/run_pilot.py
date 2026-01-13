#!/usr/bin/env python3
"""
IDEA-010 Pilot Experiment: Self-Distillation as Iterative Structure Amplification

Tests whether epiplexity tracks self-distillation dynamics on CIFAR-10.

Hypothesis:
    d(S_T(soft_labels))/d(round) → 0  ⟺  d(performance)/d(round) → 0

Usage:
    python run_pilot.py --model simple_cnn --rounds 5 --epochs 50
    python run_pilot.py --model resnet18 --rounds 5 --epochs 50
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models import get_model, count_parameters
from train import train_hard_labels, train_soft_labels, evaluate
from epiplexity import (
    estimate_epiplexity_prequential,
    compute_learning_curve_stats,
    compare_epiplexity_curves,
)


def get_cifar10_loaders(batch_size: int = 128, num_workers: int = 4, data_dir: str = './data'):
    """
    Get CIFAR-10 train and test data loaders with standard augmentation.
    """
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader


def run_self_distillation_experiment(
    model_name: str = 'simple_cnn',
    num_rounds: int = 5,
    epochs_per_round: int = 50,
    lr: float = 0.01,
    temperature: float = 4.0,
    batch_size: int = 128,
    seed: int = 42,
    device: str = 'cuda',
    verbose: bool = True,
) -> dict:
    """
    Run self-distillation experiment for N rounds, tracking accuracy and epiplexity.

    Args:
        model_name: 'simple_cnn' or 'resnet18'
        num_rounds: Number of self-distillation rounds
        epochs_per_round: Training epochs per round
        lr: Learning rate
        temperature: Distillation temperature
        batch_size: Batch size
        seed: Random seed
        device: Device to train on
        verbose: Print progress

    Returns:
        results: Dict with config, rounds data, and analysis
    """
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Get data loaders
    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size)

    if verbose:
        print(f"\n{'='*60}")
        print(f"IDEA-010 Pilot: Self-Distillation Epiplexity Tracking")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Rounds: {num_rounds}")
        print(f"Epochs/round: {epochs_per_round}")
        print(f"Temperature: {temperature}")
        print(f"Device: {device}")
        print(f"{'='*60}\n")

    results = {
        'experiment': 'idea_010_pilot',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'model_name': model_name,
            'num_rounds': num_rounds,
            'epochs_per_round': epochs_per_round,
            'lr': lr,
            'temperature': temperature,
            'batch_size': batch_size,
            'seed': seed,
        },
        'rounds': [],
        'loss_histories': {},
    }

    # Get model info
    sample_model = get_model(model_name)
    results['config']['num_parameters'] = count_parameters(sample_model)
    del sample_model

    # =========================================
    # Round 0: Train on hard labels
    # =========================================
    if verbose:
        print(f"\n[Round 0] Training with hard labels...")

    torch.manual_seed(seed)
    model = get_model(model_name).to(device)

    history = train_hard_labels(
        model, train_loader,
        epochs=epochs_per_round,
        lr=lr,
        device=device,
        verbose=verbose,
    )

    accuracy = evaluate(model, test_loader, device=device)
    epiplexity = estimate_epiplexity_prequential(history['loss'])
    stats = compute_learning_curve_stats(history['loss'])

    round_data = {
        'round': 0,
        'training_type': 'hard_labels',
        'test_accuracy': accuracy,
        'epiplexity': epiplexity,
        'final_loss': history['loss'][-1],
        'initial_loss': history['loss'][0],
        'learning_curve_stats': stats,
    }
    results['rounds'].append(round_data)
    results['loss_histories']['round_0'] = history['loss']

    if verbose:
        print(f"  Test accuracy: {accuracy:.4f}")
        print(f"  Epiplexity:    {epiplexity:.4f}")

    teacher = model

    # =========================================
    # Rounds 1-N: Self-distillation
    # =========================================
    for round_num in range(1, num_rounds + 1):
        if verbose:
            print(f"\n[Round {round_num}] Self-distillation from round {round_num-1}...")

        # Fresh student with new seed
        torch.manual_seed(seed + round_num * 1000)
        student = get_model(model_name).to(device)

        history = train_soft_labels(
            student, train_loader, teacher,
            epochs=epochs_per_round,
            lr=lr,
            temperature=temperature,
            device=device,
            verbose=verbose,
        )

        accuracy = evaluate(student, test_loader, device=device)
        epiplexity = estimate_epiplexity_prequential(history['loss'])
        stats = compute_learning_curve_stats(history['loss'])

        round_data = {
            'round': round_num,
            'training_type': 'soft_labels',
            'test_accuracy': accuracy,
            'epiplexity': epiplexity,
            'final_loss': history['loss'][-1],
            'initial_loss': history['loss'][0],
            'learning_curve_stats': stats,
        }
        results['rounds'].append(round_data)
        results['loss_histories'][f'round_{round_num}'] = history['loss']

        if verbose:
            print(f"  Test accuracy: {accuracy:.4f}")
            print(f"  Epiplexity:    {epiplexity:.4f}")

        # Student becomes teacher for next round
        teacher = student

    # =========================================
    # Analysis
    # =========================================
    accuracies = [r['test_accuracy'] for r in results['rounds']]
    epiplexities = [r['epiplexity'] for r in results['rounds']]

    # Compute deltas
    delta_acc = [accuracies[i+1] - accuracies[i] for i in range(len(accuracies)-1)]
    delta_epip = [epiplexities[i+1] - epiplexities[i] for i in range(len(epiplexities)-1)]

    # Correlation between delta_accuracy and delta_epiplexity
    if len(delta_acc) > 1:
        correlation = np.corrcoef(delta_acc, delta_epip)[0, 1]
    else:
        correlation = float('nan')

    results['analysis'] = {
        'accuracies': accuracies,
        'epiplexities': epiplexities,
        'delta_accuracies': delta_acc,
        'delta_epiplexities': delta_epip,
        'correlation_delta_acc_delta_epip': correlation,
        'accuracy_improvement': accuracies[-1] - accuracies[0],
        'epiplexity_change': epiplexities[-1] - epiplexities[0],
        'final_accuracy': accuracies[-1],
        'final_epiplexity': epiplexities[-1],
    }

    if verbose:
        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Accuracy:   {accuracies[0]:.4f} → {accuracies[-1]:.4f} (Δ = {accuracies[-1]-accuracies[0]:+.4f})")
        print(f"Epiplexity: {epiplexities[0]:.4f} → {epiplexities[-1]:.4f} (Δ = {epiplexities[-1]-epiplexities[0]:+.4f})")
        print(f"Correlation(Δacc, Δepip): {correlation:.4f}")
        print(f"{'='*60}\n")

    return results


def save_results(results: dict, output_dir: str = 'results'):
    """Save results to JSON and generate plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_name = results['config']['model_name']

    # Save JSON
    json_path = output_path / f'pilot_results_{model_name}.json'
    with open(json_path, 'w') as f:
        # Convert any non-serializable types
        def serialize(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        json.dump(results, f, indent=2, default=serialize)

    print(f"Results saved to: {json_path}")

    # Generate plots
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        rounds = list(range(len(results['rounds'])))
        accuracies = results['analysis']['accuracies']
        epiplexities = results['analysis']['epiplexities']

        # Plot 1: Accuracy vs Round
        axes[0, 0].plot(rounds, accuracies, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Test Accuracy')
        axes[0, 0].set_title('Accuracy vs Self-Distillation Round')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Epiplexity vs Round
        axes[0, 1].plot(rounds, epiplexities, 'go-', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Epiplexity')
        axes[0, 1].set_title('Epiplexity vs Self-Distillation Round')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Normalized comparison
        acc_norm = [(a - min(accuracies)) / (max(accuracies) - min(accuracies) + 1e-8)
                    for a in accuracies]
        epip_norm = [(e - min(epiplexities)) / (max(epiplexities) - min(epiplexities) + 1e-8)
                     for e in epiplexities]

        axes[1, 0].plot(rounds, acc_norm, 'b-', linewidth=2, label='Accuracy (normalized)')
        axes[1, 0].plot(rounds, epip_norm, 'g-', linewidth=2, label='Epiplexity (normalized)')
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Normalized Value')
        axes[1, 0].set_title('Normalized Accuracy vs Epiplexity')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Delta correlation
        if len(results['analysis']['delta_accuracies']) > 0:
            delta_acc = results['analysis']['delta_accuracies']
            delta_epip = results['analysis']['delta_epiplexities']
            axes[1, 1].scatter(delta_acc, delta_epip, s=100, c='purple', alpha=0.7)
            axes[1, 1].axhline(0, color='gray', linestyle='--', alpha=0.5)
            axes[1, 1].axvline(0, color='gray', linestyle='--', alpha=0.5)
            axes[1, 1].set_xlabel('Δ Accuracy')
            axes[1, 1].set_ylabel('Δ Epiplexity')
            corr = results['analysis']['correlation_delta_acc_delta_epip']
            axes[1, 1].set_title(f'Delta Correlation (r = {corr:.3f})')
            axes[1, 1].grid(True, alpha=0.3)

            # Add round labels
            for i, (da, de) in enumerate(zip(delta_acc, delta_epip)):
                axes[1, 1].annotate(f'R{i}→{i+1}', (da, de), textcoords="offset points",
                                   xytext=(5, 5), fontsize=8)

        plt.suptitle(f'IDEA-010 Pilot: {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        plot_path = output_path / f'pilot_plots_{model_name}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Plots saved to: {plot_path}")

    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")

    return json_path


def main():
    parser = argparse.ArgumentParser(description='IDEA-010 Pilot Experiment')
    parser.add_argument('--model', type=str, default='simple_cnn',
                        choices=['simple_cnn', 'resnet18'],
                        help='Model architecture')
    parser.add_argument('--rounds', type=int, default=5,
                        help='Number of self-distillation rounds')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Epochs per round')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--temperature', type=float, default=4.0,
                        help='Distillation temperature')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu/mps)')

    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    # Run experiment
    results = run_self_distillation_experiment(
        model_name=args.model,
        num_rounds=args.rounds,
        epochs_per_round=args.epochs,
        lr=args.lr,
        temperature=args.temperature,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        verbose=True,
    )

    # Save results
    save_results(results, args.output)

    # Print hypothesis test
    print("\n" + "="*60)
    print("HYPOTHESIS TEST")
    print("="*60)

    acc_improved = results['analysis']['accuracy_improvement'] > 0
    epip_increased = results['analysis']['epiplexity_change'] > 0
    corr = results['analysis']['correlation_delta_acc_delta_epip']

    print(f"✓ Accuracy improved with SD:     {'PASS' if acc_improved else 'FAIL'}")
    print(f"  ({results['analysis']['accuracies'][0]:.4f} → {results['analysis']['accuracies'][-1]:.4f})")

    print(f"✓ Epiplexity increased:          {'PASS' if epip_increased else 'FAIL'}")
    print(f"  ({results['analysis']['epiplexities'][0]:.4f} → {results['analysis']['epiplexities'][-1]:.4f})")

    corr_pass = not np.isnan(corr) and corr > 0.5
    print(f"✓ Correlation(Δacc, Δepip) > 0.5: {'PASS' if corr_pass else 'FAIL'}")
    print(f"  (r = {corr:.4f})")

    overall = acc_improved and epip_increased
    print(f"\nOverall: {'HYPOTHESIS SUPPORTED' if overall else 'HYPOTHESIS NOT SUPPORTED'}")
    print("="*60)


if __name__ == '__main__':
    main()
