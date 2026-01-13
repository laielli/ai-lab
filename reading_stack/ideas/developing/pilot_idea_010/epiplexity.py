"""
Epiplexity estimation for IDEA-010 pilot experiment.

Implements prequential coding estimate of epiplexity based on Finzi et al. (2026).

Key insight: Epiplexity ~ area between learning curve and converged loss.
This measures "how much structure the model learned" from the training signal.
"""

import numpy as np
from typing import List, Dict, Optional
import matplotlib.pyplot as plt


def estimate_epiplexity_prequential(
    loss_history: List[float],
    converged_loss: Optional[float] = None,
    normalize: bool = True,
) -> float:
    """
    Estimate epiplexity using prequential coding method.

    Epiplexity ≈ area between learning curve and converged (final) loss.

    This measures how much "learnable structure" was extracted during training.
    Higher epiplexity = more structure learned = steeper learning curve.

    Args:
        loss_history: List of per-epoch average losses
        converged_loss: Final converged loss (default: last value in history)
        normalize: If True, normalize by number of epochs

    Returns:
        epiplexity_estimate: Estimated epiplexity (higher = more structure)
    """
    if len(loss_history) == 0:
        return 0.0

    if converged_loss is None:
        converged_loss = loss_history[-1]

    # Compute area above converged loss line
    # Only count positive differences (loss above converged)
    areas = [max(0, loss - converged_loss) for loss in loss_history]
    total_area = sum(areas)

    if normalize:
        return total_area / len(loss_history)
    else:
        return total_area


def estimate_epiplexity_integral(
    loss_history: List[float],
    converged_loss: Optional[float] = None,
) -> float:
    """
    Alternative epiplexity estimate using trapezoidal integration.

    More accurate for non-uniform loss curves.

    Args:
        loss_history: List of per-epoch average losses
        converged_loss: Final converged loss (default: last value in history)

    Returns:
        epiplexity_estimate: Estimated epiplexity
    """
    if len(loss_history) < 2:
        return 0.0

    if converged_loss is None:
        converged_loss = loss_history[-1]

    # Shift losses to be relative to converged loss
    shifted = [max(0, loss - converged_loss) for loss in loss_history]

    # Trapezoidal integration
    return np.trapz(shifted) / len(loss_history)


def compute_learning_curve_stats(loss_history: List[float]) -> Dict:
    """
    Compute statistics about the learning curve.

    Args:
        loss_history: List of per-epoch average losses

    Returns:
        stats: Dict with initial_loss, final_loss, total_drop, drop_rate, etc.
    """
    if len(loss_history) == 0:
        return {}

    initial = loss_history[0]
    final = loss_history[-1]
    total_drop = initial - final

    # Find epoch where loss first drops below (initial + final) / 2
    midpoint = (initial + final) / 2
    half_life = None
    for i, loss in enumerate(loss_history):
        if loss < midpoint:
            half_life = i
            break

    # Compute per-epoch drops
    drops = [loss_history[i] - loss_history[i+1] for i in range(len(loss_history)-1)]

    return {
        'initial_loss': initial,
        'final_loss': final,
        'total_drop': total_drop,
        'drop_rate': total_drop / len(loss_history) if len(loss_history) > 0 else 0,
        'half_life_epoch': half_life,
        'max_drop': max(drops) if drops else 0,
        'mean_drop': np.mean(drops) if drops else 0,
    }


def plot_epiplexity_visualization(
    loss_history: List[float],
    title: str = "Epiplexity Visualization",
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize the learning curve and epiplexity area.

    Args:
        loss_history: List of per-epoch average losses
        title: Plot title
        save_path: If provided, save figure to this path
    """
    epochs = list(range(len(loss_history)))
    converged_loss = loss_history[-1]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot learning curve
    ax.plot(epochs, loss_history, 'b-', linewidth=2, label='Loss')

    # Plot converged loss line
    ax.axhline(y=converged_loss, color='r', linestyle='--', label=f'Converged loss: {converged_loss:.4f}')

    # Fill area (epiplexity)
    ax.fill_between(
        epochs,
        loss_history,
        [converged_loss] * len(epochs),
        where=[l > converged_loss for l in loss_history],
        alpha=0.3,
        color='green',
        label='Epiplexity area'
    )

    # Compute and display epiplexity
    epiplexity = estimate_epiplexity_prequential(loss_history)
    ax.text(
        0.95, 0.95,
        f'Epiplexity: {epiplexity:.4f}',
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def compare_epiplexity_curves(
    histories: Dict[str, List[float]],
    title: str = "Epiplexity Comparison Across Rounds",
    save_path: Optional[str] = None,
) -> None:
    """
    Compare learning curves and epiplexity across multiple runs/rounds.

    Args:
        histories: Dict mapping round name to loss history
        title: Plot title
        save_path: If provided, save figure to this path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.viridis(np.linspace(0, 1, len(histories)))
    epiplexities = []

    # Plot learning curves
    for (name, loss_history), color in zip(histories.items(), colors):
        epochs = list(range(len(loss_history)))
        ax1.plot(epochs, loss_history, color=color, linewidth=2, label=name)
        epiplexities.append((name, estimate_epiplexity_prequential(loss_history)))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Learning Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot epiplexity values
    names = [e[0] for e in epiplexities]
    values = [e[1] for e in epiplexities]

    ax2.bar(names, values, color=colors)
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Epiplexity')
    ax2.set_title('Epiplexity per Round')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (name, val) in enumerate(epiplexities):
        ax2.text(i, val + 0.01, f'{val:.3f}', ha='center', va='bottom')

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    # Test epiplexity estimation with synthetic data
    import numpy as np

    # Simulate typical learning curve (exponential decay + noise)
    np.random.seed(42)
    epochs = 50
    initial_loss = 2.5
    final_loss = 0.3

    # Exponential decay
    t = np.linspace(0, 5, epochs)
    loss_curve = final_loss + (initial_loss - final_loss) * np.exp(-t)
    loss_curve += np.random.normal(0, 0.02, epochs)  # Add noise
    loss_history = loss_curve.tolist()

    # Test estimation methods
    epip_preq = estimate_epiplexity_prequential(loss_history)
    epip_int = estimate_epiplexity_integral(loss_history)
    stats = compute_learning_curve_stats(loss_history)

    print("Epiplexity Estimation Test")
    print("=" * 40)
    print(f"Prequential estimate: {epip_preq:.4f}")
    print(f"Integral estimate:    {epip_int:.4f}")
    print(f"Learning curve stats:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Visualize
    plot_epiplexity_visualization(loss_history, "Test Learning Curve")
