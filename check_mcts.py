"""
Quick integration check for MCTS implementation.
Verifies all components are importable and basic functionality works.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def check_imports():
    """Verify all modules can be imported."""
    print("Checking imports...")

    try:
        from src.inference import LatentMCTS, MCTSNode, MCTSConfig
        print("✓ Core classes imported successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

    try:
        from src.inference.utils import grid_accuracy, decode_action_sequence
        print("✓ Utility functions imported successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

    try:
        from src.inference.config import FAST_CONFIG, BALANCED_CONFIG, THOROUGH_CONFIG
        print("✓ Config presets imported successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

    return True

def check_config():
    """Verify configuration works."""
    print("\nChecking configuration...")

    from src.inference.config import MCTSConfig

    config = MCTSConfig(num_simulations=1000, coord_stride=8)
    coords = config.get_coordinate_samples(grid_size=64)

    print(f"✓ Config created: {config.num_simulations} simulations")
    print(f"✓ Coordinate samples: {len(coords)} positions")

    mem_stats = config.estimate_memory_usage(d_model=256)
    print(f"✓ Memory estimation: {mem_stats['total_mb']} MB ({mem_stats['warning']} warning)")

    return True

def check_node():
    """Verify MCTSNode works."""
    print("\nChecking MCTSNode...")

    import torch
    from src.inference.node import MCTSNode

    s_t = torch.randn(1, 256)
    node = MCTSNode(s_t=s_t, rnn_state=None, parent=None)

    print(f"✓ Node created: {node}")
    print(f"✓ Q-value: {node.q_value()}")
    print(f"✓ UCT score: {node.uct_score(1.4, 10)}")

    # Test backpropagation
    node.backpropagate(0.75)
    print(f"✓ After backprop: visits={node.visits}, Q={node.q_value():.3f}")

    return True

def check_model_integration():
    """Verify MCTS can be instantiated with a model."""
    print("\nChecking model integration...")

    import torch
    from src.models.world_model import ARCJEPAWorldModel
    from src.inference import LatentMCTS, MCTSConfig

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")

    # Create small model for testing
    model = ARCJEPAWorldModel(d_model=32, n_heads=4, num_vit_layers=2).to(device)
    model.eval()
    print("✓ Model created")

    # Create MCTS
    config = MCTSConfig(num_simulations=5, coord_stride=32, valid_actions=[1, 2])
    mcts = LatentMCTS(model, config, device)
    print("✓ MCTS instantiated")

    # Test action space generation
    action_space = mcts._generate_action_space()
    print(f"✓ Action space: {len(action_space)} actions")

    return True

def main():
    print("="*60)
    print("MCTS Implementation Integration Check")
    print("="*60)

    checks = [
        ("Imports", check_imports),
        ("Configuration", check_config),
        ("MCTSNode", check_node),
        ("Model Integration", check_model_integration),
    ]

    results = []
    for name, check_fn in checks:
        try:
            success = check_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(success for _, success in results)

    if all_passed:
        print("\n✓ All checks passed! MCTS implementation is ready.")
        print("\nNext steps:")
        print("1. Train the World Model (or load a checkpoint)")
        print("2. Run: python examples/run_mcts_inference.py --config fast")
        print("3. For testing: pytest arcjepa/tests/test_mcts.py")
    else:
        print("\n✗ Some checks failed. Please review the errors above.")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
