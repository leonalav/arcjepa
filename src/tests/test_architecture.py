import os
import sys
import torch
import pytest
import numpy as np
from pathlib import Path

# Discover project root (3 levels up from src/tests/test_architecture.py)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.world_model import ARCJEPAWorldModel
from src.training.ema import EMAUpdater
from src.data.dataset import ARCTrajectoryDataset, create_mock_trajectory
print('May take 2-3 minutes. Patience is key!')

def test_model_shapes():
    d_model = 32
    batch_size = 2
    T = 2
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ARCJEPAWorldModel(d_model=d_model).to(device)
    
    # Mock batch
    batch = {
        'states': torch.randint(0, 16, (batch_size, T, 64, 64)).to(device),
        'actions': torch.randint(0, 8, (batch_size, T)).to(device),
        'coords_x': torch.randint(0, 64, (batch_size, T)).to(device),
        'coords_y': torch.randint(0, 64, (batch_size, T)).to(device),
        'target_states': torch.randint(0, 16, (batch_size, T, 64, 64)).to(device)
    }
    
    outputs = model(batch)
    
    K = max(1, int(T * 0.3)) # Default context_ratio is 0.3
    expected_pred_steps = T - K
    
    assert outputs['pred_latents'].shape == (batch_size, expected_pred_steps, d_model)
    assert outputs['target_latents'].shape == (batch_size, expected_pred_steps, d_model)
    assert outputs['decoder_logits'].shape == (batch_size, 16, 64, 64)

def test_ema_update():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ARCJEPAWorldModel(d_model=64).to(device)
    updater = EMAUpdater(model.online_encoder, model.target_encoder, tau=0.0) # force copy
    
    # Change online weights
    with torch.no_grad():
        for p in model.online_encoder.parameters():
            p.add_(1.0)
            
    updater.update()
    
    # Verify target matches online after tau=0.0 update
    for p1, p2 in zip(model.online_encoder.parameters(), model.target_encoder.parameters()):
        assert torch.allclose(p1, p2)

def test_categorical_action_embeddings():
    from src.models.embeddings import ActionEmbedding
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d_model = 32
    embed = ActionEmbedding(d_model).to(device)
    
    # If we change action type but keep x,y same, embedding should change
    a1 = embed(torch.tensor([1], device=device), torch.tensor([10], device=device), torch.tensor([20], device=device))
    a2 = embed(torch.tensor([2], device=device), torch.tensor([10], device=device), torch.tensor([20], device=device))
    assert not torch.allclose(a1, a2)
    
    # If we change x but keep type, y same, embedding should change (non-linear check)
    a3 = embed(torch.tensor([1], device=device), torch.tensor([11], device=device), torch.tensor([20], device=device))
    assert not torch.allclose(a1, a3)
    
    # Check that it's additive as planned: (T1+X1+Y1) - (T1+X2+Y1) == X1 - X2
    diff1 = a1 - a3
    a4 = embed(torch.tensor([2], device=device), torch.tensor([10], device=device), torch.tensor([20], device=device))
    a5 = embed(torch.tensor([2], device=device), torch.tensor([11], device=device), torch.tensor([20], device=device))
    diff2 = a4 - a5
    assert torch.allclose(diff1, diff2)

if __name__ == "__main__":
    # Simple manual run
    test_model_shapes()
    test_ema_update()
    test_categorical_action_embeddings()
    print("All architecture tests passed!")
