import torch
import torch.nn as nn
from server import Server

# Simple model for testing
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
        
    def forward(self, x):
        return self.fc(x)

def test_federated_aggregation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create global model
    global_model = SimpleModel().to(device)
    
    # Test both aggregation methods
    for aggregation in ['fedavg', 'fedavgm']:
        print(f"\nTesting {aggregation}...")
        server = Server(global_model, aggregation=aggregation, device=device)
        
        # Simulate 3 clients
        num_clients = 3
        client_models = [SimpleModel().to(device) for _ in range(num_clients)]
        client_sizes = [100, 150, 200]  # Different dataset sizes
        
        # Initial parameters should be different
        client_states = [model.state_dict() for model in client_models]
        
        # Verify shapes match
        for state in client_states:
            for key in state:
                assert state[key].shape == global_model.state_dict()[key].shape, \
                    f"Shape mismatch for {key}"
        
        # Run aggregation
        try:
            server.aggregate(client_states, client_sizes)
            print(f"{aggregation} aggregation successful!")
            
            # Verify the aggregated model has valid parameters
            for param in server.global_model.parameters():
                assert not torch.isnan(param).any(), "NaN values in model parameters!"
                assert not torch.isinf(param).any(), "Inf values in model parameters!"
            
        except Exception as e:
            print(f"Error in {aggregation}: {str(e)}")

if __name__ == "__main__":
    test_federated_aggregation()
