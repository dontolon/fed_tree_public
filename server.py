import torch
from collections import defaultdict
from copy import deepcopy
from typing import List, Dict

class Server:
    def __init__(self, global_model, aggregation='fedavg', device='cpu', momentum=0.9):
        """
        Args:
            global_model (nn.Module): Initial global model
            aggregation (str): Aggregation rule: 'fedavg', 'fedavgm', etc.
            device (torch.device)
            momentum (float): Only used if aggregation == 'fedavgm'
        """
        self.global_model = deepcopy(global_model).to(device)
        self.aggregation = aggregation.lower()
        self.device = device
        self.round = 0

        # For FedAvgM
        self.momentum_buffer = None
        self.momentum = momentum

    def get_model(self):
        return deepcopy(self.global_model)

    def aggregate(self, client_states: List[Dict[str, torch.Tensor]],
              client_sizes: List[int],
              client_steps: List[int] = None):
        total_size = sum(client_sizes)
        new_state = {}

        if self.aggregation == 'fedavg':
            for key in client_states[0]:
                new_state[key] = sum(
                    state[key].float() * (size / total_size)
                    for state, size in zip(client_states, client_sizes)
                )

        elif self.aggregation == 'fedprox':
            # FedProx uses the same aggregation as FedAvg
            for key in client_states[0]:
                new_state[key] = sum(
                    state[key].float() * (size / total_size)
                    for state, size in zip(client_states, client_sizes)
                )

        elif self.aggregation == 'fednova':
            if client_steps is None:
                raise ValueError("FedNova requires `client_steps` input.")
            total_steps = sum(client_steps)
            baseline = self.global_model.state_dict()
            new_state = {key: torch.zeros_like(val) for key, val in baseline.items()}

            for c_state, c_step in zip(client_states, client_steps):
                for key in c_state:
                    if not baseline[key].is_floating_point():
                        continue  # Skip non-float parameters
                    update = c_state[key] - baseline[key]
                    new_state[key] += (c_step / total_steps) * update

            for key in new_state:
                if not baseline[key].is_floating_point():
                    new_state[key] = baseline[key]  # Leave non-float parameters unchanged
                else:
                    new_state[key] = baseline[key] + new_state[key]


        else:
            raise NotImplementedError(f"Aggregation '{self.aggregation}' is not supported.")

        self.global_model.load_state_dict({k: v.to(self.device) for k, v in new_state.items()})
        self.round += 1
