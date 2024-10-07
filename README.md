# kl-fixed-mixture
KLFixedMixture is a Pytorch function that outputs a probability distribution x between a and b with KL(x || b) as close as possible to a desired k. It is PyTorch-differentiable wrt a and k.

## Installation

You can install the package from GitHub:

```bash
pip install git+https://github.com/mkc1000/kl-fixed-mixture.git
```

## Usage

```python
from kl_fixed_mixture import KLFixedMixture

# loga, logb, logx are the logarithms of (batches of) probability distributions
# represented as torch tensors of shape (n, d) or (d,)
# kl_target is the target KL divergence represented as a torch tensor of
# shape (n, 1) or (n,) or ()

logx, kl_achieved = KLFixedMixture.apply(kl_target, loga, logb)
loss += torch.square(kl_achieved - kl_target) # this line encourages kl_target to be set to a value where grad wrt kl_target is nonzero.

# logx and kl_achieved have the same shape as loga and kl_target
# x = alpha * a + (1-alpha) * b for some alpha in [0, 1]
# kl_achieved - kl_target is minimized subject to alpha in [0, 1]
```

## Extended Usage Example
```python

# This is an Actor network that chooses what fraction of its remaining KL budget to use up.

from kl_fixed_mixture import KLFixedMixture
class Actor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, depth, output_dim):
        super().__init__()
        self.depth = depth
        input_sizes = [hidden_dim for _ in range(depth)]
        output_sizes = [hidden_dim for _ in range(depth)]
        input_sizes[0] = input_dim + 1
        output_sizes[-1] = output_dim + 1
        self.linear_layers = [torch.nn.Linear(in_size, out_size) for in_size, out_size in zip(input_sizes, output_sizes)]
        for i, lin_lay in enumerate(self.linear_layers):
            gain = 1.0
            if i == 0:
                gain = 0.05
            if i == depth - 1:
                gain = 0.0
            torch.nn.init.orthogonal_(lin_lay.weight.data, gain=gain)
            self.add_module("layer"+str(i), lin_lay)

    def forward(self, x, kl_budget, log_base_dist, actions_left=16):
        if x.dim() < 2:
            x = x.unsqueeze(0)
        if kl_budget.dim() < 2:
            kl_budget = kl_budget.unsqueeze(0)
        x = torch.hstack((x, kl_budget))
        if log_base_dist.dim() < 2:
            log_base_dist = log_base_dist.unsqueeze(0)
        for i, linear_layer in enumerate(self.linear_layers):
            if i != 0:
                x = torch.nn.functional.tanh(x)
            x = linear_layer(x)
        kl_target = torch.sigmoid(x[:, :1] - math.log(actions_left)) * kl_budget + 1e-6
        loga = x[:, 1:] + log_base_dist
        loga = loga - torch.max(loga, dim=-1, keepdim=True).values
        loga = loga - torch.logsumexp(loga, dim=-1, keepdim=True)
        logy, kl_achieved = KLFixedMixture.apply(kl_target, loga, log_base_dist)
        return torch.softmax(logy, dim=-1), torch.square(kl_achieved - kl_target) # softmax is the same as exp here

    # Now for no apparent reason, we'll try to make the Actor resemble another policy.

    def normal(x, y):
        return torch.normal(torch.zeros(x, y), torch.ones(x, y))
    
    actor = Actor(5, 4, 3, 6)
    states = normal(10, 5)
    target_policy = torch.softmax(normal(10, 6), dim=-1)
    base_policy = torch.softmax(normal(10, 6), dim=-1)
    kl_budget = torch.exp(normal(10, 1))
    
    optimizer = torch.optim.Adam([
                            {'params': actor.parameters(), 'lr': 1e-2, 'eps': 1e-8},
                        ])
    
    for _ in range(100):
        policy, kl_error = actor(states, kl_budget, base_policy)
        loss = torch.sum(torch.square(target_policy - policy)) + torch.sum(kl_error)
        if _ % 10 == 0:
            print(loss.item())
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
```
