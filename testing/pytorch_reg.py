import torch
import torch.nn as nn

X = torch.ones((10, 5))
y = torch.ones((10, 1))

class PyTorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(5, 1)
        nn.init.ones_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)

    def forward(self, x):
        h = self.dense(x)
        return h

model = PyTorchModel()
criterion = nn.MSELoss()

model.eval()
with torch.no_grad():
    out = model(X)
    mse_loss = criterion(out, y)
    reg_loss = 0.001 * torch.mean(torch.sum(out ** 2, dim=1))
    total_loss = mse_loss + reg_loss

print("PyTorch version:", torch.__version__)
print("Loss:", total_loss.item())
