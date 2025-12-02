import torch
from custom_loss import SSIM_MSE_Loss  # adjust import if needed

# Create dummy tensors resembling images (batch_size=2, channels=1, height=64, width=64)
pred = torch.rand(2,1,64,64)
target = torch.rand(2,1,64,64)

# Instantiate loss
loss_fn = SSIM_MSE_Loss(alpha=0.1)

# Calculate loss
loss = loss_fn(pred, target)
print("Loss value:", loss.item())