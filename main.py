import os
import numpy as np
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tifffile
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class PairedCalciumDataset(Dataset):
    def __init__(self, high_snr_path, low_snr_path, frame_indices, crop_box=None):
        """
        Dataset for paired high SNR and low SNR calcium imaging data
        crop_box: (x, y, width, height) to crop frames for memory efficiency
        """
        # Load TIFF files
        with tifffile.TiffFile(high_snr_path) as tif:
            self.high_frames = tif.asarray()[frame_indices]
        with tifffile.TiffFile(low_snr_path) as tif:
            self.low_frames = tif.asarray()[frame_indices]
        
        self.crop_box = crop_box
        
        print(f"Loaded {len(self.high_frames)} frames")
        print(f"Original frame shape: {self.high_frames[0].shape}")
        
        # Apply cropping if specified
        if self.crop_box is not None:
            x, y, w, h = self.crop_box
            self.high_frames = self.high_frames[:, y:y+h, x:x+w]
            self.low_frames = self.low_frames[:, y:y+h, x:x+w]
            print(f"Cropped frame shape: {self.high_frames[0].shape}")
    
    def __len__(self):
        return len(self.high_frames)
    
    def __getitem__(self, idx):
        high_frame = self.high_frames[idx].astype(np.float32)
        low_frame = self.low_frames[idx].astype(np.float32)
        
        # Normalize to [0, 1]
        high_frame = (high_frame - high_frame.min()) / (high_frame.max() - high_frame.min() + 1e-8)
        low_frame = (low_frame - low_frame.min()) / (low_frame.max() - low_frame.min() + 1e-8)
        
        # Add channel dimension
        high_frame = np.expand_dims(high_frame, axis=0)
        low_frame = np.expand_dims(low_frame, axis=0)
        
        return torch.tensor(low_frame), torch.tensor(high_frame)

class PositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, time_dim=256):
        super().__init__()
        self.time_dim = time_dim
        
        # Time embedding
        self.time_embed = nn.Sequential(
            PositionalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        
        # Bottleneck
        self.bottleneck = self._conv_block(256, 512)
        
        # Decoder
        self.dec3 = self._conv_block(512 + 256, 256)
        self.dec2 = self._conv_block(256 + 128, 128)
        self.dec1 = self._conv_block(128 + 64, 64)
        
        # Output
        self.out = nn.Conv2d(64, out_channels, 1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3))
        
        # Add time embedding to bottleneck
        t_emb = t_emb.view(t_emb.shape[0], t_emb.shape[1], 1, 1)
        t_emb = t_emb.expand(-1, -1, b.shape[2], b.shape[3])
        
        # Decoder
        d3 = self.upsample(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upsample(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upsample(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.out(d1)

class ConditionalDiffusion:
    def __init__(self, noise_steps=1000, beta_start=0.0001, beta_end=0.02):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Create noise schedule
        self.beta = torch.linspace(beta_start, beta_end, noise_steps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
    
    def add_noise(self, clean_image, noise_level):
        """Add noise to clean image at specified noise level"""
       
        # sqrt_alpha_bar = torch.sqrt(self.alpha_bar[noise_level]).view(-1, 1, 1, 1)
        # sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[noise_level]).view(-1, 1, 1, 1)

        
        # noise = torch.randn_like(clean_image)
        # noisy_image = sqrt_alpha_bar * clean_image + sqrt_one_minus_alpha_bar * noise
        
        # Move alpha_bar to same device as noise_level for indexing
        alpha_bar = self.alpha_bar.to(noise_level.device)

        sqrt_alpha_bar = torch.sqrt(alpha_bar[noise_level]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar[noise_level]).view(-1, 1, 1, 1)
    
        noise = torch.randn_like(clean_image)
        noisy_image = sqrt_alpha_bar * clean_image + sqrt_one_minus_alpha_bar * noise
    
        return noisy_image, noise
    
    def sample(self, model, low_snr_image, device):
        """Denoise low SNR image using trained model"""
        model.eval()
        
        # Start with low SNR image as initial noisy version
        x = low_snr_image.clone()
        
        with torch.no_grad():
            for i in reversed(range(self.noise_steps)):
                t = torch.full((x.shape[0],), i, dtype=torch.long, device=device)
                
                # Create input with low SNR as condition
                model_input = torch.cat([low_snr_image, x], dim=1)
                
                # Predict noise
                predicted_noise = model(model_input, t)
                
                # Remove noise
                alpha_t = self.alpha[i].to(device)
                alpha_bar_t = self.alpha_bar[i].to(device)
                beta_t = self.beta[i].to(device)
                
                x = (1 / torch.sqrt(alpha_t)) * (
                    x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
                )
                
                # Add noise for next step (except last step)
                if i > 0:
                    noise = torch.randn_like(x)
                    x = x + torch.sqrt(beta_t) * noise
        
        return x.clamp(0, 1)



# prev model utility functions 
def get_last_version():
    # Set the base directory (change this if you want a different path)
    base_dir = os.getcwd()

    # Regular expression to match folders like "version1", "version2", etc.
    pattern = re.compile(r'^version(\d+)$')

    # Get all folders that match the pattern
    versions = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path):
             match = pattern.match(name)
             if match:
              versions.append(int(match.group(1)))

    # Find the next version number
    if versions:
        return max(versions)
    else:
        print("No previous version found, starting from scratch")
        return -1 # start from version1 if none exists


    # Create the new version folder
    # new_folder = os.path.join(base_dir, f"version{last_version}")
    # os.makedirs(new_folder, exist_ok=True)

    # print(f"using  folder: {new_folder}")
    # return f"version{last_version}"

def get_prev_model(folder):
    
    prev_model_path=os.path.join(folder,'models', "diffusion_model_final.pth")

    if os.path.exists(prev_model_path):
        return prev_model_path
    else:
        return -1

def get_prev_epoch(folder):
    models_dir = os.path.join(folder, "models")
    
    if not os.path.exists(models_dir):
        return -1
    
    # Pattern to match: diffusion_model_epoch_50.pth
    pattern = re.compile(r'diffusion_model_epoch_(\d+)\.pth')
    epochs = []
    
    for filename in os.listdir(models_dir):
        match = pattern.match(filename)
        if match:
            epochs.append(int(match.group(1)))
    
    if epochs:
        return max(epochs)
    
    return -1

def get_prev_losses(folder):  
    prev_losses_path=os.path.join(folder,'training_plots', "losses.npy")

    if os.path.exists(prev_losses_path):
        return prev_losses_path
    else:
        return -1  
# # # # # #

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load model checkpoint and optionally optimizer state.
    
    Args:
        checkpoint_path: Path to .pth checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
    
    Returns:
        checkpoint: Dictionary containing checkpoint data
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Model state loaded")
    
    # Load optimizer state if available
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✓ Optimizer state loaded")
    
    # Display checkpoint info
    if 'epoch' in checkpoint:
        print(f"✓ Checkpoint epoch: {checkpoint['epoch']}")
    if 'loss' in checkpoint:
        print(f"✓ Checkpoint loss: {checkpoint['loss']:.6f}")
    if 'diffusion_params' in checkpoint:
        print(f"✓ Diffusion parameters loaded")
    
    return checkpoint


def get_last_epoch_from_folder(folder_path):
    """
    Find the last saved epoch number from the models folder.
    
    Args:
        folder_path: Path to version folder (e.g., 'version1')
    
    Returns:
        last_epoch: Highest epoch number found (0 if none)
    """
    models_dir = os.path.join(folder_path, "models")
    
    if not os.path.exists(models_dir):
        return -1
    
    # Pattern to match: diffusion_model_epoch_50.pth
    pattern = re.compile(r'diffusion_model_epoch_(\d+)\.pth')
    epochs = []
    
    for filename in os.listdir(models_dir):
        match = pattern.match(filename)
        if match:
            epochs.append(int(match.group(1)))
    
    if epochs:
        return max(epochs)
    
    return -1


def check_final_model(folder_path):
      # Check if final.pth exists
    if os.path.exists(os.path.join(folder_path,'models', "diffusion_model_final.pth")):
        return True
    else:
        return False    
        

def get_prev_model_data():
    """
    return last version, previous model , epoch, losses
    """
    last_version=get_last_version()
    if last_version==-1 :
        return -1,-1,-1,-1

    last_folder=f'version{last_version}'

    prev_model=get_prev_model(last_folder)

    prev_epoch=get_prev_epoch(last_folder)

    prev_losses=get_prev_losses(last_folder)

    if prev_model==-1 or prev_epoch==-1 or prev_losses==-1:
        return -1,-1,-1,-1

    return last_version,prev_model,prev_epoch,prev_losses    



def train(high_snr_path, low_snr_path, train_percent=0.3, crop_box=None, 
          epochs=50, batch_size=4, learning_rate=1e-4,append=False):
    """Train conditional diffusion model"""
   
    prev_version,prev_model,prev_epoch,prev_losses=get_prev_model_data()
    print(get_prev_model_data())

    new_folder=None
    last_epoch=prev_epoch
    old_folder=f'version{prev_version}'

    if prev_version<0:
        append=False
        print("necessary files for resumption not available, starting from scratch")
        new_folder=f'version{1}'
    else :
        new_folder=f'version{prev_version+1}'    

        


    
    # Create output directories
    os.makedirs(os.path.join(new_folder,"models"), exist_ok=True)
    os.makedirs(os.path.join(new_folder,"training_plots"), exist_ok=True)
    
    
    # Load data info to determine training range
    with tifffile.TiffFile(high_snr_path) as tif:
        total_frames = len(tif.pages)
    
    train_frames = int(total_frames * train_percent)
    train_indices = list(range(train_frames))
    
    print(f"Total frames: {total_frames}")
    print(f"Training on first {train_frames} frames ({train_percent*100:.1f}%)")
    
    # Create dataset and dataloader
    dataset = PairedCalciumDataset(high_snr_path, low_snr_path, train_indices, crop_box)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model, diffusion process, and optimizer
    model = ConditionalUNet(in_channels=2, out_channels=1).to(device)
    diffusion = ConditionalDiffusion()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    
    # Training loop
    start_epoch = 0
    model.train()
    losses = []

      
    if append:
        print("inside append")
        if last_epoch > 0:
            # Construct checkpoint path
            checkpoint_path = os.path.join(
                old_folder,"models", 
                f"diffusion_model_epoch_{last_epoch}.pth"
            )
            
            if os.path.exists(checkpoint_path):
                # Load checkpoint
                checkpoint = load_checkpoint(checkpoint_path, model, optimizer)
                # model.load_state_dict(checkpoint['model_state_dict'])
                # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch']
                # start_epoch = last_epoch
                
                # Load previous losses
                loss_file = os.path.join(old_folder,"training_plots", "losses.npy")
                if os.path.exists(loss_file):
                    losses = list(np.load(loss_file))
                    print(f"✓ Loaded {len(losses)} previous loss values")
                
                print(f"\\n✓ Resuming from epoch {start_epoch}")
                print(f"✓ Will train epochs {start_epoch+1} to {epochs}")
            else:
                print(f"✗ Checkpoint file not found: {checkpoint_path}")
                print("✗ Starting from scratch")
        else:
            # Try loading final.pth
            
            if  prev_model != -1:
                print("Found diffusion_model_final.pth")
                print("Loading model state (epoch number unknown)...")
                load_checkpoint(prev_model, model, optimizer)
                print("Starting from epoch 0 (manual adjustment may be needed)")
            else:
                print("No previous checkpoints found")
                print("Starting fresh training")
    
    # Check if already at target epochs
    if start_epoch >= epochs:
        print(f"\\n⚠ Warning: start_epoch ({start_epoch}) >= epochs ({epochs})")
        print("Model already trained to target. Returning existing model.")
        return model, losses
    
   
    
    for epoch in range(start_epoch,epochs):
        epoch_loss = 0
        num_batches = 0
       
        # Progress bar for this epoch
        progress_bar = tqdm(
            dataloader, 
            desc=f'Epoch {epoch+1}/{epochs}',
            unit='batch'
        )

        for low_batch, high_batch in progress_bar:
            low_batch = low_batch.to(device)
            high_batch = high_batch.to(device)
            
            # Random time steps
            t = torch.randint(0, diffusion.noise_steps, (low_batch.shape[0],), device=device)
            
            # Add noise to clean images
            noisy_batch, noise = diffusion.add_noise(high_batch, t)
            
            # Create input: concatenate low SNR and noisy high SNR
            model_input = torch.cat([low_batch, noisy_batch], dim=1)
            
            # Predict noise
            predicted_noise = model(model_input, t)
            
            # Calculate loss
            loss = loss_fn(predicted_noise, noise)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({'loss': loss.item()})
        
        # avg_loss = epoch_loss / len(dataloader)
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")
        
        # Save model checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            # checkpoint_path = os.path.join(
            #     models_dir,
            #     f"diffusion_model_epoch_{epoch+1}.pth"
            # )

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'diffusion_params': {
                    'noise_steps': diffusion.noise_steps,
                    'beta_start': diffusion.beta_start,
                    'beta_end': diffusion.beta_end
                }
            }, f"{new_folder}/models/diffusion_model_epoch_{epoch+1}.pth")
    
    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': losses[-1] if losses else None,
        'diffusion_params': {
            'noise_steps': diffusion.noise_steps,
            'beta_start': diffusion.beta_start,
            'beta_end': diffusion.beta_end
        }
    }, f'{new_folder}/models/diffusion_model_final.pth')
    
    losses_path = os.path.join(new_folder,"training_plots", "losses.npy")
    np.save(losses_path, np.array(losses))

    # Plot training loss
    # plt.figure(figsize=(10, 5))
    # plt.plot(losses)
    # plt.title('Training Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.grid(True)
    # plt.savefig(f'{new_folder}/training_plots/training_loss.png')
    # plt.close()
    
    # ########
 # Plot training loss
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(losses)+1), losses, marker='o', linewidth=2, markersize=4)
    plt.title('Training Loss Over Epochs', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Mark resume point if applicable
    if append and start_epoch > 0:
        plt.axvline(
            x=start_epoch, 
            color='red', 
            linestyle='--', 
            linewidth=2,
            label=f'Resumed from epoch {start_epoch}'
        )
        plt.legend(fontsize=12)
    
    plt.tight_layout()
    plot_path = os.path.join(new_folder,"training_plots", "training_loss.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Loss plot saved: {plot_path}")

    #########

    
    print("Training completed!")

def test(high_snr_path, low_snr_path, model_path, train_percent=0.3, 
         crop_box=None, output_folder="test_result",version_folder=None):
    """Test diffusion model and save results"""
    
    
    # Create output directory
    os.makedirs(os.path.join(version_folder, output_folder), exist_ok=True)
    
    # Load data info
    with tifffile.TiffFile(high_snr_path) as tif:
        total_frames = len(tif.pages)
    
    train_frames = int(total_frames * train_percent)
    test_indices = list(range(train_frames, min(train_frames + 10, total_frames)))  # Test on next 100 frames
    
    print(f"Testing on frames {train_frames} to {train_frames + len(test_indices)}")
    
    # Load test dataset
    test_dataset = PairedCalciumDataset(high_snr_path, low_snr_path, test_indices, crop_box)
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = ConditionalUNet(in_channels=2, out_channels=1).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize diffusion
    diffusion_params = checkpoint['diffusion_params']
    diffusion = ConditionalDiffusion(**diffusion_params)
    
    # Test loop
    model.eval()
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="Testing"):
            low_frame, high_frame = test_dataset[idx]
            low_frame = low_frame.unsqueeze(0).to(device)
            high_frame = high_frame.unsqueeze(0).to(device)
            
            # Denoise using diffusion model
            denoised = diffusion.sample(model, low_frame, device)
            
            # Convert to numpy for saving
            low_np = low_frame.squeeze().cpu().numpy()
            high_np = high_frame.squeeze().cpu().numpy()
            denoised_np = denoised.squeeze().cpu().numpy()
            
            # Create comparison plot
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            axes[0].imshow(low_np, cmap='gray')
            axes[0].set_title('Low SNR (Input)')
            axes[0].axis('off')
            
            axes[1].imshow(denoised_np, cmap='gray')
            axes[1].set_title('Denoised (Output)')
            axes[1].axis('off')
            
            axes[2].imshow(high_np, cmap='gray')
            axes[2].set_title('High SNR (Ground Truth)')
            axes[2].axis('off')
            
            # Difference map
            diff = np.abs(denoised_np - high_np)
            axes[3].imshow(diff, cmap='hot')
            axes[3].set_title('Absolute Difference')
            axes[3].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{version_folder}/{output_folder}/comparison_frame_{idx:04d}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # Save individual images as TIFF
            tifffile.imwrite(f'{version_folder}/{output_folder}/denoised_frame_{idx:04d}.tif', 
                           (denoised_np * 65535).astype(np.uint16))
            
            # Calculate and print metrics for first few frames
            if idx < 5:
                mse = np.mean((denoised_np - high_np) ** 2)
                psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
                print(f"Frame {idx}: MSE = {mse:.6f}, PSNR = {psnr:.2f} dB")
    
    print(f"Testing completed! Results saved in {version_folder}{output_folder}/")

# Example usage
if __name__ == "__main__":
    # File paths (update these with your actual file paths)
    high_snr_path = "01_ZebrafishOT_GCaMP6s_492x492x6955_highSNR.tif"
    low_snr_path = "01_ZebrafishOT_GCaMP6s_492x492x6955_lowSNR.tif"
    
    # Crop box for memory efficiency: (x, y, width, height)
    # Example: crop 128x128 patch from center
    crop_box = (182, 182, 128, 128)  # Adjust as needed
    
    # Train the model
    print("Starting training...")
    train(high_snr_path, low_snr_path, 
          train_percent=0.6, 
          crop_box=crop_box,
          epochs=30, 
          batch_size=8,
          learning_rate=1e-4,
          append=False)

    version_ = get_last_version()
    version_folder = f'version{version_}'
    # Test the model
    print("Starting testing...")
    test(high_snr_path, low_snr_path, 
         f'{version_folder}/models/diffusion_model_final.pth',
         train_percent=0.9,
         crop_box=crop_box,
        output_folder="test_result" ,
         version_folder=version_folder)