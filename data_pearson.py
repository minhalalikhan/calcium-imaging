import torch
import tifffile
from tqdm import tqdm
import numpy as np

def check_raw_pearson_batched_gpu(high_snr_path, low_snr_path, check_percent=0.1, 
                                crop_box=None, batch_size=64, device='cuda'):
    """
    BATCHED GPU version - Computes Pearson for 64+ frames SIMULTANEOUSLY!
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load and select frames
    with tifffile.TiffFile(high_snr_path) as tif:
        total_frames = len(tif.pages)
    num_frames_to_check = int(total_frames * check_percent)
    check_indices = list(range(num_frames_to_check))
    
    print(f"Loading {num_frames_to_check} frames ({check_percent*100:.1f}%)...")
    
    # Load all frames as numpy arrays
    with tifffile.TiffFile(high_snr_path) as tif_high:
        high_frames = tif_high.asarray()[check_indices]
    with tifffile.TiffFile(low_snr_path) as tif_low:
        low_frames = tif_low.asarray()[check_indices]
    
    if crop_box is not None:
        x, y, w, h = crop_box
        high_frames = high_frames[:, y:y+h, x:x+w]
        low_frames = low_frames[:, y:y+h, x:x+w]
    
    # Convert to batched GPU tensors: [N, 1, H, W]
    high_tensor = torch.tensor(high_frames, dtype=torch.float32, device=device).unsqueeze(1)
    low_tensor = torch.tensor(low_frames, dtype=torch.float32, device=device).unsqueeze(1)
    
    print(f"GPU tensor shapes: high={high_tensor.shape}, low={low_tensor.shape}")
    
    pearsons = []
    
    # Process in batches
    for batch_start in tqdm(range(0, high_tensor.shape[0], batch_size), desc="Batches"):
        batch_end = min(batch_start + batch_size, high_tensor.shape[0])
        high_batch = high_tensor[batch_start:batch_end]
        low_batch = low_tensor[batch_start:batch_end]
        
        # BATCH NORMALIZATION (per frame in batch)
        high_min = high_batch.min(dim=2, keepdim=True)[0]
        high_min = high_min.min(dim=3, keepdim=True)[0]

        high_max = high_batch.max(dim=2, keepdim=True)[0]
        high_max = high_max.max(dim=3, keepdim=True)[0]

        low_min = low_batch.min(dim=2, keepdim=True)[0]
        low_min = low_min.min(dim=3, keepdim=True)[0]

        low_max = low_batch.max(dim=2, keepdim=True)[0]
        low_max = low_max.max(dim=3, keepdim=True)[0]
        
        high_norm = (high_batch - high_min) / (high_max - high_min + 1e-8)
        low_norm = (low_batch - low_min) / (low_max - low_min + 1e-8)
        
        # BATCH PEARSON CORRELATION
        # Flatten: [batch_size, H*W]
        high_flat = high_norm.view(high_norm.shape[0], -1)
        low_flat = low_norm.view(low_norm.shape[0], -1)
        
        # Compute means: [batch_size]
        high_mean = high_flat.mean(dim=1, keepdim=True)
        low_mean = low_flat.mean(dim=1, keepdim=True)
        
        # Centered: [batch_size, H*W]
        high_centered = high_flat - high_mean
        low_centered = low_flat - low_mean
        
        # Covariance & std: [batch_size]
        cov = (high_centered * low_centered).sum(dim=1)
        std_high = high_centered.norm(dim=1)
        std_low = low_centered.norm(dim=1)
        
        # Pearson: [batch_size]
        batch_pearsons = cov / (std_high * std_low + 1e-8)
        pearsons.extend(batch_pearsons.cpu().tolist())
    
    avg_pearson = np.mean(pearsons)
    std_pearson = np.std(pearsons)
    
    print("\n" + "="*70)
    print("🚀 BATCHED GPU PEARSON RESULTS")
    print("="*70)
    print(f"⏱️  Batches processed: {(len(pearsons)+batch_size-1)//batch_size}")
    print(f"📊 Average Pearson: {avg_pearson:.4f} ± {std_pearson:.4f}")
    print(f"📈 Min/Max: {np.min(pearsons):.4f} / {np.max(pearsons):.4f}")
    print(f"🔢 Frames: {len(pearsons)}")
    print("="*70)
    
    if avg_pearson > 0.5:
        print("✅ DATA PERFECT → Model should hit 0.8+")
    elif avg_pearson > 0.3:
        print("⚠️  DATA OK → Model can improve to 0.7+")
    else:
        print("❌ DATA PROBLEM → Fix alignment/crop first")
    
    return avg_pearson, pearsons

# Usage - 10x faster than single-frame!
if __name__ == "__main__":
    high_snr_path = "01_ZebrafishOT_GCaMP6s_492x492x6955_highSNR.tif"
    low_snr_path = "01_ZebrafishOT_GCaMP6s_492x492x6955_lowSNR.tif"
    crop_box = (182, 182, 128, 128)
    
    avg, scores = check_raw_pearson_batched_gpu(
        high_snr_path, low_snr_path,
        check_percent=0.2,  # 20% = ~1400 frames!
        batch_size=128,     # Process 128 frames AT ONCE
        crop_box=crop_box
    )
    
    np.save("batched_raw_pearsons.npy", scores)
