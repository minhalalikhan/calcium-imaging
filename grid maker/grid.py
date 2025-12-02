import os
from PIL import Image

def combine_cropped_images(folder_paths, output_path="final_result.png", max_images=8):
    """
    Combines the last 25% of the width of PNG images from 3 folders into a single image grid.
    
    Args:
        folder_paths (list[str]): List of 3 folder paths containing PNG files.
        output_path (str): Path to save the final combined PNG.
        max_images (int): Number of image rows (4–8 recommended).
    """
    assert len(folder_paths) == 3, "Exactly 3 folders must be provided."
    
    # --- Step 1: Collect PNG files from each folder ---
    all_images = []
    for folder in folder_paths:
        png_files = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(".png")]
        all_images.append(png_files[:max_images])  # take only up to max_images
    
    # --- Step 2: Crop last 25% of width for each image ---
    cropped_images = []
    for i in range(max_images):
        row_images = []
        for folder_index in range(3):
            try:
                img_path = all_images[folder_index][i]
            except IndexError:
                continue  # if a folder has fewer images, skip
                
            img = Image.open(img_path).convert("RGBA")
            w, h = img.size
            left = int(w * 0.75)
            cropped = img.crop((left, 0, w, h))
            row_images.append(cropped)
        if row_images:
            cropped_images.append(row_images)
    
    if not cropped_images:
        print("No valid images found. Check your folder paths.")
        return
    
    # --- Step 3: Determine max height and width for layout ---
    row_heights = [max(img.height for img in row) for row in cropped_images]
    row_widths = [sum(img.width for img in row) for row in cropped_images]
    
    total_height = sum(row_heights)
    max_width = max(row_widths)
    
    # --- Step 4: Create final image ---
    final_image = Image.new("RGBA", (max_width, total_height), (255, 255, 255, 255))
    
    y_offset = 0
    for row_idx, row in enumerate(cropped_images):
        x_offset = 0
        for img in row:
            # Align top of row
            final_image.paste(img, (x_offset, y_offset))
            x_offset += img.width
        y_offset += row_heights[row_idx]
    
    # --- Step 5: Save the result ---
    final_image.save(output_path)
    print(f"✅ Combined image saved as: {output_path}")


# Example usage
if __name__ == "__main__":
    folder1 = "a"
    folder2 = "b"
    folder3 = "c"
    
    combine_cropped_images([folder1, folder2, folder3], "final_result.png", max_images=6)
