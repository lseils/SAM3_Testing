import os
import glob
import cv2
import numpy as np
from ultralytics import SAM

def mask_images_with_sam3(input_folder, output_folder):
    # 1. Load Meta's SAM 3 model via Ultralytics (it will auto-download weights)
    print("Loading SAM 3 model...")
    model = SAM("sam3_b.pt") 
    
    image_paths = glob.glob(os.path.join(input_folder, "*.jpg"))
    if not image_paths:
        print(f"No images found in {input_folder}")
        return
        
    os.makedirs(output_folder, exist_ok=True)
    print(f"Processing {len(image_paths)} images with SAM 3...")
    
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        print(f" -> Masking {filename}...")
        
        # 2. Use SAM 3's text prompting to isolate the building
        results = model.predict(img_path, prompt="building facade")
        
        # 3. Read the original image
        img = cv2.imread(img_path)
        
        # Ensure SAM 3 successfully found something matching the prompt
        if results and results[0].masks is not None:
            # Extract the AI's mask data
            mask_tensor = results[0].masks.data.cpu().numpy()
            
            # If it found multiple disconnected pieces of a building, combine them
            combined_mask = np.max(mask_tensor, axis=0)
            
            # Resize mask to perfectly match the original image resolution
            combined_mask = cv2.resize(combined_mask, (img.shape[1], img.shape[0]))
            
            # Convert to a binary (0 or 1) boolean mask
            binary_mask = combined_mask > 0.5
            
            # Create an alpha (transparency) channel 
            # Building pixels become 255 (solid), everything else 0 (invisible)
            alpha_channel = (binary_mask * 255).astype(np.uint8)
            
            # Add the transparent background to the original image
            b, g, r = cv2.split(img)
            img_BGRA = cv2.merge((b, g, r, alpha_channel))
            
            # Save as a PNG (JPGs do not support transparency!)
            save_path = os.path.join(output_folder, filename.replace(".jpg", ".png"))
            cv2.imwrite(save_path, img_BGRA)
        else:
            print(f"    [!] No building detected in {filename}")

if __name__ == "__main__":
    # Your Google script saves here:
    INPUT_DIR = "street_images"
    # SAM 3 will save the clean transparent images here:
    OUTPUT_DIR = "masked_buildings"
    
    mask_images_with_sam3(INPUT_DIR, OUTPUT_DIR)
    print(f"\nDone! Transparent images saved to: {OUTPUT_DIR}")
    