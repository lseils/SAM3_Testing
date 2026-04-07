import os
import glob
import cv2
import numpy as np
from PIL import Image
from lang_sam import LangSAM

def mask_balconies(input_folder, output_folder):
    # 1. Load LangSAM (This downloads Grounding DINO and SAM automatically)
    print("Loading LangSAM (Grounding DINO + SAM)...")
    model = LangSAM()

    image_paths = glob.glob(os.path.join(input_folder, "*.jpg"))
    if not image_paths:
        print(f"No images found in {input_folder}")
        return

    os.makedirs(output_folder, exist_ok=True)
    print(f"Processing {len(image_paths)} images...")

    # The text we want to search for
    text_prompt = "balcony"

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        print(f"-> Masking {filename}...")

        # LangSAM requires PIL Images
        image_pil = Image.open(img_path).convert("RGB")

        # 2. Predict: This returns masks, bounding boxes, labels, and confidence scores
        masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)

        # 3. Process and save the results
        if len(masks) == 0:
            print(f"   No balconies found in {filename}.")
            continue

        # Convert original image to OpenCV format (BGR) for saving later
        image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        # Create an empty black image to hold all our combined masks
        combined_mask = np.zeros((image_pil.height, image_pil.width), dtype=np.uint8)

        # Loop through all detected balconies and combine their masks
        for mask in masks:
            # Convert PyTorch tensor mask to numpy array
            mask_np = mask.cpu().numpy().astype(np.uint8) * 255
            # Combine
            combined_mask = cv2.bitwise_or(combined_mask, mask_np)

        # Save the black & white mask image
        mask_output_path = os.path.join(output_folder, f"mask_{filename}")
        cv2.imwrite(mask_output_path, combined_mask)

        # Optional: Save an overlay image to visualize the result
        overlay = image_cv2.copy()
        overlay[combined_mask == 255] = [0, 0, 255] # Turn balconies red
        blended = cv2.addWeighted(image_cv2, 0.7, overlay, 0.3, 0)
        overlay_output_path = os.path.join(output_folder, f"overlay_{filename}")
        cv2.imwrite(overlay_output_path, blended)

        print(f"   Saved masks for {len(masks)} balconies.")

# To run the script:
if __name__ == "__main__":
    input_dir = "street_images" # Update this if needed
    output_dir = "masked_images"
    mask_balconies(input_dir, output_dir)