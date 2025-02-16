import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# ============ SET PATHS HERE ============
json_path = r"C:\Users\Ahmed\LV-MHP-v2\train\Mask_Gender.json"
image_folder = r"C:\Users\Ahmed\LV-MHP-v2\train\images"



with open(json_path, "r") as f:
    data = json.load(f)

annos_by_image = {}
for ann in data.get("annotations", []):
    image_id = ann["image_id"]
    annos_by_image.setdefault(image_id, []).append(ann)


for img_info in data.get("images", []):
    image_id = img_info["id"]
    image_path = os.path.join(image_folder, img_info["file_name"])
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image {image_path}")
        continue


    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    overlay = image_rgb.copy()

    ann_list = annos_by_image.get(image_id, [])
    for ann in ann_list:
        bbox = ann["bbox"]
        x, y, w, h = map(int, bbox)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if "mask_path" in ann:
            mask_path = ann["mask_path"]
            if os.path.exists(mask_path):
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:

                    mask_bin = (mask_img > 0).astype(np.uint8)
                    colored_mask = np.zeros_like(image_rgb)
                    colored_mask[:, :] = (0, 255, 0)
                    alpha = 0.5
                    try:
                        overlay = np.where(mask_bin[..., None].astype(bool),
                                           cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0),
                                           overlay)
                    except:
                        print(image_path)

            else:
                print(f"Warning: mask path {mask_path} does not exist.")

        if "category_id" in ann:
            cv2.putText(overlay, str(ann["category_id"]), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    plt.figure(figsize=(10, 8))
    plt.imshow(overlay)
    plt.title(f"Image ID: {image_id}")
    plt.axis("off")
    plt.show()
