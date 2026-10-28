import cv2
import numpy as np
import os
import random

def prepare_dirs():
    dirs = [
        'dataset/images/train', 'dataset/images/val',
        'dataset/labels/train', 'dataset/labels/val'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def generate_sample(img_id, split='train'):
    img_size = 640
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 220  # Gray metallic background
    
    # Draw metal component body
    cv2.rectangle(img, (100, 100), (540, 540), (180, 180, 180), -1)
    cv2.rectangle(img, (100, 100), (540, 540), (100, 100, 100), 4)

    has_defect = random.random() < 0.7
    labels = []

    if has_defect:
        num_defects = random.randint(1, 3)
        for _ in range(num_defects):
            defect_type = random.choice([0, 1])  # 0: Scratch, 1: Spot
            
            x1 = random.randint(150, 450)
            y1 = random.randint(150, 450)
            
            if defect_type == 0:  # Scratch
                x2 = x1 + random.randint(30, 80)
                y2 = y1 + random.randint(30, 80)
                cv2.line(img, (x1, y1), (x2, y2), (20, 20, 20), 3)
            else:  # Spot / Contamination
                r = random.randint(15, 35)
                cv2.circle(img, (x1, y1), r, (40, 40, 40), -1)
                x2, y2 = x1 + r, y1 + r
                x1, y1 = x1 - r, y1 - r

            # YOLO Normalized Bounding Box
            xc = ((x1 + x2) / 2.0) / img_size
            yc = ((y1 + y2) / 2.0) / img_size
            w = (abs(x2 - x1)) / img_size
            h = (abs(y2 - y1)) / img_size
            
            labels.append(f"{defect_type} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    # Save Image and Label
    cv2.imwrite(f"dataset/images/{split}/sample_{img_id:04d}.jpg", img)
    with open(f"dataset/labels/{split}/sample_{img_id:04d}.txt", "w") as f:
        f.write("\n".join(labels))

if __name__ == "__main__":
    prepare_dirs()
    print("Generating synthetic industrial defect dataset...")
    for i in range(100):
        generate_sample(i, split='train')
    for i in range(25):
        generate_sample(i, split='val')
    print("Dataset generation completed successfully!")