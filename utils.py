import cv2
import numpy as np
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import os
from pathlib import Path

def process_chromosome_image(image_path, txt_path, model_path, output_base="output"):
    """
    Complete chromosome processing pipeline in a single function.
    
    Args:
        image_path (str): Path to the input image
        txt_path (str): Path to the input text file with chromosome data
        model_path (str): Path to the trained model checkpoint
        output_base (str): Base directory for outputs
    """
    # Index to class mapping
    INDEX_TO_CLASS = {
        0: 1, 1: 2, 2: 11, 3: 12, 4: 13, 5: 14, 6: 15, 7: 16,
        8: 17, 9: 18, 10: 19, 11: 20, 12: 3, 13: 21, 14: 22,
        15: 'x', 16: 'y', 17: 4, 18: 5, 19: 6, 20: 7, 21: 8,
        22: 9, 23: 10
    }
    
    # Create output directory
    os.makedirs(output_base, exist_ok=True)
    
    # Load and setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 24)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)
    
    # Image transform for model
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Error: Could not read image file {image_path}")
    
    img_height, img_width = image.shape[:2]
    
    # Read chromosome data from text file
    chromosomes = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip().split()
            class_label = int(data[0])
            rotation_angle = float(data[1])
            coords = list(map(float, data[2:]))
            chromosomes.append({
                'class': class_label,
                'angle': rotation_angle,
                'x_coords': coords[::2],
                'y_coords': coords[1::2]
            })
    
    if not chromosomes:
        raise ValueError(f"Error: No chromosome data found in {txt_path}")
    
    # Process each chromosome
    updated_chromosomes = []
    chromosomes_dict = dict()
    for i, chromosome in enumerate(chromosomes):
        # Extract coordinates
        x_pixels = (np.array(chromosome['x_coords']) * img_width).astype(np.int32)
        y_pixels = (np.array(chromosome['y_coords']) * img_height).astype(np.int32)
        
        # Crop chromosome
        margin = 0
        x_min = max(0, np.min(x_pixels) - margin)
        x_max = min(img_width, np.max(x_pixels) + margin)
        y_min = max(0, np.min(y_pixels) - margin)
        y_max = min(img_height, np.max(y_pixels) + margin)
        
        cropped = image[y_min:y_max, x_min:x_max].copy()
        
        # Create mask and apply
        mask = np.zeros(cropped.shape[:2], dtype=np.uint8)
        points = np.column_stack((x_pixels - x_min, y_pixels - y_min))
        cv2.fillPoly(mask, [points], 255)
        
        result = np.ones_like(cropped) * 255
        result[mask == 255] = cropped[mask == 255]
        
        # Rotate chromosome
        if chromosome['angle'] != 0:
            center = (result.shape[1] // 2, result.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, chromosome['angle'], 1.0)
            
            cos = np.abs(rotation_matrix[0, 0])
            sin = np.abs(rotation_matrix[0, 1])
            new_width = int((result.shape[0] * sin) + (result.shape[1] * cos))
            new_height = int((result.shape[0] * cos) + (result.shape[1] * sin))
            
            rotation_matrix[0, 2] += (new_width / 2) - center[0]
            rotation_matrix[1, 2] += (new_height / 2) - center[1]
            
            rotated = cv2.warpAffine(result, rotation_matrix, (new_width, new_height),
                                   borderValue=(255, 255, 255))
            
            gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
            coords = cv2.findNonZero(cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)[1])
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                rotated_chromosome = rotated[y:y+h, x:x+w]
            else:
                rotated_chromosome = rotated
        else:
            rotated_chromosome = result
        
        if rotated_chromosome is not None:
            # Save temporary image for prediction
            temp_path = os.path.join(output_base, f"temp_chromosome_{i}.jpg")
            cv2.imwrite(temp_path, rotated_chromosome)
            
            # Predict class
            try:
                # Load and transform image for prediction
                image_pil = Image.open(temp_path).convert('RGB')
                image_tensor = transform(image_pil).unsqueeze(0)
                
                # Make prediction
                with torch.no_grad():
                    outputs = model(image_tensor)
                    _, predicted_index = torch.max(outputs, 1)
                    predicted_class = INDEX_TO_CLASS[predicted_index.item()]
                
                # Update chromosome data
                old_class = chromosome['class']
                chromosome['class'] = predicted_class
                print("p:", predicted_class, "c:", old_class)

                
                updated_chromosomes.append(chromosome)
                
                # Save chromosome to class directory
                #class_dir = os.path.join(output_base, f"class_{predicted_class}")
                #os.makedirs(class_dir, exist_ok=True)
                # output_path = os.path.join(class_dir, f"{Path(image_path).stem}_chromosome_{i}.jpg")
                output_path =os.path.join(output_base, f"{Path(image_path).stem}_chromosome_{i}.jpg") 
                cv2.imwrite(output_path, rotated_chromosome)
                if str(predicted_class) not in chromosomes_dict:
                    chromosomes_dict[str(predicted_class)] = [output_path]
                else:
                    chromosomes_dict[str(predicted_class)].append(output_path)
            
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
    
    # Update text file with new predictions
    with open(txt_path, 'w') as f:
        for chromosome in updated_chromosomes:
            coords = []
            for x, y in zip(chromosome['x_coords'], chromosome['y_coords']):
                coords.extend([str(x), str(y)])
            line = f"{chromosome['class']} {chromosome['angle']} {' '.join(coords)}\n"
            f.write(line)
    
    # Create annotation image
    annotation_img = image.copy()
    for chromosome in updated_chromosomes:
        x_pixels = (np.array(chromosome['x_coords']) * img_width).astype(np.int32)
        y_pixels = (np.array(chromosome['y_coords']) * img_height).astype(np.int32)
        
        points = np.column_stack((x_pixels, y_pixels))
        cv2.polylines(annotation_img, [points], True, (0, 255, 0), 1)
        
        center_x = int(np.mean(x_pixels))
        center_y = int(np.mean(y_pixels))
        cv2.putText(annotation_img, f"C{chromosome['class']}", 
                    (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 0, 0), 2)
        cv2.putText(annotation_img, f"C{chromosome['class']}", 
                    (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 1)
    
    # Save annotation image
    annotation_path = os.path.join(output_base, f"{Path(image_path).stem}_annotation.jpg")
    cv2.imwrite(annotation_path, annotation_img)
    
    return chromosomes_dict, annotation_path

# Example usage:
if __name__ == "__main__":
    try:
        image_path = "mytestdata/C6-0033_01.jpg"
        txt_path = "mytestdata/C6-0033_01.txt"
        model_path = "model_checkpoints/mobilenetv2_best.pth"
        output_base = "output"
        
        karyotype_table, annotation_path = process_chromosome_image(
            image_path, txt_path, model_path, output_base
        )
        print(f"Processing completed successfully!")
        print(f"Annotation image saved to: {annotation_path}")
        print(f"Updated chromosomes saved to their respective class folders in: {output_base}")
        print(karyotype_table)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
