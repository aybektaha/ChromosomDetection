import cv2
import numpy as np
import os
from pathlib import Path

def read_chromosome_data(txt_file):
    chromosomes = []
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip().split()
            class_label = int(data[0])
            rotation_angle = float(data[1])
            
            coords = list(map(float, data[2:]))
            x_coords = coords[::2]
            y_coords = coords[1::2]
            
            chromosomes.append({
                'class': class_label,
                'angle': rotation_angle,
                'x_coords': x_coords,
                'y_coords': y_coords
            })
    return chromosomes

def create_annotation_image(image, chromosomes, img_width, img_height):
    # Orijinal görüntüyü kopyala
    annotation_img = image.copy()
    
    # Her kromozom için
    for i, chromosome in enumerate(chromosomes):
        # Koordinatları piksel değerlerine dönüştür
        x_pixels = (np.array(chromosome['x_coords']) * img_width).astype(np.int32)
        y_pixels = (np.array(chromosome['y_coords']) * img_height).astype(np.int32)
        
        # Kromozom sınırlarını çiz
        points = np.column_stack((x_pixels, y_pixels))
        cv2.polylines(annotation_img, [points], True, (0, 255, 0), 2)
        
        # Kromozom numarasını ve sınıfını yaz
        center_x = int(np.mean(x_pixels))
        center_y = int(np.mean(y_pixels))
        cv2.putText(annotation_img, f"#{i+1} C{chromosome['class']}", 
                    (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 0, 0), 2)
    
    return annotation_img

def process_chromosome(image, chromosome_data, img_width, img_height, rotate=True):
    # 1. İlk kırpma işlemi için koordinatları hazırla
    x_pixels = (np.array(chromosome_data['x_coords']) * img_width).astype(np.int32)
    y_pixels = (np.array(chromosome_data['y_coords']) * img_height).astype(np.int32)
    
    # Sınırları belirle
    margin = 0
    x_min = max(0, np.min(x_pixels) - margin)
    x_max = min(img_width, np.max(x_pixels) + margin)
    y_min = max(0, np.min(y_pixels) - margin)
    y_max = min(img_height, np.max(y_pixels) + margin)
    
    # İlk kırpma
    cropped = image[y_min:y_max, x_min:x_max].copy()
    
    # 2. Maskeleme işlemi
    mask = np.zeros(cropped.shape[:2], dtype=np.uint8)
    points = np.column_stack((x_pixels - x_min, y_pixels - y_min))
    cv2.fillPoly(mask, [points], 255)
    
    # Arka planı beyaz yap
    result = np.ones_like(cropped) * 255
    result[mask == 255] = cropped[mask == 255]
    
    # 3. Görüntüyü döndür
    if rotate and chromosome_data['angle'] != 0:
        # Görüntü merkezini hesapla
        center = (result.shape[1] // 2, result.shape[0] // 2)
        
        # Döndürme matrisini oluştur
        rotation_matrix = cv2.getRotationMatrix2D(center, +chromosome_data['angle'], 1.0)
        
        # Döndürülmüş görüntünün boyutlarını hesapla
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_width = int((result.shape[0] * sin) + (result.shape[1] * cos))
        new_height = int((result.shape[0] * cos) + (result.shape[1] * sin))
        
        # Döndürme matrisini yeni merkeze göre ayarla
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Görüntüyü döndür
        rotated = cv2.warpAffine(result, rotation_matrix, (new_width, new_height),
                                borderValue=(255, 255, 255))
        
        # 4. Minimum kapsayan dikdörtgene göre kırp
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        coords = cv2.findNonZero(cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)[1])
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            final_cropped = rotated[y:y+h, x:x+w]
            return final_cropped
        
        return rotated
    
    return result

def main():
    # Görüntü ve annotation dosyası yolları
    image_path = "mytestdata/C6-0033_01.jpg"
    txt_file = "mytestdata/C6-0033_01.txt"
    
    # Görüntüyü oku
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    img_height, img_width = image.shape[:2]
    chromosomes = read_chromosome_data(txt_file)
    
    # Annotation görüntüsünü oluştur ve kaydet
    annotation_img = create_annotation_image(image, chromosomes, img_width, img_height)
    output_annotation_path = os.path.splitext(image_path)[0] + "_annotation.jpg"
    cv2.imwrite(output_annotation_path, annotation_img)
    
    # Output klasörlerini oluştur
    output_base = "output_chromosomes"
    output_base_rotated = "output_chromosomes_rotated"
    os.makedirs(output_base, exist_ok=True)
    os.makedirs(output_base_rotated, exist_ok=True)
    
    # Her kromozomu işle
    for i, chromosome in enumerate(chromosomes):
        # Klasörleri oluştur
        class_dir = os.path.join(output_base, f"class_{chromosome['class']}")
        class_dir_rotated = os.path.join(output_base_rotated, f"class_{chromosome['class']}")
        os.makedirs(class_dir, exist_ok=True)
        os.makedirs(class_dir_rotated, exist_ok=True)
        
        # Orijinal kırpılmış görüntüyü işle ve kaydet
        original_cropped = process_chromosome(image, chromosome, img_width, img_height, rotate=False)
        if original_cropped is not None:
            output_path = os.path.join(class_dir, f"chromosome_{i}.jpg")
            cv2.imwrite(output_path, original_cropped)
        
        # Döndürülmüş görüntüyü işle ve kaydet
        rotated_chromosome = process_chromosome(image, chromosome, img_width, img_height, rotate=True)
        if rotated_chromosome is not None:
            output_path_rotated = os.path.join(class_dir_rotated, f"chromosome_{i}.jpg")
            cv2.imwrite(output_path_rotated, rotated_chromosome)

if __name__ == "__main__":
    main()



