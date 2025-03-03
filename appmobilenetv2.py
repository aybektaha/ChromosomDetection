import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

# Index'ten gerçek sınıf numarasına eşleştirme
INDEX_TO_CLASS = {
    0: 0,    # Sınıf 0
    1: 1,    # Sınıf 1
    2: 10,   # Sınıf 10
    3: 11,   # Sınıf 11
    4: 12,   # Sınıf 12
    5: 13,   # Sınıf 13
    6: 14,   # Sınıf 14
    7: 15,   # Sınıf 15
    8: 16,   # Sınıf 16
    9: 17,   # Sınıf 17
    10: 18,  # Sınıf 18
    11: 19,  # Sınıf 19
    12: 2,   # Sınıf 2
    13: 20,  # Sınıf 20
    14: 21,  # Sınıf 21
    15: 22,  # Sınıf 22
    16: 23,  # Sınıf 23
    17: 3,   # Sınıf 3
    18: 4,   # Sınıf 4
    19: 5,   # Sınıf 5
    20: 6,   # Sınıf 6
    21: 7,   # Sınıf 7
    22: 8,   # Sınıf 8
    23: 9    # Sınıf 9
}

def load_model(model_path, num_classes):
    # MobileNetV2 modelini oluştur
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    # Kaydedilmiş model ağırlıklarını yükle
    checkpoint = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Değerlendirme moduna al
    model.eval()
    return model

def predict_image(model, image_path):
    # Görüntü ön işleme
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Görüntüyü yükle ve dönüştür
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Batch boyutu ekle
    
    # Tahmin yap
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_index = torch.max(outputs, 1)
        
        # Index'i gerçek sınıf numarasına dönüştür
        predicted_class = INDEX_TO_CLASS[predicted_index.item()]
        
    return predicted_index.item(), predicted_class

# Kullanım örneği
def main():
    # Model parametreleri
    model_path = "model_checkpoints/mobilenetv2_best.pth"
    num_classes = 24  # Modelinizin sınıf sayısı
    
    # Cihazı belirle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")
    
    # Modeli yükle
    model = load_model(model_path, num_classes)
    model = model.to(device)
    
    # Örnek bir görüntü üzerinde tahmin yap
    image_path = "output_chromosomes_rotated/class_0/C6-0033_01_chromosome_4.jpg"  # Test edilecek görüntünün yolu
    predicted_index, predicted_class = predict_image(model, image_path)
    print(f"Tahmin edilen class: {predicted_class}")

if __name__ == "__main__":
    main()