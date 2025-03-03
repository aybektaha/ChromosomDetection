import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd

class ChromosomeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.class_mapping = {}  # Sınıf isimlerini indekslere eşleyen sözlük
        
        # Sınıf klasörlerini bul ve sırala
        class_dirs = sorted(list(self.root_dir.glob("class_*")))
        
        # Sınıf mapping'ini oluştur
        for idx, class_dir in enumerate(class_dirs):
            class_name = int(class_dir.name.split("_")[1])
            self.class_mapping[class_name] = idx
            print(f"Sınıf {class_name} -> İndeks {idx}")
        
        # Görüntüleri topla
        for class_dir in class_dirs:
            class_name = int(class_dir.name.split("_")[1])
            class_idx = self.class_mapping[class_name]
            
            for img_path in class_dir.glob("*.jpg"):
                self.samples.append((str(img_path), class_idx))
        
        print(f"Toplam {len(self.samples)} görüntü yüklendi")
        print(f"Toplam {len(self.class_mapping)} sınıf bulundu")
        
        # Sınıf dağılımını kontrol et
        class_counts = {}
        for _, label in self.samples:
            class_counts[label] = class_counts.get(label, 0) + 1
        print("\nSınıf dağılımı:")
        for label, count in sorted(class_counts.items()):
            print(f"İndeks {label}: {count} görüntü")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Etiketin geçerli aralıkta olduğunu kontrol et
        assert 0 <= label < len(self.class_mapping), f"Geçersiz etiket: {label}"
        return image, label

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, checkpoint_path):
    best_val_acc = 0.0
    metrics_df = pd.DataFrame(columns=['Epoch', 'Train_Loss', 'Train_Precision', 'Train_Recall', 
                                       'Train_Accuracy', 'Train_F1', 'Val_Precision', 'Val_Recall',
                                       'Val_Accuracy', 'Val_F1'])
    print(device)
    for epoch in range(num_epochs):
        print(epoch)
        # Training phase
        model.train()
        running_loss = 0.0
        all_train_preds = []
        all_train_labels = []
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        # Training metrics
        train_precision = precision_score(all_train_labels, all_train_preds, average='weighted')
        train_recall = recall_score(all_train_labels, all_train_preds, average='weighted')
        train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted')
        train_acc = accuracy_score(all_train_labels, all_train_preds)

        # Validation phase
        model.eval()
        all_val_preds = []
        all_val_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
        
        val_precision = precision_score(all_val_labels, all_val_preds, average='weighted')
        val_recall = recall_score(all_val_labels, all_val_preds, average='weighted')
        val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')
        val_acc = accuracy_score(all_val_labels, all_val_preds)

        # Save metrics to file
        with open("results.txt", "a") as f:
            f.write(f"Epoch: {epoch+1}/{num_epochs}\n")
            f.write(f"Train Loss: {running_loss / len(train_loader):.4f}\n")
            f.write(f"Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}, Train Accuracy: {train_acc:.4f}\n")
            f.write(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}, Val Accuracy: {val_acc:.4f}\n")
            f.write("\n")

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
            }, checkpoint_path)
    
    return metrics_df, best_val_acc

def main():
    # Parametreler
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {DEVICE}")
    
    # Veri dönüşümleri
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Veri setini yükle
    dataset = ChromosomeDataset("output_chromosomes_rotated", transform=transform)
    num_classes = len(dataset.class_mapping)
    print(f"\nToplam sınıf sayısı: {num_classes}")
    
    # Eğitim ve doğrulama setlerini ayır
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Model 1: MobileNetV2
    model1 = models.mobilenet_v2(pretrained=True)
    model1.classifier[1] = nn.Linear(model1.classifier[1].in_features, num_classes)
    model1 = model1.to(DEVICE)
    optimizer1 = optim.Adam(model1.parameters(), lr=LEARNING_RATE)
    
    # Model 2: ResNet18
    model2 = models.resnet18(pretrained=True)
    model2.fc = nn.Linear(model2.fc.in_features, num_classes)
    model2 = model2.to(DEVICE)
    optimizer2 = optim.Adam(model2.parameters(), lr=LEARNING_RATE)
    
    criterion = nn.CrossEntropyLoss()
    
    # Checkpoint dizini oluştur
    os.makedirs("model_checkpoints", exist_ok=True)
    
    # Model 1'i eğit
    print("\nEğitim başlıyor: MobileNetV2")
    metrics_df1, best_acc1 = train_model(model1, train_loader, val_loader, criterion, 
                                       optimizer1, NUM_EPOCHS, DEVICE, 
                                       "model_checkpoints/mobilenetv2_best.pth")
    metrics_df1.to_csv("mobilenetv2_metrics.csv", index=False)
    print("Model1 ok")
    # Model 2'yi eğit
    print("\nEğitim başlıyor: ResNet18")
    metrics_df2, best_acc2 = train_model(model2, train_loader, val_loader, criterion, 
                                       optimizer2, NUM_EPOCHS, DEVICE, 
                                       "model_checkpoints/resnet18_best.pth")
    metrics_df2.to_csv("resnet18_metrics.csv", index=False)
    
    # En iyi modeli seç
    if best_acc1 > best_acc2:
        print("\nEn iyi model: MobileNetV2")
        best_model_path = "model_checkpoints/mobilenetv2_best.pth"
        best_metrics_path = "mobilenetv2_metrics.csv"
    else:
        print("\nEn iyi model: ResNet18")
        best_model_path = "model_checkpoints/resnet18_best.pth"
        best_metrics_path = "resnet18_metrics.csv"
    
    print(f"MobileNetV2 en iyi doğruluk: {best_acc1:.4f}")
    print(f"ResNet18 en iyi doğruluk: {best_acc2:.4f}")
    print(f"En iyi model kaydedildi: {best_model_path}")
    print(f"Metrikler kaydedildi: {best_metrics_path}")

if __name__ == "__main__":
    main()
