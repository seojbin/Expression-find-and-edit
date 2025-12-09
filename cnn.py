import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image, ImageFile
import os
import json
import time
from ultralytics import YOLO
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
import shutil

ImageFile.LOAD_TRUNCATED_IMAGES = True

# 설정
ORIGINAL_TRAIN_DIR = 'Training'
ORIGINAL_VAL_DIR = 'Validation'
PROCESSED_ROOT_DIR = 'Face_Data'

CLASS_TO_IDX = {
    "Joy": 0, "Neutral": 1, "Surprise": 2, "Disgust": 3,
    "Sadness": 4, "Anger": 5, "Fear": 6
}
NUM_CLASSES = 7

def preprocess_data(source_root, target_root, split_name):
    save_dir = os.path.join(target_root, split_name)
    
    if os.path.exists(save_dir):
        print(f"[{split_name}] Preprocessed data exists. Skipping.")
        return

    print(f"[{split_name}] Starting preprocessing...")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    yolo = YOLO('yolov8m-face.pt').to(device)
    
    label_base = os.path.join(source_root, '02.라벨링데이터', 'TL')
    image_base = os.path.join(source_root, '01.원천데이터', 'TS', 'image')
    
    if not os.path.exists(label_base):
        print(f"Source path not found: {label_base}")
        return

    for emotion in CLASS_TO_IDX.keys():
        os.makedirs(os.path.join(save_dir, emotion), exist_ok=True)

    processed_count = 0
    
    all_files = []
    for root, dirs, files in os.walk(label_base):
        for file in files:
            if file.endswith('.json'):
                all_files.append(os.path.join(root, file))

    for json_path in tqdm(all_files, desc=f"Processing {split_name}"):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            emotion = data.get('expression')
            if emotion not in CLASS_TO_IDX:
                continue

            rel_path = os.path.relpath(os.path.dirname(os.path.dirname(json_path)), label_base)
            img_filename = os.path.splitext(os.path.basename(json_path))[0]
            
            possible_exts = ['.jpg', '.png', '.jpeg', '.JPG']
            src_img_path = None
            
            target_img_folder = os.path.join(image_base, rel_path)
            
            for ext in possible_exts:
                temp = os.path.join(target_img_folder, img_filename + ext)
                if os.path.exists(temp):
                    src_img_path = temp
                    break
            
            if not src_img_path:
                continue

            img = Image.open(src_img_path).convert('RGB')
            results = yolo(img, verbose=False)
            
            best_box = None
            max_conf = -1.0
            
            for result in results:
                if len(result.boxes) > 0:
                    confs = result.boxes.conf.cpu().numpy()
                    best_idx = confs.argmax()
                    if confs[best_idx] > max_conf:
                        max_conf = confs[best_idx]
                        best_box = result.boxes.xyxy[best_idx].cpu().numpy().astype(int)
            
            if best_box is not None:
                x1, y1, x2, y2 = best_box
                w, h = img.size
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                face_img = img.crop((x1, y1, x2, y2))
                
                # ROTATE
                face_img = face_img.transpose(Image.ROTATE_90)
                
                face_img = face_img.resize((224, 224))
                
                save_path = os.path.join(save_dir, emotion, os.path.basename(src_img_path))
                face_img.save(save_path)
                processed_count += 1

        except Exception:
            continue

    print(f"[{split_name}] Processed: {processed_count}")


class SimpleEmotionDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.samples = []
        
        for emotion, label_idx in CLASS_TO_IDX.items():
            emotion_dir = os.path.join(self.root_dir, emotion)
            if not os.path.exists(emotion_dir):
                continue
                
            for file in os.listdir(emotion_dir):
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.samples.append((os.path.join(emotion_dir, file), label_idx))
        
        print(f"[{split}] Loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception:
            return torch.zeros((3, 224, 224)), label

def create_model(num_classes=7):
    model = EfficientNet.from_pretrained('efficientnet-b0')
    in_features = model._fc.in_features
    model._fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )
    return model

if __name__ == '__main__':
    
    preprocess_data(ORIGINAL_TRAIN_DIR, PROCESSED_ROOT_DIR, 'Training')
    preprocess_data(ORIGINAL_VAL_DIR, PROCESSED_ROOT_DIR, 'Validation')
    
    BATCH_SIZE = 64
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.0005
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(f"Training on {DEVICE}")

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10), # 작은 각도 변화만 허용 (90도 같은 큰 변화는 불가)
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = SimpleEmotionDataset(PROCESSED_ROOT_DIR, 'Training', transform=train_transforms)
    val_dataset = SimpleEmotionDataset(PROCESSED_ROOT_DIR, 'Validation', transform=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = create_model(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2)

    best_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for inputs, labels in loop:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        
        model.eval()
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
        
        val_acc = val_corrects.double() / len(val_dataset)
        
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'emotion_model.pth')
            print(f"Saved Best Model: {best_acc:.4f}")

    print("Finished")