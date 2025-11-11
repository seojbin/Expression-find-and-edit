import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import json
import time
import copy

#데이터셋 설정
class EmotionDataset(Dataset):
    # 감정(expression) 라벨을 정수(integer)로 변환하는 맵
    # 7가지 감정 기준
    CLASS_TO_IDX = {
        "Joy": 0,
        "Neutral": 1,
        "Surprise": 2,
        "Disgust": 3,
        "Sadness": 4,
        "Anger": 5,
        "Fear": 6
    }
    
    IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
    NUM_CLASSES = 7

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = [] # (이미지 경로, 라벨 인덱스) 쌍을 저장할 리스트

        label_base_path = os.path.join(root_dir, '02.라벨링데이터', 'TL')
        image_base_path = os.path.join(root_dir, '01.원천데이터', 'TS', 'image')

        print(f"[{root_dir}] 데이터셋 로드 시작")

        if not os.path.exists(label_base_path):
            print(f"오류: 라벨 경로를 찾을 수 없음: {label_base_path}")
            return

        for person_id_folder in os.listdir(label_base_path):
            person_json_dir = os.path.join(label_base_path, person_id_folder, 'json')
            person_image_dir = os.path.join(image_base_path, person_id_folder)

            if os.path.isdir(person_json_dir) and os.path.isdir(person_image_dir):
                for json_filename in os.listdir(person_json_dir):
                    if json_filename.endswith('.json'):
                        file_stem = os.path.splitext(json_filename)[0]
                        
                        image_filename = file_stem + '.png' 
                        image_path = os.path.join(person_image_dir, image_filename)
                        json_path = os.path.join(person_json_dir, json_filename)

                        if os.path.exists(image_path):
                            try:
                                with open(json_path, 'r', encoding='utf-8') as f:
                                    label_data = json.load(f)
                                
                                expression_label = label_data.get('expression')
                                
                                if expression_label in self.CLASS_TO_IDX:
                                    label_idx = self.CLASS_TO_IDX[expression_label]
                                    self.samples.append((image_path, label_idx))
                                else:
                                    print(f"Warning: {json_path} 에서 알 수 없는 라벨 '{expression_label}' 스킵")
                            
                            except Exception as e:
                                print(f"Warning: {json_path} 파일 처리 중 오류 발생: {e}")

        if not self.samples:
            print(f"오류: [{root_dir}] 경로에서 유효한 (이미지, 라벨) 쌍을 찾지 못했습니다. 경로 확인")
        else:
            print(f"[{root_dir}] 데이터셋 로드 완료. 총 {len(self.samples)} 개 샘플 발견.")


    def __len__(self):
        # 전체 데이터셋의 샘플 수를 반환
        return len(self.samples)

    def __getitem__(self, idx):
        # idx 번째 샘플을 반환.
        image_path, label = self.samples[idx]

        try:
            # 이미지 로드
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"오류: {image_path} 이미지 로드 실패: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0)) 
            
        # 전처리 및 Augmentation 적용
        if self.transform:
            image = self.transform(image)
            
        return image, label


def create_emotion_model(num_classes=7):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Fully Connected Layer 교체
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

if __name__ == '__main__':
    
    #하이퍼파라미터
    BATCH_SIZE = 32
    NUM_EPOCHS = 50 
    LEARNING_RATE = 0.001
    
    # 데이터 경로
    TRAIN_ROOT_DIR = 'Training'
    VAL_ROOT_DIR = 'Validation'
    
    # GPU 사용 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    #데이터 전처리
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(), 
            transforms.RandomRotation(10),     
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    #데이터셋 및 데이터로더 생성
    try:
        train_dataset = EmotionDataset(root_dir=TRAIN_ROOT_DIR, transform=data_transforms['train'])
        val_dataset = EmotionDataset(root_dir=VAL_ROOT_DIR, transform=data_transforms['val'])
        
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError("데이터셋 로드 실패")

        # NumPy < 2.0 버전 충돌 X -> num_workers=4, 
        #충돌시  num_workers=0 으로 유지
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        
        dataloaders = {'train': train_loader, 'val': val_loader}
        dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    except Exception as e:
        print(f"데이터로더 생성 중 치명적 오류 발생: {e}")
        exit()


    #모델, 손실 함수, 옵티마이저
    model = create_emotion_model(num_classes=EmotionDataset.NUM_CLASSES).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) #스케쥴러 필요하면 킴

    #Early Stopping
    patience = 2  # 2회 연속 val_loss가 개선되지 않음 중단
    patience_counter = 0
    best_val_loss = float('inf') # 최소 손실값을 저장하기 위해 무한대로 초기화
    BEST_MODEL_SAVE_PATH = 'best_emotion_classifier.pth' # 베스트 모델 저장 경로
    early_stop = False # 학습 중단 플래그

    print("\n--- 학습 시작 ---")
    
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        if early_stop: # Early Stop 플래그가 True 루프 탈출
            break
            
        print(f'Epoch {epoch + 1}/{NUM_EPOCHS}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1) 
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            # if phase == 'train' and scheduler:
            #     scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Early Stopping 로직 (val phase 종료 후 실행)
            if phase == 'val':
                if epoch_loss < best_val_loss:
                    # val_loss가 개선된 경우
                    print(f"  Val loss 개선 ({best_val_loss:.4f} -> {epoch_loss:.4f}). 모델 저장: {BEST_MODEL_SAVE_PATH}")
                    best_val_loss = epoch_loss
                    patience_counter = 0
                    # 가장 좋은 모델의 가중치를 저장
                    torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH)
                else:
                    # val_loss가 개선되지 않은 경우
                    patience_counter += 1
                    print(f"  Val loss 개선 없음. Early Stopping 카운트: {patience_counter}/{patience}")
                    if patience_counter >= patience:
                        # 2회에 도달하면 학습 중단
                        print(f"Early Stopping: {patience}회 연속 성능 개선이 없어 학습을 중단")
                        early_stop = True

        print()

    time_elapsed = time.time() - start_time
    print(f'학습 완료 (소요 시간: {time_elapsed // 60:.0f}분 {time_elapsed % 60:.0f}초)')
    
    #Early Stopping 결과 출력
    if best_val_loss == float('inf'):
         print("모델이 저장되지 않았습니다.")
    else:
         print(f"가장 성능이 좋았던 모델이 '{BEST_MODEL_SAVE_PATH}' 에 저장되었습니다 (최저 Val Loss: {best_val_loss:.4f}).")
