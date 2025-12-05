import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2  # OpenCV
import numpy as np
from ultralytics import YOLO # YOLOv8
from efficientnet_pytorch import EfficientNet # EfficientNet 라이브러리 필요

original_load = torch.load
def safe_load_wrapper(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = safe_load_wrapper


# CPU/GPU 설정
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"추론 장치: {DEVICE}")

#경로 설정
EMOTION_MODEL_PATH = 'emotion_model.pth' 
NUM_CLASSES = 7
CLASS_NAMES = ['Joy', 'Neutral', 'Surprise', 'Disgust', 'Sadness', 'Anger', 'Fear']


# EfficientNet

def createEmotionModel(numClasses=7):
    # 1. EfficientNet-b0 로드
    model = EfficientNet.from_name('efficientnet-b0')
    
    # 2. 출력층(Fully Connected Layer) 수정
    # EfficientNet은 마지막 레이어 변수명이 'fc'가 아니라 '_fc'입니다.
    inFeatures = model._fc.in_features
    
    # 학습할 때 사용했던 구조(Dropout -> Linear)와 동일하게 맞춰야 가중치가 로드됩니다.
    model._fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(inFeatures, numClasses)
    )
    return model

cnn_model = createEmotionModel(NUM_CLASSES)

# 저장된 가중치(.pth 파일) 로드
print(f"모델 파일 로드 시도: {EMOTION_MODEL_PATH}")
try:
    cnn_model.load_state_dict(torch.load(EMOTION_MODEL_PATH, map_location=DEVICE))
    cnn_model = cnn_model.to(DEVICE)
    cnn_model.eval() # 추론 모드
    print(f"CNN 모델 로드 완료.")
except FileNotFoundError:
    print(f"오류: '{EMOTION_MODEL_PATH}' 파일을 찾을 수 없습니다. 파일명을 확인하세요.")
    exit()
except Exception as e:
    print(f"모델 구조 불일치 또는 로드 오류: {e}")
    exit()

# YOLOv8 로드
try:
    # n보다 성능좋은 m모델
    face_detector = YOLO('yolov8m-face.pt') 
    face_detector.to(DEVICE)
    print("YOLOv8 얼굴 검출기 로드 완료.")
except Exception as e:
    print(f"YOLO 로드 실패, yolov8n-face.pt 재시도")
    try:
        face_detector = YOLO('yolov8n-face.pt')
        face_detector.to(DEVICE)
    except:
        print("pip install ultralytics 확인 필요.")
        exit()

# CNN 입력을 위한 전처리
cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


INPUT_IMAGE_PATH = 'test.jpg'  # 테스트할 이미지 경로
OUTPUT_IMAGE_PATH = 'output.jpg'

# OpenCV로 이미지 읽기
frame = cv2.imread(INPUT_IMAGE_PATH)
if frame is None:
    print(f"오류: {INPUT_IMAGE_PATH} 이미지를 읽을 수 없습니다.")
    # 테스트용으로 빈 이미지라도 생성해서 코드 흐름 확인 (실제 사용시는 exit() 하세요)
    # exit() 
    print("테스트를 위해 검은 화면을 생성합니다.")
    frame = np.zeros((600, 800, 3), dtype=np.uint8)

print(f"이미지 추론 시작...")

# YOLO로 얼굴 검출
results = face_detector(frame, verbose=False)

# 검출된 모든 얼굴에 대해 반복
for result in results:
    for box in result.boxes:
        # A. 얼굴 좌표 추출
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        
        # 좌표가 이미지 밖으로 나가지 않도록 클리핑
        y1, y2 = max(0, y1), min(frame.shape[0], y2)
        x1, x2 = max(0, x1), min(frame.shape[1], x2)

        # B. 얼굴 영역 자르기 (ROI)
        face_roi = frame[y1:y2, x1:x2]
        
        if face_roi.size == 0: continue

        # C. CNN 입력 전처리
        try:
            # OpenCV(BGR) -> PIL(RGB)
            face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
            
            # 전처리 적용
            input_tensor = cnn_transform(face_pil)
            
            # 배치 차원 추가 및 장치 이동
            input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

            # D. CNN 감정 분류 실행
            with torch.no_grad():
                outputs = cnn_model(input_tensor)
                
                # 확률 계산 (Softmax)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # 가장 높은 확률의 클래스(감정) 찾기
                confidence, pred_idx = torch.max(probabilities, 1)
                
                emotion_label = CLASS_NAMES[pred_idx.item()]
                confidence_score = confidence.item()

            # E. 원본 이미지에 결과 그리기 (OpenCV)
            label_text = f"{emotion_label} {confidence_score * 100:.0f}%"
            
            # 텍스트 크기 계산 (반응형)
            scale_factor = max(frame.shape[0], frame.shape[1]) / 1000.0
            thickness = max(1, int(2 * scale_factor))
            font_scale = max(0.5, 0.8 * scale_factor)
            
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            # 얼굴 사각형
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            
            # 텍스트 배경 (얼굴 박스 위쪽)
            cv2.rectangle(frame, 
                          (x1, y1 - text_h - int(10*scale_factor) - 5), 
                          (x1 + text_w + int(10*scale_factor), y1), 
                          (0, 255, 0), -1)
            
            # 텍스트 쓰기
            cv2.putText(frame, label_text, 
                        (x1 + int(5*scale_factor), y1 - int(5*scale_factor)), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
            
        except Exception as e:
            print(f"얼굴 영역 처리 중 오류: {e}")
            continue

# 결과 저장 및 출력
cv2.imwrite(OUTPUT_IMAGE_PATH, frame)
print(f"추론 완료. 결과가 '{OUTPUT_IMAGE_PATH}' 에 저장되었습니다.")
try:
    cv2.imshow('Emotion Detection Result', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except:
    print("화면 출력(cv2.imshow)을 건너뜁니다.")