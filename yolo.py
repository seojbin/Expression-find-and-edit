import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2  # OpenCV
import numpy as np
from ultralytics import YOLO # YOLOv8

# CPU/GPU 설정
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"추론 장치: {DEVICE}")

#학습된 CNN (감정 분류기) 로드
EMOTION_MODEL_PATH = 'emotion_classifier.pth'
NUM_CLASSES = 7
# 학습 스크립트에서 사용한 라벨 순서
CLASS_NAMES = ['Joy', 'Neutral', 'Surprise', 'Disgust', 'Sadness', 'Anger', 'Fear']

# ResNet-18 구조 정의
cnn_model = models.resnet18(weights=None) # pretrained=True가 아닌 weights=None
num_ftrs = cnn_model.fc.in_features
cnn_model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

# 저장된 가중치(.pth 파일) 로드
try:
    cnn_model.load_state_dict(torch.load(EMOTION_MODEL_PATH, map_location=DEVICE))
except FileNotFoundError:
    print(f"오류: {EMOTION_MODEL_PATH} 파일을 찾을 수 없습니다.")
    exit()

cnn_model = cnn_model.to(DEVICE)
cnn_model.eval() # 추론 모드
print(f"'{EMOTION_MODEL_PATH}' 모델 로드 완료.")

# YOLOv8 로드
# 'yolov8n-face.pt'는 얼굴 검출에 특화된 경량 모델
try:
    face_detector = YOLO('yolov8m-face.pt')
    face_detector.to(DEVICE)
    print("YOLOv8 얼굴 검출기 로드 완료.")
except Exception as e:
    print(f"YOLO 모델 로드 실패: {e}")
    print("터미널에서 'pip install ultralytics'를 실행했는지 확인하세요.")
    exit()

# 3. CNN 입력을 위한 전처리
cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#추론 실행

INPUT_IMAGE_PATH = 'test.jpg'  # 여기에 테스트할 이미지 경로를 넣으세요.
OUTPUT_IMAGE_PATH = 'output.jpg'

# OpenCV로 이미지 읽기 (BGR 순서)
frame = cv2.imread(INPUT_IMAGE_PATH)
if frame is None:
    print(f"오류: {INPUT_IMAGE_PATH} 이미지를 읽을 수 없습니다.")
    exit()

print(f"'{INPUT_IMAGE_PATH}' 이미지 추론 시작...")

# YOLO로 얼굴 검출
results = face_detector(frame, verbose=False)

# 검출된 모든 얼굴에 대해 반복
for result in results:
    for box in result.boxes:
        #얼굴 좌표 추출
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        
        # 좌표가 이미지 밖으로 나가지 않도록
        y1, y2 = max(0, y1), min(frame.shape[0], y2)
        x1, x2 = max(0, x1), min(frame.shape[1], x2)

        # B. 얼굴 영역 자르기
        face_roi = frame[y1:y2, x1:x2]
        
        # C. CNN 입력 전처리
        try:
            # OpenCV(BGR) -> PIL(RGB)
            face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
            
            # 전처리 적용
            input_tensor = cnn_transform(face_pil)
            
            # 배치 차원 추가
            input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

            # CNN 감정 분류 실행
            with torch.no_grad():
                outputs = cnn_model(input_tensor)
                
                # 확률 계산 (Softmax)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # 가장 높은 확률의 클래스(감정) 찾기
                confidence, pred_idx = torch.max(probabilities, 1)
                
                emotion_label = CLASS_NAMES[pred_idx.item()]
                confidence_score = confidence.item()

            # E. 원본 이미지에 결과 그리기 (OpenCV)
            label_text = f"{emotion_label} ({confidence_score * 100:.1f}%)"
            
            # 얼굴 영역에 사각형 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # 초록색 사각형
            
            # 텍스트 배경 그리기
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1) # 배경 채우기
            
            # 텍스트 쓰기
            cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2) # 검은색 글씨
            
        except Exception as e:
            print(f"얼굴 영역 처리 중 오류: {e}")
            continue

cv2.imwrite(OUTPUT_IMAGE_PATH, frame)
print(f"추론 완료. 결과가 {OUTPUT_IMAGE_PATH} 에 저장되었습니다.")

cv2.imshow('Emotion Detection Result', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
