import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFilter
import cv2
import numpy as np
import diffusers.utils.logging
import warnings
import random

# 안전한 torch.load 환경 설정
original_load = torch.load
def safe_load_wrapper(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = safe_load_wrapper


from ultralytics import YOLO
from diffusers import StableDiffusionXLInpaintPipeline, UniPCMultistepScheduler
from transformers import CLIPVisionModelWithProjection
from efficientnet_pytorch import EfficientNet 

warnings.filterwarnings('ignore')
diffusers.utils.logging.set_verbosity_error()

# --- 1. 초기 설정 ---

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
print(f"사용 장치: {device}")

classNames = ['Joy', 'Neutral', 'Surprise', 'Disgust', 'Sadness', 'Anger', 'Fear']

# 프롬프트 설정
EMOTION_PROMPTS = {
    "Joy": "(smile:2.0), (upturned both lip corners:1.7), cheek muscles lifted, happy, realistic skin texture",
    "Neutral": "neutral face, calm expression, serious, closed mouth, (poker face:2.0), (no emotion:2.0)",
    "Surprise": "open mouth, jaw drop, (surprised face:2.0), (shocked:2.0)",
    "Disgust": "disgusted face, (grimace:2.0), (ew expression:2.0), unpleasant",
    "Sadness": "sad face, crying, frowning, (gloomy:2.0), upset",
    "Anger": "(angry face:2.0), frowning, furrowed brows, mad, rage",
    "Fear": "scared face, (terrified:2.0), (screaming:2.0), fear, pale"
}

NEGATIVE_PROMPT = "cartoon, anime, painting, blur, low quality, deformation, bad anatomy, extra fingers, ugly, blurry, long chin, long face, 3d render"

# --- 2. 모델 로드 ---

print("1. YOLO 모델 로딩 중 (Medium)...")
try:
    faceDetector = YOLO('yolov8m-face.pt')
    faceDetector.to(device)
except Exception as e:
    print(f"yolov8m-face.pt 로드 실패, n 모델로 대체 시도: {e}")
    faceDetector = YOLO('yolov8n-face.pt')

print("2. 감정 분석 모델(EfficientNet) 로딩 중...")
def createEmotionModel(numClasses=7):
    model = EfficientNet.from_name('efficientnet-b0')
    inFeatures = model._fc.in_features
    model._fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(inFeatures, numClasses)
    )
    return model

cnnTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

cnnModel = None
try:
    cnnModel = createEmotionModel(7)
    cnnModel.load_state_dict(torch.load('emotion_model.pth', map_location=device))
    cnnModel = cnnModel.to(device)
    cnnModel.eval()
    print("   -> 감정 분석 모델 로드 완료")
except Exception as e:
    print(f"   -> 감정 모델 로드 실패 (분석 기능 꺼짐): {e}")
    cnnModel = None

print("3. 생성 모델(RealVisXL) 및 Image Encoder 로딩 중...")
try:
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter", 
        subfolder="models/image_encoder",
        torch_dtype=torch_dtype
    ).to(device)

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "SG161222/RealVisXL_V4.0_Lightning",
        image_encoder=image_encoder,
        torch_dtype=torch_dtype,
        variant="fp16",
        use_safetensors=True
    ).to(device)
    
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter-plus-face_sdxl_vit-h.bin")
    pipe.set_ip_adapter_scale(0.5) 

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_tiling()
    print("   -> 생성 모델 로드 완료")
    
except Exception as e:
    print(f"생성 모델 로드 실패: {e}")
    exit()


# --- 3. 유틸리티 함수 ---

def resizeWithPadding(image, targetSize=(224, 224)):
    oldSize = image.size
    ratio = float(targetSize[0]) / max(oldSize)
    newSize = tuple([int(x * ratio) for x in oldSize])
    image = image.resize(newSize, Image.Resampling.LANCZOS)
    newIm = Image.new("RGB", targetSize, (0, 0, 0))
    newIm.paste(image, ((targetSize[0] - newSize[0]) // 2, (targetSize[1] - newSize[1]) // 2))
    return newIm

# 색상 정합 함수 (YCrCb 기반으로 수정하여 흰 픽셀 오류 방지 및 색상 보존력 향상)
def match_color_stats(source_pil, target_pil):
    # PIL -> OpenCV RGB -> YCrCb (밝기 Y, 색상 Cr/Cb)
    source = cv2.cvtColor(np.array(source_pil), cv2.COLOR_RGB2YCrCb).astype(np.float32)
    target = cv2.cvtColor(np.array(target_pil), cv2.COLOR_RGB2YCrCb).astype(np.float32)

    # 통계 계산
    # Y 채널은 보존하고 Cr/Cb 채널(색상)만 매칭하는 것이 안전함.
    # 하지만 여기서는 전체 3채널(YCrCb) 모두 매칭하여 색상 불일치를 최대한 해결합니다.
    source_mean, source_std = cv2.meanStdDev(source)
    target_mean, target_std = cv2.meanStdDev(target)
    
    # 평균 및 표준편차 매칭
    result = source.copy()
    for i in range(3): # Y, Cr, Cb 채널별 적용
        # 표준편차가 0인 경우 나누기 오류 방지
        if source_std[i] > 1e-6:
            # 표준편차 정규화 및 평균 이동
            result[:,:,i] = (result[:,:,i] - source_mean[i]) * (target_std[i] / source_std[i]) + target_mean[i]
        
    # 클리핑 (0~255) 및 데이터 타입 변환
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # YCrCb -> RGB -> PIL
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_YCrCb2RGB))


# 특정 감정 점수 가져오기
def getEmotionScore(facePil, targetEmotion=None):
    if cnnModel is None: return "N/A", 0.0
    try:
        facePil = resizeWithPadding(facePil, (224, 224))
        inputTensor = cnnTransform(facePil).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = cnnModel(inputTensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # 1. 특정 목표 감정(targetEmotion)의 점수 반환
            if targetEmotion and targetEmotion in classNames:
                target_idx = classNames.index(targetEmotion)
                score = probabilities[0][target_idx].item()
                return targetEmotion, score
            
            # 2. 아니면 가장 높은 감정 반환
            else:
                confidences, indices = torch.topk(probabilities, 1, dim=1)
                score = confidences.item()
                label = classNames[indices.item()]
                return label, score
    except: return "Error", 0.0


def analyzeImage(inputPilImage):
    if inputPilImage is None: return None
    frame = cv2.cvtColor(np.array(inputPilImage), cv2.COLOR_RGB2BGR)
    h, w = frame.shape[:2]
    
    # 이미지 크기에 따른 스케일 팩터 계산
    scaleFactor = max(h, w) / 1000.0
    thickness = max(2, int(1 * scaleFactor))
    fontScale = max(0.8, 0.5 * scaleFactor)

    results = faceDetector(frame, verbose=False)
    annotatedFrame = frame.copy()
    
    # YOLO 결과 필터링 기준 정의
    min_confidence = 0.3  # 신뢰도 30% 미만 필터링
    min_face_area = (h * w) * 0.005 # 이미지 전체 면적의 0.5% 미만 필터링

    for result in results:
        for box in result.boxes:
            confidence = box.conf[0].cpu().numpy()
            
            # 1. 신뢰도 필터링
            if confidence < min_confidence:
                 continue
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # 2. 박스 크기 필터링
            face_area = (x2 - x1) * (y2 - y1)
            if face_area < min_face_area:
                continue
                
            faceRoi = frame[y1:y2, x1:x2]
            facePil = Image.fromarray(cv2.cvtColor(faceRoi, cv2.COLOR_BGR2RGB))
            
            label, score = getEmotionScore(facePil)
            
            # 1. 얼굴 박스 그리기
            cv2.rectangle(annotatedFrame, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            
            text = f"{label} {score*100:.0f}%"
            
            # 2. 텍스트 크기 계산
            (textW, textH), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale, thickness)
            
            # 3. 글자 배경 박스 그리기 (초록색 채움)
            cv2.rectangle(annotatedFrame, (x1, y1 - textH - int(10*scaleFactor) - 5), (x1 + textW + 10, y1), (0, 255, 0), -1)
            
            # 4. 텍스트 그리기 (검은색 글씨)
            cv2.putText(annotatedFrame, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 0), thickness)
            
    return Image.fromarray(cv2.cvtColor(annotatedFrame, cv2.COLOR_BGR2RGB))


def processUploadedImage(image):
    return image, analyzeImage(image)


# --- 4. 핵심 변환 로직 ---

def selectAndChange(originalImage, targetEmotion, evt: gr.SelectData, progress=gr.Progress(track_tqdm=True)):
    if originalImage is None: return None, "이미지를 먼저 업로드해주세요"

    click_x, click_y = evt.index
    
    frameBgr = cv2.cvtColor(np.array(originalImage), cv2.COLOR_RGB2BGR)
    h, w = frameBgr.shape[:2]
    
    # YOLO 결과 필터링
    min_confidence = 0.3
    min_face_area = (h * w) * 0.005 

    # 1. 얼굴 찾기
    results = faceDetector(frameBgr, verbose=False)
    clickedBox = None
    for result in results:
        for box in result.boxes:
            confidence = box.conf[0].cpu().numpy()
            if confidence < min_confidence:
                 continue

            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().astype(int)
            
            face_area = (bx2 - bx1) * (by2 - by1)
            if face_area < min_face_area:
                continue

            # 클릭된 박스 찾기
            if bx1 <= click_x <= bx2 and by1 <= click_y <= by2:
                clickedBox = (bx1, by1, bx2, by2)
                break
        if clickedBox: break
            
    if clickedBox is None: return originalImage, "클릭 위치에서 유효한 얼굴을 찾을 수 없습니다."
    
    bx1, by1, bx2, by2 = clickedBox
    
    # 2. 변환 전 '목표 감정' 점수 측정
    ori_face_bgr = frameBgr[by1:by2, bx1:bx2]
    ori_face_pil = Image.fromarray(cv2.cvtColor(ori_face_bgr, cv2.COLOR_BGR2RGB))
    _, before_target_score = getEmotionScore(ori_face_pil, targetEmotion=targetEmotion)
    
    # 3. 얼굴 Crop
    face_w, face_h = bx2 - bx1, by2 - by1
    cx, cy = bx1 + face_w // 2, by1 + face_h // 2
    size = int(max(face_w, face_h) * 1.8)
    
    crop_x1 = max(0, cx - size // 2)
    crop_y1 = max(0, cy - size // 2)
    crop_x2 = min(w, cx + size // 2)
    crop_y2 = min(h, cy + size // 2)
    
    crop_img_bgr = frameBgr[crop_y1:crop_y2, crop_x1:crop_x2]
    crop_pil = Image.fromarray(cv2.cvtColor(crop_img_bgr, cv2.COLOR_BGR2RGB))

    # 4. 리사이즈 (768px)
    process_size = (768, 768) 
    resized_face = crop_pil.resize(process_size, Image.Resampling.LANCZOS)
    
    # 5. 마스크 생성
    mask = Image.new("L", process_size, 0)
    draw = ImageDraw.Draw(mask)
    
    # 0.45 지점부터 아래쪽을 칠함 (눈 보존, 하관 변형)
    mask_start_y = int(process_size[1] * 0.45)
    draw.rectangle([0, mask_start_y, process_size[0], process_size[1]], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=30))

    # 6. 프롬프트
    prompt = f"close-up photo of a person, {EMOTION_PROMPTS[targetEmotion]}, high quality, 8k, raw photo, realistic skin texture"
    neg_prompt = NEGATIVE_PROMPT

    # 7. 생성
    progress(0.4, desc="표정 생성 중...")
    generator = torch.Generator(device).manual_seed(random.randint(0, 999999))
    
    with torch.inference_mode():
        generated_face = pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            image=resized_face,
            mask_image=mask,
            ip_adapter_image=resized_face,
            num_inference_steps=6,
            guidance_scale=1.5,
            strength=0.65,
            generator=generator
        ).images[0]
        
    # 8. 합성
    
    # 색상 정합 수행
    restored_face_resized = generated_face.resize(crop_pil.size, Image.Resampling.LANCZOS)
    color_matched_face = match_color_stats(restored_face_resized, crop_pil)
    
    final_image = originalImage.copy()
    
    # 마스크 블러 반경 증가
    paste_mask = Image.new("L", color_matched_face.size, 255)
    paste_mask = paste_mask.filter(ImageFilter.GaussianBlur(radius=20)) # radius를 20으로 유지
    
    final_image.paste(color_matched_face, (crop_x1, crop_y1), paste_mask)


    final_frame_bgr = cv2.cvtColor(np.array(final_image), cv2.COLOR_RGB2BGR)
    modified_crop = final_frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
    modified_pil = Image.fromarray(cv2.cvtColor(modified_crop, cv2.COLOR_BGR2RGB))
    
    _, after_target_score = getEmotionScore(modified_pil, targetEmotion=targetEmotion)
    
    # 결과 메시지: 목표 감정 점수 변화량 표시
    msg = f"완료: {targetEmotion} ({before_target_score*100:.1f}% → {after_target_score*100:.1f}%)"
    
    return final_image, msg


# --- UI ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI 표정 변환기")
    gr.Markdown("클릭한 얼굴의 표정을 바꿉니다.")
    
    originalImageState = gr.State()
    
    with gr.Row():
        with gr.Column():
            imageDisplay = gr.Image(type="pil", label="1. 이미지 업로드", sources=["upload", "clipboard"])
            emotionDropdown = gr.Dropdown(choices=classNames, label="2. 목표 감정", value="Joy")
        with gr.Column():
            imageOutput = gr.Image(type="pil", label="3. 결과")
            textOutput = gr.Textbox(label="감정 변화 분석", show_copy_button=True)

    imageDisplay.upload(fn=processUploadedImage, inputs=imageDisplay, outputs=[originalImageState, imageDisplay])
    imageDisplay.select(fn=selectAndChange, inputs=[originalImageState, emotionDropdown], outputs=[imageOutput, textOutput])

if __name__ == "__main__":
    print("UI 실행 중...")
    demo.launch(share=False)