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

original_load = torch.load
def safe_load_wrapper(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = safe_load_wrapper
# ==========================================

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
    # fontScale을 이미지 크기에 비례해서 키움 (최소 0.8)
    fontScale = max(0.8, 0.5 * scaleFactor)

    results = faceDetector(frame, verbose=False)
    annotatedFrame = frame.copy()
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            faceRoi = frame[y1:y2, x1:x2]
            facePil = Image.fromarray(cv2.cvtColor(faceRoi, cv2.COLOR_BGR2RGB))
            
            label, score = getEmotionScore(facePil)
            
            # 1. 얼굴 박스 그리기
            cv2.rectangle(annotatedFrame, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            
            text = f"{label} {score*100:.0f}%"
            
            # 2. 텍스트 크기 계산
            (textW, textH), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale, thickness)
            
            # 3. 글자 배경 박스 그리기 (초록색 채움)
            # 글자가 얼굴을 가리지 않게 박스 바로 위에 그리기
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
    
    # 1. 얼굴 찾기
    results = faceDetector(frameBgr, verbose=False)
    clickedBox = None
    for result in results:
        for box in result.boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().astype(int)
            if bx1 <= click_x <= bx2 and by1 <= click_y <= by2:
                clickedBox = (bx1, by1, bx2, by2)
                break
        if clickedBox: break
            
    if clickedBox is None: return originalImage, "얼굴을 찾을 수 없습니다."
    
    bx1, by1, bx2, by2 = clickedBox
    
    # 2. 변환 전 '목표 감정' 점수 측정
    ori_face_bgr = frameBgr[by1:by2, bx1:bx2]
    ori_face_pil = Image.fromarray(cv2.cvtColor(ori_face_bgr, cv2.COLOR_BGR2RGB))
    _, before_target_score = getEmotionScore(ori_face_pil, targetEmotion=targetEmotion)
    
    # 3. 얼굴 Crop (여유 있게 1.8배)
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
    
    # 5. 마스크 생성 (하관 60% 유지)
    mask = Image.new("L", process_size, 0)
    draw = ImageDraw.Draw(mask)
    
    # 0.45 (45%) 지점부터 아래쪽을 칠함 (눈 보존, 하관 변형)
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
    restored_face = generated_face.resize(crop_pil.size, Image.Resampling.LANCZOS)
    final_image = originalImage.copy()
    
    paste_mask = Image.new("L", restored_face.size, 255)
    paste_mask = paste_mask.filter(ImageFilter.GaussianBlur(radius=15)) 
    
    final_image.paste(restored_face, (crop_x1, crop_y1), paste_mask)

    # 9. [요청 반영] 변환 후 '목표 감정' 점수 재측정
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