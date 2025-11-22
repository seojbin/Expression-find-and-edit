import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFilter 
import cv2
import numpy as np
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPVisionModelWithProjection, pipeline
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
from efficientnet_pytorch import EfficientNet 
import warnings
import random 

warnings.filterwarnings('ignore')

# PyTorch 호환성
originalLoad = torch.load
def unsafeLoad(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return originalLoad(*args, **kwargs)
torch.load = unsafeLoad

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torchDtype = torch.float16 if torch.cuda.is_available() else torch.float32
print(f"사용 장치 {device}")

modelPath = 'emotion_model.pth' 
numClasses = 7
classNames = ['Joy', 'Neutral', 'Surprise', 'Disgust', 'Sadness', 'Anger', 'Fear']

# [유지] 입꼬리/입술 최적화 프롬프트
EMOTION_PROMPTS = {
    "Joy": "(smile:2.0), (upturned both lip corners:1.7), cheek muscles lifted, happy",
    "Neutral": "neutral face, calm expression, serious, closed mouth, poker face, no emotion",
    "Surprise": "(open mouth:1.3), (jaw drop:1.2), surprised face, wide eyes, shocked, amazed",
    "Disgust": "disgusted face, grimace, ew expression, frowning nose, unpleasant",
    "Sadness": "sad face, crying, tears, frowning, gloomy, upset",
    "Anger": "angry face, frowning, furrowed brows, mad, rage, shouting",
    "Fear": "scared face, terrified, screaming, fear, pale"
}

# CNN 모델
def createEmotionModel(numClasses=7):
    model = EfficientNet.from_name('efficientnet-b0')
    inFeatures = model._fc.in_features
    model._fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(inFeatures, numClasses)
    )
    return model

try:
    cnnModel = createEmotionModel(numClasses)
    cnnModel.load_state_dict(torch.load(modelPath, map_location=device))
    cnnModel = cnnModel.to(device)
    cnnModel.eval()
    print("CNN 모델 로드 완료")
except Exception as e:
    print(f"CNN 로드 실패: {e}")

cnnTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def resizeWithPadding(image, targetSize=(224, 224)):
    oldSize = image.size
    ratio = float(targetSize[0]) / max(oldSize)
    newSize = tuple([int(x * ratio) for x in oldSize])
    image = image.resize(newSize, Image.Resampling.LANCZOS)
    newIm = Image.new("RGB", targetSize, (0, 0, 0))
    newIm.paste(image, ((targetSize[0] - newSize[0]) // 2, (targetSize[1] - newSize[1]) // 2))
    return newIm

# YOLO
yoloPath = 'yolov8m-face.pt'
try:
    faceDetector = YOLO(yoloPath)
    faceDetector.to(device)
except Exception as e:
    print(f"YOLO 로드 실패: {e}")
    exit()

# BLIP & Depth
try:
    blipProcessor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blipModel = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base", torch_dtype=torchDtype
    ).to(device)
    
    depth_device = 0 if torch.cuda.is_available() else -1
    depth_estimator = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas", device=depth_device)
    print("BLIP & Depth 로드 완료")
except Exception as e:
    print(f"모델 로드 실패: {e}")
    exit()

# [수정 1] BLIP이 얼굴 특징(눈 색, 머리 등)을 자세히 묘사하도록 유도
def getFaceDescription(faceImage):
    # "with"로 끝나는 문장을 주어 뒤에 특징(예: brown eyes, glasses)이 이어지게 함
    text = "a close-up portrait of a "
    inputs = blipProcessor(faceImage, text=text, return_tensors="pt").to(device, torchDtype)
    
    # max_new_tokens를 늘려 자세한 설명을 허용
    out = blipModel.generate(**inputs, max_new_tokens=10, repetition_penalty=1.5)
    description = blipProcessor.decode(out[0], skip_special_tokens=True)
    
    return description

def get_depth_map(image, target_size):
    result = depth_estimator(image)
    depth_image = result["depth"]
    depth_image = depth_image.convert("RGB")
    return depth_image.resize(target_size)

# 생성 모델
print("생성 모델 로딩 중...")
try:
    imageEncoder = CLIPVisionModelWithProjection.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", 
        torch_dtype=torchDtype
    ).to(device)

    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0", 
        torch_dtype=torchDtype
    )
    
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        controlnet=controlnet,
        image_encoder=imageEncoder,
        torch_dtype=torchDtype
    )
    
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter-plus-face_sdxl_vit-h.bin")
    pipe.set_ip_adapter_scale(0.7) 
    
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    
    if hasattr(pipe, "image_encoder") and pipe.image_encoder is not None:
        pipe.image_encoder.to(device)

    pipe.set_progress_bar_config(disable=True) 
    print("생성 모델 로드 완료")
except Exception as e:
    print(f"생성 모델 실패: {e}")
    exit()

def verifyFaceRoi(faceRoiBgr):
    try:
        if faceRoiBgr.size == 0: return [], []
        facePil = Image.fromarray(cv2.cvtColor(faceRoiBgr, cv2.COLOR_BGR2RGB))
        facePil = resizeWithPadding(facePil, (224, 224))
        inputTensor = cnnTransform(facePil).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = cnnModel(inputTensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences, indices = torch.topk(probabilities, 3, dim=1)
            confs = confidences.squeeze().cpu().tolist()
            idxs = indices.squeeze().cpu().tolist()
            if not isinstance(idxs, list): idxs = [idxs]; confs = [confs]
            topLabels = [classNames[i] for i in idxs]
            return topLabels, confs
    except: return [], []

def analyzeImage(inputPilImage):
    if inputPilImage is None: return None
    frame = cv2.cvtColor(np.array(inputPilImage), cv2.COLOR_RGB2BGR)
    h, w = frame.shape[:2]
    scaleFactor = max(h, w) / 1000.0
    thickness = max(2, int(3 * scaleFactor))
    fontScale = max(0.6, 1.0 * scaleFactor)
    results = faceDetector(frame, verbose=False)
    annotatedFrame = frame.copy()
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            padding = int(20 * scaleFactor)
            cropY1, cropY2 = max(0, y1 - padding), min(h, y2 + padding)
            cropX1, cropX2 = max(0, x1 - padding), min(w, x2 + padding)
            faceRoi = frame[cropY1:cropY2, cropX1:cropX2]
            topLabels, topScores = verifyFaceRoi(faceRoi)
            if not topLabels: continue
            label = topLabels[0]
            score = topScores[0]
            cv2.rectangle(annotatedFrame, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            labelText = f"{label} {score*100:.1f}%"
            (textW, textH), _ = cv2.getTextSize(labelText, cv2.FONT_HERSHEY_SIMPLEX, fontScale, thickness)
            cv2.rectangle(annotatedFrame, (x1, y1 - textH - int(10 * scaleFactor)), (x1 + textW + int(10 * scaleFactor), y1), (0, 255, 0), -1)
            cv2.putText(annotatedFrame, labelText, (x1 + int(5 * scaleFactor), y1 - int(5 * scaleFactor)), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 0), thickness)
    return Image.fromarray(cv2.cvtColor(annotatedFrame, cv2.COLOR_BGR2RGB))

def processUploadedImage(image):
    return image, analyzeImage(image)

def selectAndChange(originalImage, targetEmotion, evt: gr.SelectData, progress=gr.Progress(track_tqdm=True)):
    if originalImage is None: return None, "이미지를 먼저 업로드해주세요"

    x, y = evt.index
    print(f"좌표 {x} {y} 변환 요청: {targetEmotion}")
    
    frameBgr = cv2.cvtColor(np.array(originalImage), cv2.COLOR_RGB2BGR)
    h, w = frameBgr.shape[:2]
    
    progress(0.1, desc="얼굴 찾는 중")
    results = faceDetector(frameBgr, verbose=False)
    
    clickedBox = None
    for result in results:
        for box in result.boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().astype(int)
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                clickedBox = (bx1, by1, bx2, by2)
                break
        if clickedBox: break
            
    if clickedBox is None: return originalImage, "얼굴을 찾을 수 없습니다"
    
    x1, y1, x2, y2 = clickedBox
    
    scaleFactor = max(h, w) / 1000.0
    padding = int(30 * scaleFactor)
    cropY1, cropY2 = max(0, y1 - padding), min(h, y2 + padding)
    cropX1, cropX2 = max(0, x1 - padding), min(w, x2 + padding)
    
    faceRoiBgr = frameBgr[cropY1:cropY2, cropX1:cropX2]
    facePil = Image.fromarray(cv2.cvtColor(faceRoiBgr, cv2.COLOR_BGR2RGB))
    ipAdapterFaceImage = resizeWithPadding(facePil, (224, 224))
    
    # [수정 2] 개선된 BLIP 함수 사용 (특징 상세 묘사)
    description = getFaceDescription(facePil)
    print(f"Extracted Features: {description}")
    
    targetSize = (1024, 1024)
    controlImagePil = get_depth_map(facePil, targetSize)

    fullMask = Image.new("L", originalImage.size, 0)
    draw = ImageDraw.Draw(fullMask)
    faceHeight = y2 - y1
    
    # 마스크 범위: 광대 포함 (입꼬리 상승 공간)
    maskTop = y1 + int(faceHeight * 0.60) 
    
    draw.rectangle([x1, maskTop, x2, y2], fill=255)
    fullMask = fullMask.filter(ImageFilter.GaussianBlur(radius=15))

    control_scale = 0.5
    strength = 0.65
    guidance_scale = 9.0
    control_guidance_end = 0.2 
    ip_scale = 0.4 

    if targetEmotion in ["Joy", "Surprise", "Anger"]:
        print(f"{targetEmotion}")
        control_scale = 0.3      
        strength = 0.85         
        guidance_scale = 10.0   
        control_guidance_end = 0.1 
        ip_scale = 0.3          

    emotionKeywords = EMOTION_PROMPTS.get(targetEmotion, "expression")
    
    # 프롬프트에 상세 얼굴 묘사(description) 포함
    prompt = (
        f"{emotionKeywords}, "
        f"{description}, " # 여기에 눈 색, 특징 등이 들어감
        f"detailed texture, realistic, sharp focus, "
        f"best quality, masterpiece"
    )
    
    negative_basic = "cartoon, anime, painting, blur, low quality, deformation, bad anatomy, extra fingers, ugly, blurry, long chin, long face"
    lip_negative = "thick lips, swollen lips, pouty lips, botox, fillers, distorted lips"
    
    if targetEmotion == "Joy":
        negativePrompt = f"{negative_basic}, {lip_negative}, gap, sad, angry, frowning, crying"
    elif targetEmotion == "Neutral":
        negativePrompt = f"{negative_basic}, smile, laughing, open mouth, teeth"
    else:
        negativePrompt = f"{negative_basic}, different person, changing identity"

    imgResized = originalImage.resize(targetSize)
    maskResized = fullMask.resize(targetSize)
    controlResized = controlImagePil

    pipe.set_ip_adapter_scale(ip_scale) 
    
    progress(0.4, desc=f"생성 중... (Stop Ctrl at {control_guidance_end*100}%, IP {ip_scale})")
    generator = torch.Generator(device=device).manual_seed(random.randint(0, 999999))

    with torch.inference_mode():
        generatedResized = pipe(
            prompt=prompt,
            image=imgResized,
            mask_image=maskResized,
            control_image=controlResized,
            ip_adapter_image=ipAdapterFaceImage,
            negative_prompt=negativePrompt,
            num_inference_steps=40,
            guidance_scale=guidance_scale, 
            controlnet_conditioning_scale=control_scale,
            control_guidance_end=control_guidance_end, 
            strength=strength,
            generator=generator
        ).images[0]

    generatedBack = generatedResized.resize(originalImage.size)
    
    src = cv2.cvtColor(np.array(generatedBack), cv2.COLOR_RGB2BGR)
    dst = frameBgr.copy()
    center = ((x1 + x2) // 2, (maskTop + y2) // 2)
    src_mask = np.zeros(src.shape, src.dtype)
    cv2.rectangle(src_mask, (x1, maskTop), (x2, y2), (255, 255, 255), -1)
    
    finalOutputPil = originalImage
    
    try:
        mixed = cv2.seamlessClone(src, dst, src_mask, center, cv2.NORMAL_CLONE)
        finalOutputPil = Image.fromarray(cv2.cvtColor(mixed, cv2.COLOR_BGR2RGB))
        
        genRoi = mixed[cropY1:cropY2, cropX1:cropX2]
        
        facePil = Image.fromarray(cv2.cvtColor(genRoi, cv2.COLOR_BGR2RGB))
        facePil = resizeWithPadding(facePil, (224, 224))
        inputTensor = cnnTransform(facePil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = cnnModel(inputTensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            target_idx = classNames.index(targetEmotion)
            target_score = probs[0][target_idx].item()
            
        msg = f"완료: {targetEmotion} {target_score*100:.1f}%"

    except Exception as e:
        print(f"Clone 실패 {e}")
        tempOutput = originalImage.copy()
        tempOutput.paste(generatedBack, mask=fullMask)
        finalOutputPil = tempOutput
        msg = f"완료 (합성 실패, 단순 붙여넣기): {targetEmotion}"

    return finalOutputPil, msg

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI 표정 변환기")
    originalImageState = gr.State()
    with gr.Row():
        with gr.Column():
            imageDisplay = gr.Image(type="pil", label="이미지 업로드", sources=["upload", "clipboard"])
            emotionDropdown = gr.Dropdown(choices=classNames, label="목표 감정", value="Joy")
        with gr.Column():
            imageOutput = gr.Image(type="pil", label="변환 결과")
            textOutput = gr.Textbox(label="상태")

    imageDisplay.upload(fn=processUploadedImage, inputs=imageDisplay, outputs=[originalImageState, imageDisplay])
    imageDisplay.select(fn=selectAndChange, inputs=[originalImageState, emotionDropdown], outputs=[imageOutput, textOutput])

print("UI 실행 중")
demo.launch(share=False)