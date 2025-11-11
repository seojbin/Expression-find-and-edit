import gradio as gr
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import cv2  # OpenCV
import numpy as np
from ultralytics import YOLO # YOLOv8
# OpenPose ControlNet 관련 모듈 임포트
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector 
import warnings
import random 

warnings.filterwarnings('ignore', category=UserWarning, message='Failed to initialize NumPy')

# CPU/GPU 설정
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
print(f"사용 장치: {DEVICE}")

#CNN 분류기 로드
EMOTION_MODEL_PATH = 'emotion_classifier.pth' # 학습시킨 모델
NUM_CLASSES = 7
CLASS_NAMES = ['Joy', 'Neutral', 'Surprise', 'Disgust', 'Sadness', 'Anger', 'Fear']

#ResNet-18
def create_emotion_model(num_classes=7):
    model = models.resnet18(weights=None) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# CNN 가중치 로드
try:
    cnn_model = create_emotion_model(NUM_CLASSES)
    cnn_model.load_state_dict(torch.load(EMOTION_MODEL_PATH, map_location=DEVICE))
    cnn_model = cnn_model.to(DEVICE)
    cnn_model.eval() # 추론 모드
    print(f"CNN 분류기 '{EMOTION_MODEL_PATH}' 로드 완료.")
except FileNotFoundError:
    print(f"오류: CNN 모델 '{EMOTION_MODEL_PATH}' 파일을 찾을 수 없습니다.")
    exit()
except Exception as e:
    print(f"CNN 모델 로드 중 오류: {e}")
    exit()

# CNN 입력을 위한 전처리 (학습 시 'val'과 동일)
cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 2. [검출용] YOLO 로드 (변경 없음) ---
YOLO_MODEL_PATH = 'yolov8m-face.pt' # 사용자가 지정한 모델
try:
    face_detector = YOLO(YOLO_MODEL_PATH)
    face_detector.to(DEVICE)
    print(f"YOLO 검출기 '{YOLO_MODEL_PATH}' 로드 완료.")
except FileNotFoundError:
    print(f"오류: YOLO 모델 '{YOLO_MODEL_PATH}' 파일을 찾을 수 없습니다.")
    print("터미널에서 'Invoke-WebRequest -Uri \"https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-face.pt\" -OutFile \"yolov8m-face.pt\"' 를 실행하세요.")
    exit()
except Exception as e:
    print(f"YOLO 모델 로드 중 오류: {e}")
    exit()

# --- 3. [생성용] OpenPose ControlNet 파이프라인 로드 (변경 없음) ---
print("ControlNet 및 OpenPose (뼈대) 감지기")
try:
    openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
    
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose", torch_dtype=torch_dtype
    )
    print("ControlNet 로드 완료.")

except Exception as e:
    print(f" ControlNet 모델 로드 실패: {e}. 'pip install controlnet-aux'를 실행했는지 확인하세요.")
    exit()

print("Stable Diffusion Inpainting 모델 로드 중")
try:
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        controlnet=controlnet, # ControlNet 결합
        torch_dtype=torch_dtype
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(DEVICE)
    pipe.set_progress_bar_config(disable=True) 
    print("Stable Diffusion + ControlNet 로드 완료.")
except Exception as e:
    print(f"Diffusion 파이프라인 로드 실패: {e}")
    exit()


# 얼굴 영역(ROI) 검증 함수
def verify_face_roi(face_roi_bgr):
    #OpenCV BGR *얼굴 영역*을 입력받아 Top-2 감정을 분류
    try:
        face_pil = Image.fromarray(cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2RGB))
        input_tensor = cnn_transform(face_pil).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = cnn_model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            #Top 2 예측 결과 추출
            confidences, indices = torch.topk(probabilities, 2, dim=1)
            
            confidences = confidences.squeeze().cpu().tolist()
            indices = indices.squeeze().cpu().tolist()

            #만약 클래스가 1개뿐이거나 결과가 1개만 나오면
            if not isinstance(indices, list): 
                indices = [indices]
                confidences = [confidences]

            # 1순위 정보
            top_label = CLASS_NAMES[indices[0]]
            top_score = confidences[0]
            
            # 2순위 정보 (결과가 2개 이상일 때만)
            second_label = "N/A"
            second_score = 0.0
            if len(indices) > 1:
                second_label = CLASS_NAMES[indices[1]]
                second_score = confidences[1]
                
            return top_label, top_score, second_label, second_score
    
    except Exception as e:
        print(f"CNN 검증 중 오류: {e}")
        return "검증 오류", 0.0, "N/A", 0.0

def select_and_change(input_pil_image, target_emotion, evt: gr.SelectData, progress=gr.Progress(track_tqdm=True)):
    if input_pil_image is None:
        return None, "이미지를 업로드하세요."

    #클릭 좌표
    x, y = evt.index
    print(f"\n'{target_emotion}' 표정으로 변환 (클릭 좌표: {x}, {y})")
    progress(0.1, desc="얼굴 검출 중 (YOLO)...")
    
    frame_bgr = cv2.cvtColor(np.array(input_pil_image), cv2.COLOR_RGB2BGR)

    #얼굴 검출 (YOLO) 및 클릭된 얼굴 찾기
    results = face_detector(frame_bgr, verbose=False)
    
    if not results or len(results[0].boxes) == 0:
        return input_pil_image, "얼굴을 찾을 수 없습니다."

    all_boxes = [box.xyxy[0].cpu().numpy().astype(int) for result in results for box in result.boxes]
    
    clicked_box = None
    for box in all_boxes:
        x1, y1, x2, y2 = box
        if x1 <= x <= x2 and y1 <= y <= y2:
            clicked_box = box
            break
            
    if clicked_box is None:
        print("클릭한 위치에서 얼굴을 찾지 못했습니다.")
        return input_pil_image, "클릭한 위치에서 얼굴을 찾지 못했습니다. 얼굴 중앙을 다시 클릭해주세요."
    
    print(f"선택된 얼굴 BBox: {clicked_box}")
    x1, y1, x2, y2 = clicked_box

    #원본 표정 검증
    progress(0.2, desc="원본 표정 검증 중 (CNN)")
    original_face_roi = frame_bgr[y1:y2, x1:x2]
    original_label, _, _, _ = verify_face_roi(original_face_roi)
    print(f"원본 얼굴 표정 검증: {original_label}")

    # ControlNet OpenPose(뼈대) 이미지 생성
    progress(0.3, desc="OpenPose 뼈대 추출 중")
    control_image_pil = openpose(input_pil_image)

    #정밀 마스크 생성 (눈/입)
    progress(0.4, desc="정밀 마스크 생성 중")
    face_h = y2 - y1
    face_w = x2 - x1
    padding_w = int(face_w * 0.1) # 좌우 패딩
    
    mask_image = Image.new("L", input_pil_image.size, 0) # 검은색 배경
    draw = ImageDraw.Draw(mask_image)
    
    # 눈/눈썹 영역 (상단 40%)
    mask_y1_top = y1
    mask_y2_top = y1 + int(face_h * 0.40)
    draw.rectangle(
        [max(0, x1 - padding_w), mask_y1_top, 
         min(input_pil_image.width, x2 + padding_w), mask_y2_top], 
        fill=255
    )
    
    # 입/턱 하관 영역 (하단 40%)
    mask_y1_bottom = y1 + int(face_h * 0.60)
    mask_y2_bottom = y2
    draw.rectangle(
        [max(0, x1 - padding_w), mask_y1_bottom, 
         min(input_pil_image.width, x2 + padding_w), mask_y2_bottom], 
        fill=255
    )

    # 재시도 루프, ControlNet Inpainting 생성
    MAX_ATTEMPTS = 3 # 최대 3회 시도
    
    prompt = f"a high resolution photo of a face with a {target_emotion.lower()} expression, realistic, detailed, high quality, sharp focus"
    negative_prompt = "cartoon, disfigured, low quality, ugly, blurry, watermark, bad anatomy, extra limbs, bad face, deformed eyes"

    generated_image_pil = None
    verified_label = ""
    verified_score = 0.0

    # 리사이즈
    img_512 = input_pil_image.resize((512, 512))
    mask_512 = mask_image.resize((512, 512))
    control_512 = control_image_pil.resize((512, 512))

    for i in range(MAX_ATTEMPTS):
        attempt_num = i + 1
        print(f"--- [시도 {attempt_num}/{MAX_ATTEMPTS}] ---")
        progress(0.3 + (i * 0.2), desc=f"[{attempt_num}/{MAX_ATTEMPTS}] 표정 생성 중 (OpenPose)...")
        
        generator = torch.Generator(device=DEVICE).manual_seed(random.randint(0, 9999999))

        with torch.inference_mode():
            # ControlNet Inpainting 파이프라인 호출
            generated_face_512 = pipe(
                prompt=prompt,
                image=img_512,          # 원본 이미지 (512)
                mask_image=mask_512,    # 마스크 이미지 (512)
                control_image=control_512, # OpenPose 맵 전달
                negative_prompt=negative_prompt,
                num_inference_steps=30,
                guidance_scale=8.0,
                generator=generator,
                controlnet_conditioning_scale=0.7
            ).images[0]

        # 원본 이미지 크기로 최종 결과물 리사이즈
        output_image_pil = generated_face_512.resize(input_pil_image.size)
        
        #  Top-2 검증 로직
        progress(0.4 + (i * 0.2), desc=f"[{attempt_num}/{MAX_ATTEMPTS}] 생성 결과 검증 중 (CNN)...")
        
        # 생성된 이미지에서 동일한 좌표의 얼굴 영역을 잘라 검증
        generated_bgr = cv2.cvtColor(np.array(output_image_pil), cv2.COLOR_RGB2BGR)
        generated_face_roi = generated_bgr[y1:y2, x1:x2]
        
        verified_label, verified_score, second_label, second_score = verify_face_roi(generated_face_roi)
        
        print(f"시도 {attempt_num} 검증: 1순위={verified_label} ({verified_score*100:.1f}%), 2순위={second_label} ({second_score*100:.1f}%)")

        # 1순위가 목표 감정인가? (예: Joy == Joy)
        if verified_label == target_emotion:
            print("검증 성공 (1순위 일치)")
            break
        
        # 1순위가 원본 감정이고, 2순위가 목표 감정인가? (예: Fear == Fear AND Joy == Joy)
        if verified_label == original_label and second_label == target_emotion:
            print("검증 성공 (원본 감정 1순위, 목표 감정 2순위)")
            # 성공으로 간주하기 위해 값을 교체
            verified_label = second_label
            verified_score = second_score
            break

    # --- Step 8: 최종 결과 반환 ---
    if verified_label == target_emotion:
        output_text = f"성공 (시도 {attempt_num}회): {verified_label} ({verified_score*100:.1f}%)"
    else:
        output_text = f"실패 (시도 {MAX_ATTEMPTS}회): 최종 1순위={verified_label}, 2순위={second_label}"
    
    print(output_text)
    return output_image_pil, output_text


# --- 7. Gradio 인터페이스 실행 ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # AI 표정 변환 및 검증 시스템
        YOLO로 얼굴을 검출하고, ControlNet(OpenPose)으로 뼈대를 고정한 뒤 정밀 마스킹(Inpainting)으로
        피부색을 보존하며 표정을 변환합니다. CNN 모델로 결과를 검증하고 실패 시 재시도합니다.
        
        **사용법:**
        1. 이미지를 업로드합니다.
        2. 변경할 감정을 선택합니다.
        3. 이미지에서 표정을 바꾸고 싶은 얼굴을 클릭하면 변환이 실행됩니다.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="1. 이미지 업로드 (클릭하여 선택)")
            emotion_dropdown = gr.Dropdown(
                choices=CLASS_NAMES, label="2. 목표 감정 선택", value="Joy"
            )
        
        with gr.Column(scale=1):
            image_output = gr.Image(type="pil", label="3. 변환된 이미지")
            text_output = gr.Textbox(label="4. CNN 검증 결과", interactive=False)

    image_input.select(
        fn=select_and_change, 
        inputs=[image_input, emotion_dropdown], 
        outputs=[image_output, text_output],
        show_progress="full"
    )

print("Gradio UI를 실행합니다. http://127.0.0.1:7860 에 접속하세요.")
demo.launch(share=True)
