import streamlit as st
import torch, cv2, numpy as np, random
from PIL import Image
import clip
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import os, requests
import plotly.graph_objects as go
import matplotlib.patches as mpatches

st.set_page_config(page_title="Emotion Segmentation in Art", layout="wide")

# ---------------------------
# 1️⃣ SAM 가중치 자동 다운로드
# ---------------------------
@st.cache_resource
def load_sam_weights():
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    filename = "sam_vit_b_01ec64.pth"
    if not os.path.exists(filename):
        st.write("Downloading SAM weights...")
        r = requests.get(url)
        open(filename, "wb").write(r.content)
    return filename


# ---------------------------
# 2️⃣ 모델 로드
# ---------------------------
@st.cache_resource
def load_models():
    sam_path = load_sam_weights()
    sam = sam_model_registry["vit_b"](checkpoint=sam_path)
    # 세밀한 세그멘테이션 적용
    mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return mask_generator, model, preprocess, device


mask_generator, clip_model, preprocess, device = load_models()

# ---------------------------
# 3️⃣ UI 구성
# ---------------------------
st.title("🎨 Emotion Segmentation in Art")
st.write("SAM + CLIP 기반 예술 작품 감정 시각화")

uploaded = st.file_uploader("예술 작품 이미지를 업로드하세요!", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img_np = np.array(image)
    st.image(image, caption="Original Artwork", use_container_width=True)

    if st.button("▶️ 감정 분석 실행"):
        progress_text = "감정 기반 시각화 생성 중입니다..."
        progress_bar = st.progress(0, text=progress_text)

        # Step 1: SAM 마스크 생성
        masks = mask_generator.generate(img_np)

        # 🧩 너무 작은 마스크 필터링
        h, w, _ = img_np.shape
        min_area = (h * w) * 0.02  # 전체의 2% 미만은 제거
        masks = [m for m in masks if np.sum(m["segmentation"]) >= min_area]

        # 🎭 감정 프롬프트 (구체적이고 CLIP에 맞게)
        emotions = [
            "a painting full of sunlight and yellow tones expressing joy and warmth",       # joy
            "a painting with cold blue hues and a lonely figure expressing sadness",        # sadness
            "a painting with red aggressive brushstrokes showing anger and chaos",          # anger
            "a dark composition with heavy shadows evoking fear and tension",               # fear
            "a soft pastel landscape with balance and calm atmosphere",                     # peace
            "a golden glowing artwork symbolizing hope and rebirth",                        # hope
            "a nostalgic blue painting evoking melancholy and longing",                     # melancholy
            "a vivid red expressive scene full of passion and intensity",                   # passion
            "a surreal, dreamlike painting with distorted forms and mystery",               # mystery
            "a gentle balanced composition radiating serenity and harmony"                  # serenity
        ]

        # 🎨 감정별 랜덤 색상 자동 생성
        def random_color():
            return tuple(random.randint(60, 255) for _ in range(3))

        color_map = {e: random_color() for e in emotions}
        emotion_scores = {e: [] for e in emotions}
        emotion_map = np.zeros_like(img_np)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(img_np)
        ax[0].set_title("Original Artwork")
        ax[1].imshow(img_np)
        ax[1].set_title("Emotion Segmentation + Artistic Moods")

        total_masks = len(masks)
        for idx, mask in enumerate(masks):
            seg = mask["segmentation"]
            crop = img_np * seg[..., None]
            image_input = preprocess(Image.fromarray(crop)).unsqueeze(0).to(device)
            text_inputs = torch.cat([clip.tokenize(e) for e in emotions]).to(device)

            with torch.no_grad():
                image_features = clip_model.encode_image(image_input)
                text_features = clip_model.encode_text(text_inputs)

                # 🔧 특징 정규화
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                sim = (image_features @ text_features.T).softmax(dim=-1)

            probs = sim.squeeze().cpu().numpy()

            # 감정별 확률 저장
            for i, e in enumerate(emotions):
                emotion_scores[e].append(probs[i])

            top_idx = int(np.argmax(probs))
            top_emotion = emotions[top_idx]
            top_prob = probs[top_idx]

            # 🎨 마스크 색상 표시
            emotion_map[seg] = color_map[top_emotion]

            # 🔤 확률 높은 감정만 텍스트 표시
            if top_prob > 0.45:
                y, x = np.where(seg)
                if len(x) > 0 and len(y) > 0:
                    cx, cy = np.mean(x), np.mean(y)
                    ax[1].text(
                        cx, cy, f"{top_prob*100:.1f}%\n{top_emotion.split(' ')[-1]}",
                        color="white", fontsize=7, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.5),
                    )

            # 진행률 업데이트
            progress = int(((idx + 1) / total_masks) * 100)
            progress_bar.progress(progress, text=f"감정 분석 중... {progress}% 완료")

        progress_bar.empty()
        st.success("✅ 감정 분석이 완료되었습니다!")

        # 🎨 감정 색상 오버레이 출력
        ax[1].imshow(emotion_map, alpha=0.80)
        patches = [mpatches.Patch(color=np.array(c)/255, label=e) for e, c in color_map.items()]
        ax[1].legend(handles=patches, loc='lower right', fontsize=6)
        st.pyplot(fig)

        # ----------------------------------------
        # 🎭 감정 확률 평균 계산
        # ----------------------------------------
        avg_scores = {e: np.mean(v) if v else 0 for e, v in emotion_scores.items()}
        sorted_avg = dict(sorted(avg_scores.items(), key=lambda x: x[1], reverse=True))
        dominant_emotion = next(iter(sorted_avg))  # 1위 감정

        # ----------------------------------------
        # 🎨 상위 5개 감정 막대 그래프 (복원)
        # ----------------------------------------
        st.subheader("🎨 Top 5 Artistic Emotions")

        top5 = dict(list(sorted_avg.items())[:5])
        bar_fig = go.Figure(go.Bar(
            x=[v * 100 for v in top5.values()],
            y=[f"{i+1}. {k}" for i, k in enumerate(top5.keys())],
            orientation='h',
            marker_color=[
                'orange' if k == dominant_emotion else 'lightgray'
                for k in top5.keys()
            ],
        ))

        bar_fig.update_layout(
            xaxis_title="Probability (%)",
            yaxis_title="Emotion / Mood",
            xaxis=dict(range=[0, 100]),
            height=500,
            margin=dict(l=150, r=20, t=30, b=30)
        )

        st.plotly_chart(bar_fig, use_container_width=True)
