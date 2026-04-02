import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
import torchvision.transforms as transforms
import sys
import os
sys.path.append('src')
from model import get_model

# ─── CONFIG ───────────────────────────────────────────────
MODEL_PATH = 'data/best_model.pth'
IMG_SIZE   = 224
LABELS     = {
    0: ('No DR',         '🟢', 'No diabetic retinopathy detected. Keep monitoring annually.', '#00C853'),
    1: ('Mild',          '🟡', 'Mild DR detected. Schedule a follow-up in 6 months.', '#FFD600'),
    2: ('Moderate',      '🟠', 'Moderate DR detected. See an ophthalmologist soon.', '#FF6D00'),
    3: ('Severe',        '🔴', 'Severe DR! Visit an ophthalmologist urgently.', '#D50000'),
    4: ('Proliferative', '🚨', 'Proliferative DR! Immediate medical attention required.', '#6A1B9A'),
}
# ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DR Screening AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CUSTOM CSS ───────────────────────────────────────────
st.markdown("""
<style>
    /* Background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }

    /* Hero header */
    .hero {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .hero h1 {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(90deg, #f953c6, #b91d73, #00c3ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: #aaa;
        font-size: 1.1rem;
    }

    /* Cards */
    .card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }
    .card h3 {
        color: white;
        margin-bottom: 1rem;
        font-size: 1.1rem;
        letter-spacing: 0.05rem;
    }

    /* Metric cards */
    .metric-card {
        background: rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: white;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #aaa;
        text-transform: uppercase;
        letter-spacing: 0.1rem;
    }

    /* Result badge */
    .result-badge {
        border-radius: 50px;
        padding: 0.8rem 2rem;
        font-size: 1.5rem;
        font-weight: 800;
        text-align: center;
        margin: 1rem 0;
        display: inline-block;
        width: 100%;
    }

    /* Upload box */
    .upload-box {
        border: 2px dashed rgba(249,83,198,0.5);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        background: rgba(249,83,198,0.05);
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #f953c6, #b91d73);
        border-radius: 10px;
    }

    /* Stage pill */
    .stage-pill {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }

    /* Divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(249,83,198,0.5), transparent);
        margin: 1.5rem 0;
    }

    /* Hide streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* File uploader */
    .stFileUploader {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── FUNCTIONS ────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = get_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

def ben_graham(image, sigma=10):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    blurred  = cv2.GaussianBlur(image, (0,0), sigma)
    enhanced = cv2.addWeighted(image, 4, blurred, -4, 128)
    mask = np.zeros(enhanced.shape, dtype=np.uint8)
    h, w = enhanced.shape[:2]
    cv2.circle(mask, (w//2, h//2), int(min(h,w)*0.45), (1,1,1), -1)
    enhanced = enhanced * mask + 128*(1-mask)
    enhanced = cv2.cvtColor(enhanced.astype(np.uint8), cv2.COLOR_BGR2RGB)
    return Image.fromarray(enhanced)

def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])
    return transform(image)

def grad_cam(model, img_tensor):
    gradients   = []
    activations = []

    def save_grad(grad):
        gradients.append(grad)

    def fwd_hook(module, input, output):
        activations.append(output)
        output.register_hook(save_grad)

    handle = model.backbone.features[-1].register_forward_hook(fwd_hook)
    output = model(img_tensor.unsqueeze(0))
    pred   = output.argmax().item()

    with torch.no_grad():
        probs = torch.softmax(output, dim=1)[0].numpy()

    model.zero_grad()
    output[0, pred].backward()
    handle.remove()

    grad = gradients[0].squeeze(0).mean(dim=[1,2])
    act  = activations[0].squeeze(0)
    cam  = torch.zeros(act.shape[1:])
    for i, w in enumerate(grad):
        cam += w * act[i]

    cam = torch.relu(cam).detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = np.array(Image.fromarray(
        (cam*255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE)))
    return cam, pred, probs

def overlay_gradcam(original_img, cam):
    import matplotlib.cm as colormap
    img_arr = np.array(original_img.resize((IMG_SIZE, IMG_SIZE)))
    heatmap = colormap.jet(cam)[:,:,:3]
    heatmap = (heatmap * 255).astype(np.uint8)
    overlay = (0.55 * img_arr + 0.45 * heatmap).astype(np.uint8)
    return Image.fromarray(overlay)

# ─── HERO HEADER ──────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>👁️ DR Screening AI</h1>
    <p>Diabetic Retinopathy Detection powered by EfficientNet-B3 & Grad-CAM</p>
</div>
""", unsafe_allow_html=True)

# ─── TOP STATS ROW ────────────────────────────────────────
s1, s2, s3, s4 = st.columns(4)
with s1:
    st.markdown("""<div class="metric-card">
        <div class="metric-value" style="color:#f953c6;">0.749</div>
        <div class="metric-label">Val Kappa Score</div>
    </div>""", unsafe_allow_html=True)
with s2:
    st.markdown("""<div class="metric-card">
        <div class="metric-value" style="color:#00c3ff;">11M</div>
        <div class="metric-label">Model Parameters</div>
    </div>""", unsafe_allow_html=True)
with s3:
    st.markdown("""<div class="metric-card">
        <div class="metric-value" style="color:#00C853;">3,662</div>
        <div class="metric-label">Training Images</div>
    </div>""", unsafe_allow_html=True)
with s4:
    st.markdown("""<div class="metric-card">
        <div class="metric-value" style="color:#FFD600;">5</div>
        <div class="metric-label">DR Stages</div>
    </div>""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─── MAIN LAYOUT ──────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="card"><h3>📤 Upload Fundus Image</h3>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=['png','jpg','jpeg'],
                                 label_visibility="collapsed")
    if uploaded:
        image = Image.open(uploaded).convert('RGB')
        st.image(image, caption="Original Image", use_container_width=True)
    else:
        st.markdown("""
        <div class="upload-box">
            <h2>🖼️</h2>
            <p style="color:#aaa;">Drag & drop or click to upload<br>
            <small>PNG, JPG, JPEG supported</small></p>
            <p style="color:#666; font-size:0.8rem;">
            Sample images: data/raw/train_images/</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    if uploaded:
        st.markdown('<div class="card"><h3>🔥 Grad-CAM Analysis</h3>',
                    unsafe_allow_html=True)
        with st.spinner("🧠 Running AI analysis..."):
            model    = load_model()
            enhanced = ben_graham(image)
            img_t    = preprocess(enhanced)
            cam, pred, probs = grad_cam(model, img_t)
            overlay  = overlay_gradcam(enhanced, cam)

        st.image(overlay, caption="Model Focus Heatmap",
                 use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card" style="height:100%; min-height:300px;
             display:flex; align-items:center; justify-content:center;
             text-align:center;">
            <div>
                <h2>🔥</h2>
                <h3>Grad-CAM Heatmap</h3>
                <p style="color:#aaa;">Upload an image to see<br>
                what the AI focuses on</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─── RESULTS ──────────────────────────────────────────────
if uploaded:
    label, emoji, advice, color = LABELS[pred]

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📊 Diagnosis Dashboard")

    r1, r2, r3 = st.columns([1,1,1], gap="large")

    with r1:
        st.markdown(f"""
        <div class="card" style="border-color:{color}40; text-align:center;">
            <h3>🎯 Diagnosis</h3>
            <div style="font-size:3rem;">{emoji}</div>
            <div style="font-size:1.8rem; font-weight:800;
                 color:{color};">{label}</div>
            <div style="color:#aaa; margin-top:0.5rem;">
                Stage {pred} of 4</div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        conf = float(probs[pred]) * 100
        st.markdown(f"""
        <div class="card" style="text-align:center;">
            <h3>📈 Confidence</h3>
            <div style="font-size:3rem; font-weight:800;
                 background:linear-gradient(90deg,#f953c6,#00c3ff);
                 -webkit-background-clip:text;
                 -webkit-text-fill-color:transparent;">
                 {conf:.1f}%</div>
            <div style="color:#aaa;">Model certainty</div>
        </div>
        """, unsafe_allow_html=True)

    with r3:
        st.markdown(f"""
        <div class="card" style="text-align:center;">
            <h3>⚕️ Severity</h3>
            <div style="margin:1rem 0;">
        """, unsafe_allow_html=True)
        st.progress(pred / 4)
        st.markdown(f"""
            <div style="color:#aaa; font-size:0.9rem;">
                {'▓' * (pred+1)}{'░' * (4-pred)} {pred+1}/5
            </div>
        </div></div>
        """, unsafe_allow_html=True)

    # Advice banner
    if pred == 0:
        st.success(f"✅ {advice}")
    elif pred <= 2:
        st.warning(f"⚠️ {advice}")
    else:
        st.error(f"🚨 {advice}")

    # All probabilities
    st.markdown('<div class="card"><h3>📊 Probability per DR Stage</h3>',
                unsafe_allow_html=True)
    cols = st.columns(5)
    for i, col in enumerate(cols):
        lbl, emj, _, clr = LABELS[i]
        prob_pct = float(probs[i]) * 100
        with col:
            st.markdown(f"""
            <div style="text-align:center; padding:0.5rem;
                 background:rgba(255,255,255,0.05);
                 border-radius:12px; border: 1px solid {clr}40;">
                <div style="font-size:1.5rem;">{emj}</div>
                <div style="font-size:1.2rem; font-weight:700;
                     color:{clr};">{prob_pct:.1f}%</div>
                <div style="font-size:0.75rem; color:#aaa;">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(float(probs[i]))
    st.markdown('</div>', unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div style="text-align:center; color:#666; font-size:0.8rem;
         margin-top:2rem;">
    ⚠️ This tool is for screening purposes only and does not replace
    professional medical diagnosis. Always consult a qualified ophthalmologist.
    </div>
    """, unsafe_allow_html=True)