import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="湘农智诊 · 水稻病害识别", layout="centered")
st.title("🌾 湘农智诊 · 水稻病害识别系统")
st.write("上传一张水稻叶片照片，AI 将自动判断病害类别和置信度。")

model = YOLO("D:/rice_project/runs/classify/train/weights/best.pt")

uploaded_file = st.file_uploader("📷 点击上传图片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="上传的图片", width=400)

    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = model(img_cv)

    if results and len(results) > 0:
        probs = results[0].probs
        if probs is not None:
            top1_idx = probs.top1
            top1_conf = float(probs.top1conf)
            names = model.names

            st.subheader("🔍 诊断结果")
            st.markdown(f"**预测病害**：`{names[top1_idx]}`")
            st.markdown(f"**置信度**：`{top1_conf:.2%}`")

            st.subheader("📊 各类别概率")
            for i, name in names.items():
                prob = float(probs.data[i])
                st.progress(prob, text=f"{name}: {prob:.2%}")
        else:
            st.warning("⚠️ 模型未返回有效预测结果，请换一张图片重试。")
    else:
        st.warning("⚠️ 图片无法识别，请上传清晰的水稻叶片照片。")