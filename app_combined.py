import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="水稻病害综合诊断系统", layout="centered")
st.title("🌾 水稻病害综合诊断系统")
st.write("上传一张水稻叶片照片，AI 将同时进行病害分类和病斑定位。")

# 加载两个模型
cls_model = YOLO("D:/rice_project/runs/classify/train/weights/best.pt")      # 分类模型
det_model = YOLO("D:/rice_project/runs/detect/train-12/weights/best.pt")      # 检测模型

uploaded_file = st.file_uploader("📷 点击上传图片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="上传的图片", width=400)

    # ---------- 分类结果 ----------
    st.subheader("🔍 分类诊断结果")
    cls_results = cls_model(image)
    if cls_results and len(cls_results) > 0:
        probs = cls_results[0].probs
        if probs is not None:
            top1_idx = probs.top1
            top1_conf = float(probs.top1conf)
            cls_name = cls_model.names[top1_idx]
            st.markdown(f"**预测病害**：`{cls_name}`")
            st.markdown(f"**置信度**：`{top1_conf:.2%}`")
            # 显示各类别概率条
            for i, name in cls_model.names.items():
                prob = float(probs.data[i])
                st.progress(prob, text=f"{name}: {prob:.2%}")

    # ---------- 检测结果 ----------
    st.subheader("🔍 检测结果（病斑定位）")
    det_results = det_model(image)
    if det_results and len(det_results) > 0:
        annotated_img = det_results[0].plot()
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        st.image(annotated_img, caption="检测结果（带框）", width=400)

        if det_results[0].boxes is not None:
            for box in det_results[0].boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                st.write(f"✅ {det_model.names[cls_id]}（置信度：{conf:.2%}）")
        else:
            st.warning("⚠️ 未检测到病斑，可能是健康叶片或背景。")