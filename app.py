import streamlit as st
import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import tempfile
import time
import math
import json
from PIL import Image

# 頁面配置
st.set_page_config(page_title="Baseball Velocity AI Analyzer", layout="wide")

# AI 模型快取 (避免重複載入)
@st.cache_resource
def load_models():
    # MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    # YOLOv8 Ball Detection
    model = YOLO('yolov8n.pt')
    return pose, model, mp_pose

pose_model, yolo_model, mp_pose_lib = load_models()

# 標題與介紹
st.title("⚾ 棒球投球初速 AI 分析網頁版")
st.markdown("""
這個工具可以自動偵測投手的**釋球點 (T0)** 並計算**初速**。
支援電腦、手機與平板，直接上傳影片即可分析。
""")

# 側邊欄設定
st.sidebar.header("⚙️ 參數設定")
fps = st.sidebar.number_input("影片 FPS (如 iPhone 慢動作為 240)", value=240, min_value=1)
ball_dia = st.sidebar.number_input("棒球實際直徑 (cm)", value=7.3, format="%.1f")
conf_threshold = st.sidebar.slider("AI 偵測門檻 (Confidence)", 0.0, 1.0, 0.15)

# 檔案上傳
uploaded_file = st.sidebar.file_uploader("📂 上傳投球影片", type=['mp4', 'mov', 'avi'])

if uploaded_file is not None:
    # 建立暫存檔讀取影片
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty() # 用於動態顯示畫面
    
    # 初始化分析變數
    ball_history = []
    distances = []
    release_frame = -1
    velocity_kmh = 0.0
    current_frame_idx = 0
    trajectory_data = []
    
    # 建立處理進度條
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    st.info("🚀 正在進行 AI 分析與追蹤...")
    
    # 預備一個畫布容器
    container = st.container()
    col1, col2 = container.columns([3, 1])
    
    # 處理影片
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        current_frame_idx += 1
        h, w, _ = frame.shape
        
        # 1. MediaPipe Pose 偵測手腕
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_model.process(rgb_frame)
        
        wrist_pos = None
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            rw = lm[mp_pose_lib.PoseLandmark.RIGHT_WRIST]
            lw = lm[mp_pose_lib.PoseLandmark.LEFT_WRIST]
            target_wrist = rw if rw.visibility > lw.visibility else lw
            if target_wrist.visibility > 0.5:
                wrist_pos = (int(target_wrist.x * w), int(target_wrist.y * h))
                cv2.circle(frame, wrist_pos, 6, (0, 255, 0), -1)

        # 2. YOLOv8 偵測球
        ball_pos = None
        yolo_results = yolo_model.predict(frame, classes=[32], conf=conf_threshold, verbose=False)
        
        pixel_to_cm = 0
        for r in yolo_results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                ball_pos = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                ball_bbox = (int(x1), int(y1), int(x2), int(y2))
                
                # 空間定標
                bw, bh = x2 - x1, y2 - y1
                pixel_to_cm = ball_dia / ((bw + bh) / 2)
                
                # 領先邊緣追蹤
                leading_edge_pos = ball_pos
                if len(ball_history) > 0:
                    prev_ball_pos = ball_history[-1][1]
                    dx = ball_pos[0] - prev_ball_pos[0]
                    if dx > 2: leading_edge_pos = (ball_bbox[2], ball_pos[1])
                    elif dx < -2: leading_edge_pos = (ball_bbox[0], ball_pos[1])
                
                ball_history.append((current_frame_idx, ball_pos, leading_edge_pos, pixel_to_cm))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                break

        # 3. T0 偵測與初速計算
        if wrist_pos and ball_pos and release_frame == -1:
            dist = math.sqrt((ball_pos[0]-wrist_pos[0])**2 + (ball_pos[1]-wrist_pos[1])**2)
            distances.append((current_frame_idx, dist))
            if len(distances) > 5 and dist - distances[-2][1] > 12:
                release_frame = current_frame_idx

        # 計算速度 (一旦有了 T0)
        if release_frame != -1 and velocity_kmh == 0.0:
            t0_data = next((x for x in ball_history if x[0] == release_frame), None)
            target_data = next((x for x in ball_history if x[0] >= release_frame + 3), None)
            
            if t0_data and target_data:
                f0, _, p0, _ = t0_data
                ft, _, pt, p_to_cm = target_data
                p_dist = math.sqrt((pt[0]-p0[0])**2 + (pt[1]-p0[1])**2)
                time_s = (ft - f0) / fps
                velocity_kmh = ((p_dist * p_to_cm) / 100 / time_s) * 3.6
                
        # 4. 收集軌跡數據 (用於導出)
        if release_frame != -1 and ball_pos and len(trajectory_data) < 10:
            trajectory_data.append({
                "frame": current_frame_idx,
                "x": ball_pos[0], "y": ball_pos[1],
                "t": (current_frame_idx - release_frame) / fps
            })

        # 繪製 UI 資訊到幀上
        if release_frame != -1:
            cv2.putText(frame, f"T0 Detected: Frame {release_frame}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if velocity_kmh > 0:
            cv2.putText(frame, f"Velocity: {velocity_kmh:.1f} km/h", (20, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # 顯示處理中的畫面
        st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        progress_bar.progress(current_frame_idx / total_frames)

    cap.release()
    st.success("✅ 分析完成！")
    
    # 顯示最終數據報表
    st.subheader("📊 分析結果")
    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("釋球點 T0", f"{release_frame} 幀" if release_frame != -1 else "未偵測")
    res_col2.metric("初始速度 (km/h)", f"{velocity_kmh:.1f} km/h" if velocity_kmh > 0 else "--")
    res_col3.metric("處理總幀數", f"{total_frames}")

    # 導出 JSON 功能
    if trajectory_data:
        json_str = json.dumps({
            "metadata": {"velocity_kmh": velocity_kmh, "fps": fps, "t0_frame": release_frame},
            "trajectory": trajectory_data
        }, indent=4)
        st.download_button(
            label="📥 導出軌跡數據 (JSON)",
            data=json_str,
            file_name=f"velocity_analysis_{int(time.time())}.json",
            mime="application/json"
        )
else:
    st.info("💡 請在左側上傳投球影片開始分析。")
    st.image("https://images.unsplash.com/photo-1508344928928-7165b67de128?auto=format&fit=crop&w=1000&q=80", caption="準備分析您的球速")
