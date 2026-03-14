import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import math
import mediapipe as mp

# 終極導入方案：嘗試多種可能的路徑
mp_pose = None
mp_drawing = None

try:
    # 方案 A: 標準導入
    import mediapipe.solutions.pose as tmp_pose
    import mediapipe.solutions.drawing_utils as tmp_drawing
    mp_pose = tmp_pose
    mp_drawing = tmp_drawing
except:
    try:
        # 方案 B: 內部路徑導入 (針對某些 Windows 安裝)
        from mediapipe.python.solutions import pose as tmp_pose
        from mediapipe.python.solutions import drawing_utils as tmp_drawing
        mp_pose = tmp_pose
        mp_drawing = tmp_drawing
    except:
        mp_pose = None
        mp_drawing = None

from ultralytics import YOLO
import numpy as np
import json
import os
import time

class BaseballVelocityAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("棒球投球初速分析器 (專業回放版)")

        # AI 模型初始化
        try:
            if mp_pose is not None:
                self.mp_pose = mp_pose
                self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            else:
                raise Exception("無法載入 MediaPipe 模組路徑")
        except Exception as e:
            messagebox.showerror("AI 初始化失敗", f"MediaPipe 啟動失敗: {e}\n請嘗試在終端機執行: python -m pip install mediapipe --upgrade")
            self.pose = None
        
        # YOLOv8 載入
        try:
            self.model = YOLO('yolov8n.pt') 
            self.yolo_available = True
        except Exception as e:
            print(f"YOLO 載入失敗: {e}")
            self.yolo_available = False

        # 影片與狀態變數
        self.cap = None
        self.playing = False
        self.frame = None
        self.current_frame_idx = 0
        self.video_frames = [] # 用於回放的緩存
        
        # 回放變數
        self.replay_mode = False
        self.replay_start_idx = 0
        self.replay_end_idx = 0
        self.replay_current_idx = 0
        
        # 使用者參數
        self.fps_var = tk.StringVar(value="240")
        self.ball_dia_var = tk.StringVar(value="7.3") # cm
        
        # 追蹤與分析變數
        self.enable_ai = tk.BooleanVar(value=True)
        self.wrist_pos = None
        self.ball_pos = None
        self.ball_bbox = None
        self.ball_pixel_diameter = 0
        self.pixel_to_cm = 0
        
        self.distances = []
        self.ball_history = []
        self.release_frame = -1
        self.velocity_kmh = 0.0
        self.trajectory_data = [] # 用於導出 JSON (x, y, t)
        self.points = []

        # UI 介面設定
        self.setup_ui()

    def setup_ui(self):
        # 1. 儀表板區塊 (Dashboard)
        self.dashboard = tk.Frame(self.root, bg="#2c3e50", height=80)
        self.dashboard.pack(side=tk.TOP, fill=tk.X)
        
        # 幀數顯示
        self.lbl_frame_info = tk.Label(self.dashboard, text="FRAME: 0000", fg="white", bg="#2c3e50", font=("Consolas", 14, "bold"))
        self.lbl_frame_info.pack(side=tk.LEFT, padx=20, pady=10)
        
        # 狀態顯示
        self.lbl_ai_status = tk.Label(self.dashboard, text="STATUS: IDLE", fg="#ecf0f1", bg="#2c3e50", font=("Arial", 12))
        self.lbl_ai_status.pack(side=tk.LEFT, padx=20)
        
        # 初速顯示 (核心數據)
        self.lbl_v_display = tk.Label(self.dashboard, text="INITIAL VELOCITY: -- km/h", fg="#2ecc71", bg="#2c3e50", font=("Arial", 16, "bold"))
        self.lbl_v_display.pack(side=tk.RIGHT, padx=30)

        # 2. 控制列 (Control Bar)
        self.control_bar = tk.Frame(self.root, bg="#ecf0f1")
        self.control_bar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(self.control_bar, text="開啟影片", command=self.open_video, width=10).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(self.control_bar, text="播放/暫停", command=self.toggle_play, width=10).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Checkbutton(self.control_bar, text="AI 追蹤", variable=self.enable_ai, bg="#ecf0f1").pack(side=tk.LEFT, padx=10)
        
        tk.Label(self.control_bar, text="FPS:", bg="#ecf0f1").pack(side=tk.LEFT, padx=2)
        tk.Entry(self.control_bar, textvariable=self.fps_var, width=5).pack(side=tk.LEFT, padx=2)
        tk.Label(self.control_bar, text="球徑(cm):", bg="#ecf0f1").pack(side=tk.LEFT, padx=2)
        tk.Entry(self.control_bar, textvariable=self.ball_dia_var, width=5).pack(side=tk.LEFT, padx=2)
        
        tk.Button(self.control_bar, text="重設分析", command=self.reset_analysis, width=10).pack(side=tk.RIGHT, padx=5)
        tk.Button(self.control_bar, text="導出數據", command=self.export_data_to_json, width=10).pack(side=tk.RIGHT, padx=5)

        # 3. 畫布
        self.canvas = tk.Canvas(self.root, bg="black")
        self.canvas.pack(expand=True, fill=tk.BOTH)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

    def reset_analysis(self):
        self.current_frame_idx = 0
        self.release_frame = -1
        self.distances = []
        self.ball_history = []
        self.trajectory_data = []
        self.video_frames = []
        self.replay_mode = False
        self.velocity_kmh = 0.0
        self.lbl_frame_info.config(text="FRAME: 0000")
        self.lbl_ai_status.config(text="STATUS: RESET")
        self.lbl_v_display.config(text="INITIAL VELOCITY: -- km/h")
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if ret: self.show_frame(frame)

    def open_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                messagebox.showerror("錯誤", "無法開啟影片檔案")
                return
            self.reset_analysis()
            self.lbl_ai_status.config(text="STATUS: LOADED")
            
    def toggle_play(self):
        if self.cap is None: return
        self.playing = not self.playing
        if self.playing:
            self.lbl_ai_status.config(text="STATUS: ANALYZING")
            self.update_video()
        else:
            self.lbl_ai_status.config(text="STATUS: PAUSED")

    def update_video(self):
        if not self.playing: return

        # 回放模式邏輯
        if self.replay_mode:
            if self.replay_current_idx < len(self.video_frames):
                frame = self.video_frames[self.replay_current_idx]
                self.show_frame(frame)
                self.lbl_frame_info.config(text=f"REPLAY: {self.replay_start_idx + self.replay_current_idx}")
                self.replay_current_idx += 1
                # 慢動作回放 (100ms 一幀，約 10 FPS)
                self.root.after(100, self.update_video)
            else:
                self.replay_current_idx = 0 # 循環回放
                self.root.after(100, self.update_video)
            return

        # 正常播放與分析模式
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame_idx += 1
                self.lbl_frame_info.config(text=f"FRAME: {self.current_frame_idx:04d}")
                
                # 緩存幀以供後續回放 (限制緩存大小避免內存爆炸)
                self.video_frames.append(frame.copy())
                if len(self.video_frames) > 500: self.video_frames.pop(0)

                if self.enable_ai.get():
                    frame = self.process_ai(frame)
                
                self.show_frame(frame)
                
                # 檢查是否偵測到釋球且已過足夠幀數來計算速度與啟動回放
                if self.release_frame != -1 and self.current_frame_idx == self.release_frame + 10:
                    self.start_auto_replay()
                else:
                    self.root.after(10, self.update_video)
            else:
                self.playing = False
                self.lbl_ai_status.config(text="STATUS: END")

    def process_ai(self, frame):
        if self.pose is None: return frame
        
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        self.wrist_pos = None
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            rw = lm[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            lw = lm[self.mp_pose.PoseLandmark.LEFT_WRIST]
            target_wrist = rw if rw.visibility > lw.visibility else lw
            if target_wrist.visibility > 0.5:
                self.wrist_pos = (int(target_wrist.x * w), int(target_wrist.y * h))
                cv2.circle(frame, self.wrist_pos, 6, (0, 255, 0), -1)

        # 2. YOLOv8 偵測球
        self.ball_pos = None
        if self.yolo_available:
            yolo_results = self.model.predict(frame, classes=[32], conf=0.15, verbose=False) # 降低門檻以防漏抓
            for r in yolo_results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    self.ball_pos = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    self.ball_bbox = (int(x1), int(y1), int(x2), int(y2))
                    
                    bw = x2 - x1
                    bh = y2 - y1
                    self.ball_pixel_diameter = (bw + bh) / 2
                    try:
                        self.pixel_to_cm = float(self.ball_dia_var.get()) / self.ball_pixel_diameter
                    except: pass
                    
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                    break 

        # 3. 記錄球體歷史與偵測 T0
        if self.ball_pos:
            # 領先邊緣追蹤
            leading_edge_pos = self.ball_pos
            if len(self.ball_history) > 0:
                prev_ball_pos = self.ball_history[-1][1]
                dx = self.ball_pos[0] - prev_ball_pos[0]
                if dx > 2: leading_edge_pos = (self.ball_bbox[2], self.ball_pos[1])
                elif dx < -2: leading_edge_pos = (self.ball_bbox[0], self.ball_pos[1])
            
            self.ball_history.append((self.current_frame_idx, self.ball_pos, leading_edge_pos))

            # 偵測釋球點 T0 (需要手腕位置)
            if self.wrist_pos and self.release_frame == -1:
                dist = math.sqrt((self.ball_pos[0]-self.wrist_pos[0])**2 + (self.ball_pos[1]-self.wrist_pos[1])**2)
                self.distances.append((self.current_frame_idx, dist))
                
                if len(self.distances) > 5:
                    if dist - self.distances[-2][1] > 12: # 稍微調低閥值提高敏感度
                        self.release_frame = self.current_frame_idx
                        self.lbl_ai_status.config(text=f"STATUS: T0 DETECTED @ {self.release_frame}")
                        print(f"偵測到釋球點 T0: 第 {self.release_frame} 幀")

            # 收集 JSON 數據 (離手後 10 幀)
            if self.release_frame != -1 and self.current_frame_idx >= self.release_frame:
                if len(self.trajectory_data) < 10:
                    try:
                        fps = float(self.fps_var.get())
                        t_sec = (self.current_frame_idx - self.release_frame) / fps
                        self.trajectory_data.append({
                            "frame": self.current_frame_idx,
                            "x_px": self.ball_pos[0],
                            "y_px": self.ball_pos[1],
                            "t_sec": round(t_sec, 6)
                        })
                    except: pass
        else:
            # 即使當前幀沒偵測到球，如果已經過了釋球點，也可能需要維持軌跡數據的連續性
            # 這裡暫不處理漏抓格，但未來可以考慮插值
            pass

        # 4. 初速計算
        if self.release_frame != -1 and self.velocity_kmh == 0.0:
            self.calculate_velocity()
            
        # 繪製除錯資訊
        if self.enable_ai.get():
            debug_text = f"Ball: {'OK' if self.ball_pos else 'LOST'} | Wrist: {'OK' if self.wrist_pos else 'LOST'}"
            cv2.putText(frame, debug_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if self.velocity_kmh > 0:
            cv2.putText(frame, f"{self.velocity_kmh:.1f} km/h", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        return frame

    def calculate_velocity(self):
        t0_data = None
        t_target_data = None
        
        # 尋找 T0 幀的數據
        for f_idx, b_pos, l_pos in self.ball_history:
            if f_idx == self.release_frame:
                t0_data = (f_idx, l_pos)
                break
        
        if not t0_data: return

        # 尋找 T0 之後最接近且至少間隔 3 幀的有效數據 (例如 T+3, T+4, T+5...)
        for f_idx, b_pos, l_pos in self.ball_history:
            if f_idx >= self.release_frame + 3:
                t_target_data = (f_idx, l_pos)
                break
        
        if t0_data and t_target_data:
            f0, p0 = t0_data
            ft, pt = t_target_data
            
            pixel_dist = math.sqrt((pt[0]-p0[0])**2 + (pt[1]-p0[1])**2)
            real_dist_m = (pixel_dist * self.pixel_to_cm) / 100
            
            try:
                frame_diff = ft - f0
                fps = float(self.fps_var.get())
                time_s = frame_diff / fps
                self.velocity_kmh = (real_dist_m / time_s) * 3.6
                self.lbl_v_display.config(text=f"INITIAL VELOCITY: {self.velocity_kmh:.1f} km/h")
                print(f"計算成功: 位移 {pixel_dist:.1f}px, 幀差 {frame_diff}, 時速 {self.velocity_kmh:.1f}")
            except Exception as e:
                print(f"計算失敗: {e}")

    def start_auto_replay(self):
        # 截取 T0 前後 10 幀的緩存
        start_idx = max(0, len(self.video_frames) - 21) # 21 幀 = T0 前 10 + T0 + 後 10
        self.video_frames = self.video_frames[start_idx:]
        self.replay_start_idx = self.current_frame_idx - 20
        self.replay_mode = True
        self.replay_current_idx = 0
        self.lbl_ai_status.config(text="STATUS: AUTO REPLAY", fg="#f1c40f")
        print("啟動自動回放...")

    def export_data_to_json(self):
        if not self.trajectory_data:
            messagebox.showwarning("警告", "無軌跡數據可導出")
            return
        
        filename = f"trajectory_{int(time.time())}.json"
        export_content = {
            "metadata": {
                "velocity_kmh": round(self.velocity_kmh, 2),
                "fps": self.fps_var.get(),
                "ball_diameter_cm": self.ball_dia_var.get(),
                "release_frame": self.release_frame
            },
            "trajectory": self.trajectory_data
        }
        
        with open(filename, 'w') as f:
            json.dump(export_content, f, indent=4)
        
        messagebox.showinfo("成功", f"數據已導出至 {filename}")

    def show_frame(self, frame):
        self.frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(self.frame_rgb)
        if self.canvas.winfo_width() != img.width:
            self.canvas.config(width=img.width, height=img.height)
        self.photo = ImageTk.PhotoImage(image=img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def on_canvas_click(self, event):
        if self.playing: return
        # 點擊處理維持原樣...
        pass

if __name__ == "__main__":
    root = tk.Tk()
    app = BaseballVelocityAnalyzer(root)
    root.mainloop()
