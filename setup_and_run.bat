@echo off
setlocal
echo ==========================================
echo    棒球測速軟體 - 專屬環境自動設定
echo ==========================================

:: 1. 偵測 Python 3.12 路徑 (優先使用您的 3.12)
set PY_PATH="C:\Users\garyl\AppData\Local\Programs\Python\Python312\python.exe"

echo [1/3] 正在建立專屬隔離環境 (.venv)...
%PY_PATH% -m venv .venv

echo [2/3] 正在專屬環境中安裝必要套件...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install opencv-python pillow mediapipe ultralytics

echo [3/3] 啟動軟體中...
.venv\Scripts\python.exe velocity_analyzer.py

echo ==========================================
echo    分析完成，按任意鍵關閉。
pause