# AudioGhost AI 啟動指南

## 快速啟動

### 1. 啟動 Redis (使用 Docker)
```powershell
cd d:\sam_audio
docker-compose up -d
```

### 2. 建立 Anaconda 環境
```powershell
# 建立新環境 (Python 3.11+ 必要)
conda create -n audioghost python=3.11 -y

# 啟動環境
conda activate audioghost
```

### 3. 安裝 PyTorch + xformers (CUDA 12.4)
```powershell
pip install torch==2.9.0+cu126 torchvision==0.24.0+cu126 torchaudio==2.9.0+cu126 --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple
```

### 4. 安裝 FFmpeg (TorchCodec 需要)
```powershell
conda install -c conda-forge ffmpeg -y
```

### 5. 安裝 SAM Audio


```powershell
cd d:\sam_audio
pip install .
```

### 6. 安裝 Backend 依賴
```powershell
cd d:\sam_audio\backend
pip install -r requirements.txt
```

### 7. 啟動 Backend API
```powershell
cd d:\sam_audio\backend
uvicorn main:app --reload --port 9722
```

### 8. 啟動 Celery Worker (新終端機)

```powershell
conda activate audioghost
cd d:\sam_audio\backend
celery -A workers.celery_app worker --loglevel=info --pool=solo
```

### 9. 啟動 Frontend (新終端機)
```powershell
cd d:\sam_audio\frontend
npm run dev
```

### 10. 開啟瀏覽器
訪問 http://localhost:3000




## 首次使用

1. 點擊右上角 "Connect HuggingFace" 按鈕
2. 前往 https://huggingface.co/facebook/sam-audio-large 申請存取權限
3. 建立 Access Token: https://huggingface.co/settings/tokens
4. 將 Token 貼入並連接

## 功能使用

- **上傳音訊**：拖放或點擊上傳區域
- **語意分離**：選擇快捷按鈕或輸入自訂描述
- **時間鎖定**：在波形圖上選取區域
- **三軌輸出**：Original / Ghost / Clean
