# Hướng Dẫn Cài Đặt VnCoreNLP để Kết Quả Chính Xác

## ⚠️ VẤN ĐỀ
PhoBERT được train với dữ liệu **đã word-segmented** (tách từ tiếng Việt).  
Không tách từ → Kết quả sai lệch!

## ✅ GIẢI PHÁP: Cài VnCoreNLP

### Bước 1: Cài đặt package
```bash
pip install vncorenlp
```

### Bước 2: Tải VnCoreNLP JAR
Chạy lần lượt các lệnh sau trong PowerShell:

```powershell
# Tạo thư mục vncorenlp
mkdir "c:\Users\DELL\Downloads\NLP Project\NLP Project\Code\vncorenlp"
mkdir "c:\Users\DELL\Downloads\NLP Project\NLP Project\Code\vncorenlp\models\wordsegmenter"

# Download VnCoreNLP JAR
cd "c:\Users\DELL\Downloads\NLP Project\NLP Project\Code\vncorenlp"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar" -OutFile "VnCoreNLP-1.1.1.jar"

# Download word segmenter models
cd "models\wordsegmenter"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab" -OutFile "vi-vocab"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr" -OutFile "wordsegmenter.rdr"
```

### Bước 3: Kiểm tra cấu trúc thư mục
```
Code/
├── app.py
├── vncorenlp/                     ← MỚI TẠO
│   ├── VnCoreNLP-1.1.1.jar        ← FILE JAR
│   └── models/
│       └── wordsegmenter/
│           ├── vi-vocab           ← MODEL FILE
│           └── wordsegmenter.rdr  ← MODEL FILE
├── templates/
└── static/
```

### Bước 4: Khởi động lại server
```bash
uvicorn app:app --reload
```

### Bước 5: Kiểm tra log
Bạn sẽ thấy:
```
✅ Model loaded on cuda
✅ VnCoreNLP loaded successfully  ← QUAN TRỌNG!
```

---

## 📝 LƯU Ý

### Nếu KHÔNG CÓ VnCoreNLP:
- App vẫn chạy nhưng kết quả **kém chính xác**
- Log hiển thị: `⚠️ WARNING: VnCoreNLP not installed`

### Nếu CÓ VnCoreNLP:
- Kết quả **chính xác như Colab**
- Text được tách từ trước khi đưa vào model

---

## 🔍 So Sánh

### Không tách từ:
```
Input: "Thủ tướng Phạm Minh Chính"
→ Model nhận: "thủ tướng phạm minh chính" (sai)
```

### Có tách từ (VnCoreNLP):
```
Input: "Thủ tướng Phạm Minh Chính"
→ Sau tách: "thủ_tướng phạm_minh_chính"
→ Model nhận đúng → Kết quả chính xác!
```

---

## ⚡ NHANH GỌN (Nếu có Python script):

Tạo file `setup_vncorenlp.py`:

```python
import os
import urllib.request

BASE_DIR = "c:/Users/DELL/Downloads/NLP Project/NLP Project/Code/vncorenlp"
os.makedirs(f"{BASE_DIR}/models/wordsegmenter", exist_ok=True)

files = {
    "VnCoreNLP-1.1.1.jar": "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar",
    "models/wordsegmenter/vi-vocab": "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab",
    "models/wordsegmenter/wordsegmenter.rdr": "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr"
}

for file, url in files.items():
    print(f"Downloading {file}...")
    urllib.request.urlretrieve(url, f"{BASE_DIR}/{file}")
    print(f"✅ {file} downloaded")

print("\n✅ VnCoreNLP setup complete!")
```

Chạy: `python setup_vncorenlp.py`
