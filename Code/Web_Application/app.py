import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import os
import re

# Try to import VnCoreNLP (optional, falls back to no preprocessing if not available)
try:
    from vncorenlp import VnCoreNLP
    VNCORENLP_AVAILABLE = True
except ImportError:
    VNCORENLP_AVAILABLE = False
    print("WARNING: VnCoreNLP not installed. Install with: pip install vncorenlp")
    print("Predictions may be less accurate without word segmentation!")

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "..", "Model_PhoBERT")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
VNCORENLP_PATH = os.path.join(BASE_DIR, "vncorenlp", "VnCoreNLP-1.1.1.jar")

# Initialize FastAPI
app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Load Model, Tokenizer, and VnCoreNLP
print("Loading model and tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on {device}")
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

# Load VnCoreNLP for word segmentation
rdrsegmenter = None
if VNCORENLP_AVAILABLE:
    try:
        if os.path.exists(VNCORENLP_PATH):
            rdrsegmenter = VnCoreNLP(VNCORENLP_PATH, annotators="wseg", max_heap_size='-Xmx500m')
            print("VnCoreNLP loaded successfully")
        else:
            print(f"VnCoreNLP jar not found at: {VNCORENLP_PATH}")
            print("Continuing without word segmentation (predictions may be less accurate)")
    except Exception as e:
        print(f"Could not load VnCoreNLP: {e}")
        print("Continuing without word segmentation")

# Label Mapping
LABEL_MAP = {
    0: "THỂ THAO",
    1: "SỨC KHỎE",
    2: "GIÁO DỤC",
    3: "PHÁP LUẬT",
    4: "KINH DOANH",
    5: "THƯ GIÃN",
    6: "KHOA HỌC CÔNG NGHỆ",
    7: "XE CỘ",
    8: "ĐỜI SỐNG",
    9: "THẾ GIỚI"
}

# Text preprocessing functions (same as Colab)
def clean_text(text):
    """Clean and normalize Vietnamese text"""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    # Keep Vietnamese characters and basic punctuation
    text = re.sub(r'[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ0-9.,?!]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess(text):
    """Preprocess text with word segmentation (if VnCoreNLP available)"""
    text = clean_text(text)
    
    if rdrsegmenter is not None:
        try:
            sentences = rdrsegmenter.tokenize(text)
            return " ".join([" ".join(sentence) for sentence in sentences])
        except Exception as e:
            print(f"Word segmentation failed: {e}")
            return text
    else:
        # No word segmentation available
        return text

class TextInput(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(input_data: TextInput):
    text = input_data.text
    if not text:
        return {"label": "Error", "confidence": 0.0}

    # 1. Tiền xử lý (Clean + Tách từ VnCoreNLP)
    processed_text = preprocess(text)
    
    # 2. Tạo danh sách các từ (Word List)
    word_list = processed_text.split()
    
    # 3. Tokenize
    encoding = tokenizer(
        word_list, 
        is_split_into_words=True, 
        padding=True, 
        truncation=True, 
        max_length=256, 
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in encoding.items()}

    # 4. Inference & Lấy Attention
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
        
        # Lấy Attention của lớp cuối cùng và trung bình cộng các heads
        last_layer_attention = outputs.attentions[-1]
        avg_attention = torch.mean(last_layer_attention, dim=1)
        
        # Lấy sự chú ý của token [CLS]
        cls_attention = avg_attention[0, 0, :] 

    # 5. Gộp điểm Attention từ Token về Word
    # Manual mapping vì không có Fast Tokenizer
    input_ids = inputs['input_ids'][0].cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    word_ids = [None] # Token đầu tiên là <s>
    
    current_token_idx = 1
    for i, word in enumerate(word_list):
        if current_token_idx >= len(tokens) - 1: 
            break
            
        sub_tokens = tokenizer.tokenize(word)
        
        for _ in sub_tokens:
            if current_token_idx < len(tokens) - 1:
                word_ids.append(i)
                current_token_idx += 1
            else:
                break
                
    while len(word_ids) < len(tokens):
        word_ids.append(None) 
    
    # Cộng dồn điểm attention
    word_scores = {}
    token_scores = cls_attention.tolist()
    
    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
            
        if word_id not in word_scores:
            word_scores[word_id] = 0.0
            
        word_scores[word_id] += token_scores[idx]

    # 6. Tạo danh sách kết quả
    explanation_list = []
    
    for idx, word in enumerate(word_list):
        score = word_scores.get(idx, 0.0)
        display_word = word.replace("_", " ")
        
        explanation_list.append({
            "word": display_word,
            "score": score
        })

    # 7. Trả về kết quả
    predicted_label = LABEL_MAP.get(predicted_class.item(), "Unknown")
    confidence_score = confidence.item()

    return {
        "label": predicted_label,
        "confidence": f"{confidence_score:.2%}",
        "segmentation_status": "Đã tách từ (VnCoreNLP)" if rdrsegmenter else "Chưa tách từ (Cảnh báo)",
        "explanation": explanation_list
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
