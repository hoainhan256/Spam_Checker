import re
import nltk
import numpy as np
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

import torch
from transformers import BertTokenizer, BertForSequenceClassification

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ===== LOAD MODELS =====
# Load LSTM model
lstm_model = load_model('spam_email_lstm_model.keras')

# Load BERT model
bert_tokenizer = BertTokenizer.from_pretrained('bert_model')
bert_model = BertForSequenceClassification.from_pretrained('bert_model')

# Load tokenizer used during LSTM training (giả định bạn đã lưu nó với pickle)
import pickle
with open('tokenizer.pickle', 'rb') as handle:
    lstm_tokenizer = pickle.load(handle)

MAX_LEN = 100  # Phải trùng lúc huấn luyện LSTM

# ===== PREPROCESSING =====
def preprocess_email(email):
    email = email.lower()
    email = re.sub(r'\W', ' ', email)
    email = re.sub(r'\s+[a-zA-Z]\s+', ' ', email)
    email = re.sub(r'\^[a-zA-Z]\s+', ' ', email)
    email = re.sub(r'\s+', ' ', email, flags=re.I)
    email = re.sub(r'^b\s+', '', email)
    email = ' '.join([word for word in email.split() if word not in stop_words])
    return email

# ===== PREDICT WITH LSTM =====
def predict_with_lstm(email_text):
    cleaned = preprocess_email(email_text)
    sequence = lstm_tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=MAX_LEN)
    prediction = lstm_model.predict(padded)[0][0]
    return prediction

# ===== PREDICT WITH BERT =====
def predict_with_bert(email_text):
    inputs = bert_tokenizer(email_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        prediction = probs[0][1].item()  # Xác suất là SPAM
    return prediction

# ===== DJANGO VIEW =====
@csrf_exempt
def home(request):
    # Khởi tạo giá trị mặc định
    email_text = ""
    result = None
    confidence = None
    model_choice = "lstm"  # Giá trị mặc định là LSTM
    model_used = "LSTM (Keras)"  # Giá trị mặc định cho model_used

    # Kiểm tra request.method == 'POST'
    if request.method == 'POST':
        email_text = request.POST.get('email_text', '')  # Lấy email từ form
        model_choice = request.POST.get('model_choice', 'lstm')  # Lấy model_choice từ form
        
        if not email_text:
            context = {'error': "Bạn chưa nhập nội dung email!"}
            return render(request, 'home/home.html', context)
        
        # Dự đoán dựa trên mô hình đã chọn
        if model_choice == 'bert':
            prediction = predict_with_bert(email_text)
            model_used = 'BERT Transformers'
        else:
            prediction = predict_with_lstm(email_text)
            model_used = 'LSTM (Keras)'

        result = 'SPAM' if prediction >= 0.5 else 'HAM'
        confidence = f"{prediction * 100:.2f}%"

    # Cập nhật context với các giá trị cần hiển thị
    context = {
        'email_text': email_text,
        'result': result,
        'confidence': confidence,
        'model_used': model_used,
        'model_choice': model_choice,
    }

    return render(request, 'home/home.html', context)
