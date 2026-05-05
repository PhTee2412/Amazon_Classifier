import os
import re
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import gradio as gr


stop_words = set([
    'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'am', 'to', 'for',
    'in', 'on', 'at', 'by', 'with', 'this', 'that', 'of', 'from', 'it',
    'its', 'they', 'them', 'their', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'which', 'who', 'whom',
    'whose', 'where', 'when', 'how', 'all', 'any', 'both', 'each',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', 'should', 'now', 'i', 'me', 'my', 'myself',
    'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
    'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
    'themselves'
])

top_brands_list = [
    'western digital', 'apple', 'samsung', 'dell', 'hp', 'lenovo', 'asus',
    'acer', 'sony', 'logitech', 'jbl', 'bose', 'tp-link', 'sandisk',
    'xiaomi', 'oppo', 'vivo', 'oneplus', 'google', 'amazon', 'nintendo',
    'playstation', 'xbox', 'msi', 'gigabyte'
]

premium_brands = [
    'apple', 'samsung', 'sony', 'dell', 'hp', 'bose', 'playstation',
    'xbox', 'nintendo'
]

def clean_text_pro(text):
    """Làm sạch tên sản phẩm: lowercase, xoá ký tự đặc biệt, loại bỏ stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\.]', ' ', text)
    words = [w for w in text.split() if w not in stop_words and len(w) > 1]
    return " ".join(words)

def get_brand_tier(name):
    """Phân hạng thương hiệu: 2 = cao cấp, 1 = phổ biến, 0 = khác."""
    name_lower = str(name).lower()
    for brand in premium_brands:
        if re.search(r'\b' + re.escape(brand) + r'\b', name_lower):
            return 2
    for brand in top_brands_list:
        if re.search(r'\b' + re.escape(brand) + r'\b', name_lower):
            return 1
    return 0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')

print("Loading model and transformers...")
model = joblib.load(os.path.join(MODEL_DIR, 'amazon_classifier.pkl'))
le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
pca = joblib.load(os.path.join(MODEL_DIR, 'pca_pro.pkl'))
svd = joblib.load(os.path.join(MODEL_DIR, 'svd_pro.pkl'))
tfidf = joblib.load(os.path.join(MODEL_DIR, 'tfidf_pro.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler_pro.pkl'))
numeric_cols = joblib.load(os.path.join(MODEL_DIR, 'numeric_cols.pkl'))

# Load sentence-transformer model
sent_model = SentenceTransformer('all-MiniLM-L6-v2')
print("All artifacts loaded successfully!\n")

# ============================ HÀM DỰ ĐOÁN ============================

def preprocess_and_predict(name, price, rating, reviews_count, bought_count, actual_price):
    """
    Nhận dữ liệu người dùng, biến đổi thành vector đặc trưng,
    chạy dự đoán và trả về dictionary {tên_lớp: xác_suất}.
    """
    # 1. Văn bản
    text = clean_text_pro(str(name))
    emb = pca.transform(sent_model.encode([text]))         
    tfidf_vec = svd.transform(tfidf.transform([text]))      

    # 2. Biến số
    feat = {
        'price_log': np.log1p(price),
        'rating_clean': float(rating) if rating else 4.2,
        'reviews_log': np.log1p(float(reviews_count)) if reviews_count else 0,
        'bought_count_log': np.log1p(bought_count),
        'discount_pct': (actual_price - price) / actual_price if actual_price > price else 0,
        'social_proof': np.log1p(float(rating) * float(reviews_count)) if rating and reviews_count else 0,
        'word_count': len(str(name).split()),
        'brand_tier': get_brand_tier(name),
        'is_best_seller_flag': 0,
        'has_coupon': 0
    }

    # 3. Scale & chọn các cột đã được tối ưu qua VIF
    scaler_input_cols = [
        'price_log', 'rating_clean', 'reviews_log', 'bought_count_log',
        'discount_pct', 'social_proof', 'word_count', 'brand_tier',
        'is_best_seller_flag', 'has_coupon'
    ]
    df_input = pd.DataFrame([[feat.get(c, 0) for c in scaler_input_cols]], columns=scaler_input_cols)
    scaled_input = pd.DataFrame(scaler.transform(df_input), columns=scaler_input_cols)
    final_num = scaled_input[numeric_cols].values        # chỉ giữ cột quan trọng

    # 4. Ghép toàn bộ đặc trưng
    X = np.hstack([final_num, emb, tfidf_vec])

    # 5. Dự đoán
    proba = model.predict_proba(X)[0]
    result = {le.classes_[i]: float(proba[i]) for i in range(len(le.classes_))}
    return result



description_text = (
    "Nhập thông tin sản phẩm từ Amazon để mô hình phân loại vào 1 trong 10 danh mục. "
    "Mô hình sử dụng LightGBM + Sentence‑BERT, đạt độ chính xác ~96%."
)

# Giao diện
demo = gr.Interface(
    fn=preprocess_and_predict,
    inputs=[
        gr.Textbox(
            label="Tên sản phẩm",
            placeholder="VD: Apple iPhone 15 Pro Max 256GB Titanium",
            lines=2
        ),
        gr.Number(label="Giá hiện tại (discounted)", value=0.0, precision=2),
        gr.Slider(label="Rating (1–5)", minimum=1.0, maximum=5.0, step=0.1, value=4.5),
        gr.Number(label="Số reviews", value=100, precision=0),
        gr.Number(label="Số lượng mua tháng trước", value=0, precision=0),
        gr.Number(label="Giá gốc (listed)", value=0.0, precision=2),
    ],
    outputs=gr.Label(label="Xác suất dự đoán", num_top_classes=3),
    title="Amazon Product Classifier",
    description=description_text,
    article="### Cách hoạt động\nMô hình phân tích tên sản phẩm, giá, rating và reviews để dự đoán danh mục. "
            "Được huấn luyện trên hơn 40.000 mẫu với LightGBM và Optuna tuning."
)

if __name__ == "__main__":
    demo.launch(share=False) 