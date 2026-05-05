import os
import re
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import lightgbm as lgb
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

# ============================ HELPER FUNCTIONS ============================
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

model = joblib.load(os.path.join(MODEL_DIR, 'amazon_classifier.pkl'))
le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
pca = joblib.load(os.path.join(MODEL_DIR, 'pca_pro.pkl'))
svd = joblib.load(os.path.join(MODEL_DIR, 'svd_pro.pkl'))
tfidf = joblib.load(os.path.join(MODEL_DIR, 'tfidf_pro.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler_pro.pkl'))
numeric_cols = joblib.load(os.path.join(MODEL_DIR, 'numeric_cols.pkl'))

sent_model = SentenceTransformer('all-MiniLM-L6-v2')



def preprocess_and_predict(name, price, rating, reviews_count, bought_count, actual_price):
    """
    Nhận dữ liệu từ UI, biến đổi thành vector đặc trưng và dự đoán.
    Trả về dictionary {tên_lớp: xác_suất} cho Gradio hiển thị.
    """
    # 1. Văn bản (Text Features)
    text = clean_text_pro(str(name))
    emb = pca.transform(sent_model.encode([text]))         
    tfidf_vec = svd.transform(tfidf.transform([text]))      

    # 2. Biến số (Numerical Features)
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

    # 3. Scale các đặc trưng liên tục (Numerical scaling)
    scaler_input_cols = [
        'price_log', 'rating_clean', 'reviews_log', 'bought_count_log',
        'discount_pct', 'social_proof', 'word_count', 'brand_tier',
        'is_best_seller_flag', 'has_coupon'
    ]
    df_input = pd.DataFrame([[feat.get(c, 0) for c in scaler_input_cols]], columns=scaler_input_cols)
    scaled_input = pd.DataFrame(scaler.transform(df_input), columns=scaler_input_cols)
    final_num = scaled_input[numeric_cols].values # Chỉ giữ cột có sau VIF

    X = np.hstack([final_num, emb, tfidf_vec])

    # 5. Dự đoán
    proba = model.predict_proba(X)[0]
    result = {le.classes_[i]: float(proba[i]) for i in range(len(le.classes_))}
    sorted_result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
    max_conf = max(proba)

    if max_conf < 0.5:
        top_result = dict(list(sorted_result.items())[:1])
        warning_msg = f"**Cảnh báo**: Mức độ tin cậy cao nhất chỉ đạt **{max_conf*100:.1f}%**. Sản phẩm này có thông tin không rõ ràng hoặc có thể thuộc danh mục khác."
    else:
        top_result = dict(list(sorted_result.items())[:3])
        warning_msg = f"Dự đoán thành công với mức độ tin cậy (**{max_conf*100:.1f}%**)."

    return top_result, warning_msg


custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
)

with gr.Blocks(theme=custom_theme, title="Amazon Product Classifier", css="footer {visibility: hidden}") as demo:
    gr.Markdown(
        """
        # Ứng dụng Phân loại Sản phẩm Amazon (Amazon Product Classifier)
        **Mô hình phân loại Machine Learning** dự đoán danh mục sản phẩm trên Amazon thông qua **LightGBM** và **Sentence-BERT**.
        Mô hình đạt độ chính xác **~96%**, phân loại thành 1 trong 10 danh mục khác nhau.
        """
    )
    
    with gr.Row():
        # Cột Input
        with gr.Column(scale=2):
            gr.Markdown("###  Nhập thông tin Sản phẩm")
            name = gr.Textbox(
                label="Tên sản phẩm",
                placeholder="VD: Apple iPhone 15 Pro Max 256GB Titanium, Blue",
                lines=2
            )
            
            with gr.Row():
                price = gr.Number(label="Giá khuyến mãi ($)", value=0.0, precision=2)
                actual_price = gr.Number(label="Giá gốc ($)", value=0.0, precision=2)
            
            with gr.Row():
                rating = gr.Slider(label="Điểm đánh giá (Rating)", minimum=1.0, maximum=5.0, step=0.1, value=4.5)
                reviews_count = gr.Number(label="Số lượt đánh giá (Reviews)", value=100, precision=0)
                bought_count = gr.Number(label="Lượt mua tháng trước", value=0, precision=0)
            
            predict_btn = gr.Button(" Dự đoán Danh mục", variant="primary", size="lg")
            
        # Cột Output
        with gr.Column(scale=1):
            gr.Markdown("###  Kết quả Dự đoán")
            output_label = gr.Label(label="Top 3 danh mục phù hợp nhất", num_top_classes=3)
            warning_text = gr.Markdown(value="", visible=True)
            
    # Sự kiện khi bấm nút Predict
    predict_btn.click(
        fn=preprocess_and_predict,
        inputs=[name, price, rating, reviews_count, bought_count, actual_price],
        outputs=[output_label, warning_text]
    )
    
    # Bảng Ví dụ để test nhanh
    gr.Markdown("###  Ví dụ (Click để điền tự động)")
    gr.Examples(
        examples=[
            ["Apple iPhone 15 Pro Max 256GB Titanium", 1199, 4.8, 5000, 1000, 1299],
            ["Samsung 32-inch Odyssey G5 Gaming Monitor", 300, 4.5, 1200, 500, 350],
            ["Logitech MX Master 3S Wireless Mouse", 99, 4.9, 3000, 800, 99],
            ["PlayStation 5 Console (PS5)", 499, 4.9, 55000, 5000, 499],
            ["Premium Yoga Mat for Exercise", 20, 4.0, 150, 50, 30]
        ],
        inputs=[name, price, rating, reviews_count, bought_count, actual_price],
        label="Dữ liệu mẫu"
    )

    gr.Markdown(
        """
        ---
        **Tác giả**: Phương Thảo | **Email**: phtee2412@gmail.com
        """
    )

if __name__ == "__main__":
    demo.launch(share=False)
