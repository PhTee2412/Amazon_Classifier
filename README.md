# Amazon Product Classifier

**Dự đoán danh mục sản phẩm Amazon bằng NLP & Học Máy – từ dữ liệu chưa có nhãn**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vQbgzVixv8uaIGvLJACGdlxaM2jeR4vL?usp=sharing)

---

##  Giới thiệu

Hệ thống học máy pipeline hoàn chỉnh, giải quyết bài toán **gán nhãn tự động và phân loại sản phẩm** khi dữ liệu gốc **không có cột danh mục (category)**. Toàn bộ quá trình từ xử lý thô, tạo nhãn thông minh, trích xuất đặc trưng đa chiều cho đến huấn luyện & triển khai đều được xây dựng từ đầu.

- **Bài toán:** Phân loại đa lớp (Multi-class Classification)
- **Nguồn dữ liệu:** [Amazon Products Sales Dataset 42k Items (2025)](https://www.kaggle.com/datasets/ikramshah512/amazon-products-sales-dataset-42k-items-2025?select=amazon_products_sales_data_uncleaned.csv) từ Kaggle.  
   **Đặc điểm cốt lõi:** Dữ liệu chỉ bao gồm tên sản phẩm, giá, đánh giá,... **hoàn toàn không có nhãn danh mục**. Điều này biến bài toán trở thành tự động tạo ra “ground truth” trước khi có thể huấn luyện mô hình phân loại.
-  **Demo trực tiếp:** [https://huggingface.co/spaces/PhTee/amazon-classifier](https://huggingface.co/spaces/PhTee/amazon-classifier)
- ** 10 danh mục mục tiêu do tôi định nghĩa** (dựa trên phân tích thị trường và đặc trưng sản phẩm):
  1. `Laptops`
  2. `Phones`
  3. `Audio`
  4. `Cameras`
  5. `TV & Display`
  6. `Computers & Accessories`
  7. `Other Electronics`
  8. `Wearables`
  9. `Gaming`
  10. `Home & Kitchen`

- ** Phân tích chi tiết (code, biểu đồ, bản đồ phân cụm, ma trận nhầm lẫn, feature importance,...):** [Google Colab Notebook](https://colab.research.google.com/drive/1vQbgzVixv8uaIGvLJACGdlxaM2jeR4vL?usp=sharing)

> **Điểm đặc biệt của bài toán**
> - **Không có nhãn mục tiêu** – toàn bộ quá trình gán nhãn được thiết kế thủ công bằng luật và embedding AI, sau đó lan truyền nhãn bằng học máy.
> - **Mất cân bằng lớp nghiêm trọng** – đồ điện tử áp đảo, các lớp như `Wearables`, `Home & Kitchen` vô cùng ít mẫu. Hệ thống dùng SMOTE, class weighting và cơ chế cảnh báo độ tin cậy để đảm bảo công bằng và an toàn cho người dùng.

---

##  Pipeline tổng thể

| Giai đoạn | Nội dung chính | Kỹ thuật / Công nghệ |
|---|---|---|
| **1. Gán nhãn tự động** | Tạo nhãn từ dữ liệu không có target | Keyword Matching, Sentence-BERT (`all-MiniLM-L6-v2`) + Cosine Similarity, Ensemble, Label Propagation (Logistic Regression) |
| **2. Làm sạch & lọc bẫy** | Loại bỏ sản phẩm “gây nhiễu” | `trap_word_filter` (bảo hành, ốp lưng, sticker…), `price_sanity_check` |
| **3. Trích xuất đặc trưng** | Số hóa mọi tín hiệu từ sản phẩm | TF‑IDF, TruncatedSVD (LSA), PCA, Social Proof, Brand Tier, Word Count, Discount %, Log Transform |
| **4. Tiền xử lý & chọn lọc** | Giảm chiều, loại bỏ đa cộng tuyến | `StandardScaler`, VIF, `np.hstack` |
| **5. Huấn luyện & tối ưu** | Xây dựng mô hình phân loại cuối cùng | LightGBM, Optuna (TPE, SQLite), `StratifiedKFold`, SMOTE, `class_weight='balanced'`, Early Stopping |
| **6. Triển khai** | Ứng dụng thực tế | Gradio, `ClassificationModel`, ngưỡng tin cậy linh hoạt |

---

##  Công nghệ sử dụng – Chi tiết

###  Xử lý ngôn ngữ & Gán nhãn
- **Sentence-BERT** (`all-MiniLM-L6-v2`): embedding 384 chiều cho tên sản phẩm, so khớp với tập nhãn ứng viên qua **Cosine Similarity**.
- **Keyword Matching (Expert Rules):** bộ từ khóa chuyên biệt cho 10 ngành hàng, hỗ trợ regex linh hoạt (`flexible_kw_match`) bắt cả số nhiều, viết liền hay cách.
- **Ensemble Keyword + AI:** kết hợp có trọng số (alpha/beta) giữa độ tin cậy từ khóa và similarity score.
- **Label Propagation:** dùng Logistic Regression huấn luyện trên các mẫu tự tin (≥60%), lan truyền nhãn cho phần còn lại với ngưỡng xác suất ≥45%.

###  Làm sạch & Logic nghiệp vụ
- **`trap_word_filter`:** phát hiện các cụm như *protection plan, warranty, case for, guide, sticker* → đưa về nhóm `Other Electronics` hoặc `Other Category`.
- **`price_sanity_check`:** dùng logic giá để sửa nhãn (VD: Laptop giá dưới $80 → không thể là laptop).

###  Trích xuất đặc trưng (Feature Engineering)
- **TF‑IDF** (5000 features, 1‑2 grams) → **TruncatedSVD (256 chiều)**: nắm bắt ngữ nghĩa ẩn từ văn bản.
- **PCA** trên embedding (giữ 90% variance): giảm chiều, chống overfitting.
- **Social Proof:** `log(rating × number_of_reviews)` – đại diện cho độ uy tín.
- **Brand Tier:** mã hóa thương hiệu 3 mức (0‑thường, 1‑top brand, 2‑premium) dựa trên danh sách thương hiệu được định nghĩa sẵn.
- **Word Count, Discount %, Log Transform:** các đặc trưng thống kê giúp mô hình nhận diện kiểu sản phẩm.

### Cân bằng dữ liệu & Chọn biến
- **SMOTE** (Synthetic Minority Over-sampling): sinh mẫu tổng hợp trong mỗi fold khi cross‑validation.
- **Class Weight = 'balanced'** trong LightGBM.
- **VIF (Variance Inflation Factor):** loại bỏ tuần tự biến có VIF > 5 để giảm đa cộng tuyến, giữ lại tập đặc trưng ổn định.

###  Mô hình & Tối ưu
- **LightGBM Classifier** (Gradient Boosting): mô hình chính, hỗ trợ GPU, huấn luyện trên dữ liệu lớn.
- **Optuna** (TPE Sampler, SQLite storage): tối ưu `learning_rate`, `num_leaves`, `max_depth`, `reg_alpha`, `reg_lambda`, `subsample`, `colsample_bytree` với 50 trials trong 4 giờ.
- **Stratified K‑Fold (k=5) + Early Stopping:** đảm bảo đánh giá khách quan và dừng đúng lúc.

### Triển khai
- **Gradio:** giao diện web trực quan.
- **Cảnh báo độ tin cậy:** tự động hiển thị cảnh báo nếu xác suất cao nhất < ngưỡng (50%) để cảnh báo người dùng khỏi các dự đoán kém chắc chắn.

---

## Kết quả huấn luyện

| Chỉ số | Giá trị |
|---|---|
| **Overall Accuracy** | ~96% |
| **Balanced Accuracy** | ~0.94 |
| **Macro F1‑Score** | ~0.956 |

> **Lưu ý:**  
> - Accuracy cao một phần do dữ liệu mất cân bằng (một số lớp chiếm ưu thế).  
> - Balanced Accuracy và Macro F1 cho thấy mô hình vẫn hoạt động khá tốt trên các lớp nhỏ, nhưng chưa thực sự ổn định khi gặp sản phẩm mới.  
> - Trong thực tế, một số danh mục ít mẫu (ví dụ: Home & Kitchen, Wearables) vẫn dễ bị dự đoán sai nếu thiếu từ khóa đặc trưng. Cơ chế cảnh báo khi độ tin cậy thấp có thể phần nào giúp người dùng nhận biết được một số trường hợp này.

---

## Hạn chế đã biết (Known Limitations)

Mặc dù mô hình đạt độ chính xác tổng thể ~96%, trong thực tế triển khai vẫn tồn tại nhiều trường hợp dự đoán chưa chính xác, đặc biệt với các danh mục có số lượng mẫu ít. Nguyên nhân chủ yếu đến từ **sự hạn chế của tập từ khóa** và **ảnh hưởng từ các đặc trưng số**:

- **Từ khóa chưa được làm giàu đầy đủ**: Hệ thống gán nhãn và trích xuất đặc trưng hiện dựa trên một bộ từ khóa thủ công (`category_keywords`) còn khá khiêm tốn. Các sản phẩm có tên không chứa từ khóa đặc trưng (ví dụ: “Kindle”, “Instant Pot”, v.v) dễ bị mô hình liên tưởng sang nhóm khác chiếm ưu thế.
- **Ảnh hưởng từ các đặc trưng số**: Do dữ liệu huấn luyện mất cân bằng (đa số là đồ điện tử), các biến như `price`, `rating`, `reviews` có thể “lấn át” tín hiệu ngữ nghĩa yếu từ tên sản phẩm. Điều này khiến mô hình dễ dự đoán về các lớp phổ biến (ví dụ: Computers & Accessories) thay vì đúng lớp hiếm (Home & Kitchen, Wearables).
- **Dữ liệu huấn luyện gốc hoàn toàn không có nhãn**: Việc phải tự động gán nhãn bằng kết hợp luật và AI tiềm ẩn sai số lan truyền, nhất là ở những vùng dữ liệu sản phẩm không rõ ràng.

Để cải thiện, hướng phát triển tiếp theo có thể tập trung vào:
- Bổ sung thêm mẫu cho các lớp thiểu số và làm giàu từ khóa đặc trưng.
- Fine‑tune Sentence‑BERT trên chính tập dữ liệu Amazon.
- Xây dựng thêm các đặc trưng ngữ nghĩa (ví dụ: cờ `is_kitchen_appliance`, `is_ereader`) để giảm phụ thuộc vào giá và đánh giá.

---
  
##  Hướng dẫn cài đặt & Chạy Demo

1. **Yêu cầu:** Python 3.9+
2. **Clone repository:**
   ```bash
   git clone https://github.com/PhTee2412/Amazon_Classifier

   cd Amazon_Classifier

---

- **Email**: phtee2412@gmail.com
