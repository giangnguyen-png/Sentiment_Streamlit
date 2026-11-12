# Streamlit – Phân tích cảm xúc bình luận phim

## Có những gì?
- **Phân tích dữ liệu:** Biểu đồ nhãn Overall, nguồn bình luận, 5 khía cạnh phim.
- **Dự đoán cảm xúc:** Nhập bình luận mới, dùng Pipeline TF-IDF + mô hình đã huấn luyện trong file `04_ML_Training.ipynb`.

## Bước chuẩn bị (một lần)

1. Giải nén thư mục `Sentiment_Streamlit` trên máy tính Windows.
2. Dữ liệu `data/processed/cleaned_dataset.csv` đã được tạo từ `final_dataset.csv` bạn gửi, theo cách làm sạch ở notebook 02.
3. **Bắt buộc** tải file `best_sentiment_model.joblib` trong Google Drive `SentimentAnalysis/models/` về và đặt vào thư mục `Sentiment_Streamlit/models/`. ZIP này **không chứa** file mô hình vì file đó chưa được gửi vào cuộc trò chuyện.
4. Mở Terminal/PowerShell trong thư mục `Sentiment_Streamlit`, chạy:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit sẽ mở trang web chạy trên máy tính. Nếu không tự mở, dùng URL `http://localhost:8501`.

## Lưu ý

- Không chạy `app.py` bằng nút Run Python thông thường; cần dùng lệnh `streamlit run`.
- Nếu mô hình báo lỗi không tương thích phiên bản scikit-learn, hãy kiểm tra phiên bản scikit-learn tại Google Colab bằng `import sklearn; print(sklearn.__version__)`, rồi pin đúng phiên bản đó trong `requirements.txt` (ví dụ `scikit-learn==X.Y.Z`).
- `clean_text()` của ứng dụng giữ đúng 3 bước từ notebook 02: Unicode NFC, chữ thường, chuẩn hóa khoảng trắng. Dataset có nhiều từ đã nối bằng `_`, nên khi nhập bình luận tự do, chất lượng dự đoán có thể khác Test. Ứng dụng chưa bổ sung tokenizer tiếng Việt để giữ code đơn giản.
- Các biểu đồ dùng nhãn gốc của dataset, không phải dự đoán ML.
- `joblib.load` chỉ dùng với mô hình do chính bạn tạo / nguồn đáng tin cậy.
- Khi triển khai trên internet, bạn cần đưa các file cần thiết vào repo và tuân thủ quyền sử dụng/chia sẻ dataset.
