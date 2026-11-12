import re
import unicodedata
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
DATA_FILE = ROOT / "data/processed/cleaned_dataset.csv"
MODEL_FILE = ROOT / "models/best_sentiment_model.joblib"
LABELS = ["positive", "negative", "neutral"]
ASPECTS = ["plot", "acting", "director", "music", "visual"]
NAMES = {"positive": "Tích cực", "negative": "Tiêu cực", "neutral": "Trung lập"}


def clean_text(text):
    # Giống hàm clean_text trong notebook 02
    text = unicodedata.normalize("NFC", str(text)).lower()
    return re.sub(r"\s+", " ", text).strip()


@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


st.set_page_config(page_title="Phân tích cảm xúc phim", layout="wide")
st.title("🎬 Phân tích cảm xúc bình luận phim tiếng Việt")

if not DATA_FILE.exists():
    st.error("Thiếu file data/processed/cleaned_dataset.csv")
    st.stop()

df = load_data()
tab1, tab2 = st.tabs(["📊 Phân tích dữ liệu", "🤖 Dự đoán cảm xúc"])

with tab1:
    sources = sorted(df["source"].dropna().unique().tolist())
    source = st.selectbox("Lọc theo nguồn bình luận", ["Tất cả"] + sources)
    filtered = df if source == "Tất cả" else df[df["source"] == source]
    labeled = filtered[filtered["overall"].isin(LABELS)]

    col1, col2, col3 = st.columns(3)
    col1.metric("Tổng bình luận", f"{len(filtered):,}")
    col2.metric("Có nhãn Overall", f"{len(labeled):,}")
    col3.metric("Nguồn dữ liệu", filtered["source"].nunique())

    st.subheader("Phân bố cảm xúc Overall")
    st.bar_chart(labeled["overall"].value_counts().reindex(LABELS, fill_value=0))
    st.caption("Không tính overall = none; các nhãn là nhãn gốc của dataset.")

    st.subheader("Tỷ lệ cảm xúc theo nguồn (%)")
    source_counts = pd.crosstab(labeled["source"], labeled["overall"])
    source_counts = source_counts.reindex(columns=LABELS, fill_value=0)
    source_rates = source_counts.div(source_counts.sum(axis=1), axis=0) * 100
    st.bar_chart(source_rates)

    st.subheader("Tỷ lệ cảm xúc theo khía cạnh phim (%)")
    aspect_counts = pd.DataFrame({
        name: filtered[name].value_counts().reindex(LABELS, fill_value=0)
        for name in ASPECTS
    }).T
    aspect_rates = aspect_counts.div(
        aspect_counts.sum(axis=1).replace(0, float("nan")), axis=0
    ).fillna(0) * 100
    st.bar_chart(aspect_rates)
    st.caption("Tính tỷ lệ trên các bình luận có nhãn cho từng khía cạnh; không tính none.")

with tab2:
    st.subheader("Thử dự đoán một bình luận mới")
    comment = st.text_area("Nội dung bình luận", placeholder="Phim này rất hay, diễn xuất tuyệt vời!")

    if not MODEL_FILE.exists():
        st.warning("Chưa có mô hình. Hãy chép best_sentiment_model.joblib từ file 04 vào thư mục models/.")
    elif st.button("Dự đoán", type="primary"):
        if not comment.strip():
            st.warning("Bạn hãy nhập một bình luận.")
        else:
            model = load_model()
            prediction = model.predict([clean_text(comment)])[0]
            st.success("Cảm xúc dự đoán: " + NAMES.get(prediction, prediction))
            st.caption("Đây là dự đoán của mô hình, không phải nhãn chắc chắn đúng.")

st.caption("Đề tài: Phân tích và dự đoán cảm xúc bình luận phim tiếng Việt")
