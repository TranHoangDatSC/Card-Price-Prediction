# 🚗 Car Price Prediction

Dự án này xây dựng một mô hình **Machine Learning** để dự đoán giá xe hơi dựa trên các thông số kỹ thuật (engine size, horsepower, curb weight, mpg, v.v.).

## 📂 Dataset
- File dữ liệu: `CarPrice.csv`  
- Gồm các đặc trưng (features) như:
  - enginesize (kích cỡ động cơ), horsepower (mã lực), curbweight (khối lượng không tải), citympg (mức tiêu thụ trong city, highwaympg (mức tiêu thụ trên cao tốc), ...
  - Biến mục tiêu (target): price (giá cả)

## ⚙️ Công nghệ sử dụng
- **Python** 🐍
- **Pandas, NumPy** → xử lý dữ liệu
- **Matplotlib, Seaborn** → trực quan hóa dữ liệu
- **Scikit-learn** → xây dựng và đánh giá mô hình ML

## 📊 Trực quan hóa dữ liệu
- **Histogram** phân phối giá xe
- **Heatmap** mức độ tương quan giữa các đặc trưng

## 🧠 Machine Learning Model
- Mô hình sử dụng: **Decision Tree Regressor**
- Train/Test split: 80/20

### Đánh giá mô hình
- **R² Score**: đo độ chính xác
- **MAE (Mean Absolute Error)**: đo sai số trung bình tuyệt đối

