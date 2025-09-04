import numpy as np                # Thư viện tính toán số học (ma trận, mảng…)
import pandas as pd               # Thư viện xử lý dữ liệu dạng bảng (giống Excel)
import matplotlib.pyplot as plt   # Thư viện vẽ biểu đồ cơ bản
import seaborn as sns             # Thư viện vẽ biểu đồ nâng cao, đẹp hơn matplotlib

from sklearn.model_selection import train_test_split  # Dùng để chia dữ liệu thành train/test
from sklearn.tree import DecisionTreeClassifier       # Dùng để phân loại
from pathlib import Path                              # Dùng để xử lý đường dẫn

# Xác định thư mục chứa file dữ liệu
BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR / "CarPrice.csv"

# Đọc dữ liệu từ file CSV
data = pd.read_csv(data_path)

# Thử nghiệm 5 dòng đầu tiên
data.head()

# Vẽ biểu đồ phân phối giá xe 
sns.set_style("whitegrid")
plt.figure(figsize=(15,10))
sns.histplot(data["price"], kde=True, bins=30)  # kde=True để có đường cong
plt.show()

# Biểu đồ heatmap thể hiện mức độ tương quan giữa các cột số
plt.figure(figsize=(20, 15))
correlations = data.select_dtypes(include=[np.number]).corr() # Lấy ra các cột số rồi tính mức tương quan
sns.heatmap(correlations, cmap="coolwarm", annot=True)        # annot=True để hiển thị số
plt.show()

# ---------------------------
# CHUẨN BỊ DỮ LIỆU CHO MACHINE LEARNING
# ---------------------------

# Chỉ chọn các cột số có ý nghĩa cho mô hình dự đoán
predict = "price"
data = data[["symboling", "wheelbase", "carlength", 
             "carwidth", "carheight", "curbweight", 
             "enginesize", "boreratio", "stroke", 
             "compressionratio", "horsepower", "peakrpm", 
             "citympg", "highwaympg", "price"]]

# Tách dữ liệu thành input (X) và output (y)
# X: các đặc điểm của xe (engine size, horsepower, ...)
# y: giá xe (price)
x = np.array(data.drop([predict], axis=1)) # Lấy tất cả trừ price
y = np.array(data[predict])                # Chỉ lấy price    

# Chia dữ liệu thành tập huấn luyện (train) và tập kiểm tra (test)
# test_size=0.2 nghĩa là 20% dữ liệu dành cho test, 80% dành cho train
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)


# ---------------------------
# XÂY DỰNG VÀ ĐÁNH GIÁ MÔ HÌNH
# ---------------------------
from sklearn.tree import DecisionTreeRegressor # Dùng cây quyết định để dự đoán giá xe
model = DecisionTreeRegressor()     # Tạo mô hình
model.fit(xtrain, ytrain)           # Huấn luyện mô hình với dữ liệu train
predictions = model.predict(xtest)  # Dự đoán giá xe với dữ liệu test


# ---------------------------
# ĐÁNH GIÁ MÔ HÌNH
# ---------------------------
from sklearn.metrics import mean_absolute_error

# Điểm số chính xác của mô hình (R^2 score). Gần 1 là tốt.
print("Score:", model.score(xtest, ytest))

# Sai số trung bình tuyệt đối (Mean Absolute Error).
# Giá trị càng nhỏ thì mô hình dự đoán càng gần với thực tế.
mae = mean_absolute_error(ytest, predictions)
print("Mean Absolute Error:", mae)