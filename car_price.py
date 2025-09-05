import numpy as np                # Thư viện tính toán số học (ma trận, mảng…)
import pandas as pd               # Thư viện xử lý dữ liệu dạng bảng (giống Excel)
import matplotlib.pyplot as plt   # Thư viện vẽ biểu đồ cơ bản
import seaborn as sns             # Thư viện vẽ biểu đồ nâng cao, đẹp hơn matplotlib

from sklearn.model_selection import train_test_split  # Dùng để chia dữ liệu thành train/test
from sklearn.tree import DecisionTreeClassifier       # Dùng để phân loại
from pathlib import Path                              # Dùng để xử lý đường dẫn
from tqdm import tqdm                                 # Thư viện hiển thị thanh tiến trình

# Xác định thư mục chứa file dữ liệu
BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR / "CarPrice.csv"

# Đọc dữ liệu từ file CSV
data = pd.read_csv(data_path)
print("✅ Dữ liệu đã nạp thành công!")

# Thử nghiệm 5 dòng đầu tiên
print("🔎 Thử nghiệm 5 dòng đầu tiên:\n", data.head(), "\n")

# Kiểm tra các giá trị null
data.isnull().sum()

# Kiểm tra các giá trị của dữ liệu
print("📊 Kiểm tra thông tin dữ liệu...")
print(data.info(), "\n")
print("📊 Mô tả dữ liệu số:\n", data.describe(), "\n")

# Trích xuất giá trị riêng biệt của tên xe
print("🔎 Trích xuất giá trị riêng biệt (Tên xe)")
data.CarName.unique()

# Vẽ biểu đồ phân phối
print("📈 Vẽ phân phối giá xe...")
sns.set_style("whitegrid")
plt.figure(figsize=(10,5))
sns.histplot(data["price"], stat="density", bins=30, color="skyblue", edgecolor="black")
sns.kdeplot(data["price"], color="red", linewidth=2)
plt.show()

# Vẽ bản đồ nhiệt
print("📈 Vẽ heatmap tương quan...")
plt.figure(figsize=(10,8))
correlations = data.select_dtypes(include=[np.number]).corr()  # Chỉ lấy các cột số
sns.heatmap(correlations, cmap="coolwarm",annot=True)
plt.show()

################## 
# Training Model #
##################
print("\n🤖 Đang chuẩn bị train model...")

# Cột mục tiêu cần dự đoán
predict = "price"

# Các features + cột mục tiêu
data = data[["symboling", "wheelbase", "carlength", 
             "carwidth", "carheight", "curbweight", 
             "enginesize", "boreratio", "stroke", 
             "compressionratio", "horsepower", "peakrpm", 
             "citympg", "highwaympg", "price"]]

# Tạo tập dữ liệu X (features) và y (label)
# drop([predict], axis=1) => bỏ cột "price" khỏi X
x = np.array(data.drop([predict], axis=1))   # input features
y = np.array(data[predict])                  # target (price)

# Chia dữ liệu thành tập train và test (80% train, 20% test)
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)

# Khởi tạo và huấn luyện mô hình Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)

print("🚀 Đang train model (DecisionTree)...")
for _ in tqdm(range(1), desc="Training Progress"):
    model.fit(xtrain, ytrain)

predictions = model.predict(xtest)

# Đánh giá mô hình
from sklearn.metrics import mean_absolute_error, r2_score
print("\n📊 Đánh giá model:")
mae = mean_absolute_error(ytest, predictions)
r2 = r2_score(ytest, predictions)

print(f"✅ Độ chính xác R² score: {r2:.2f}")
print(f"✅ Sai số tuyệt đối trung bình (MAE): {mae:.2f}")
