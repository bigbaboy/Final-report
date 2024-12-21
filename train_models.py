import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

# Tạo thư mục models nếu chưa tồn tại
if not os.path.exists('models'):
    os.makedirs('models')

# Đọc dữ liệu
df = pd.read_csv("D://Download//survey lung cancer.csv")  # Thay đổi đường dẫn

# Đổi tên cột target
df = df.rename(columns={'LUNG_CANCER': 'target'})

# Xử lý GENDER (Label Encoding)
df['GENDER'] = df['GENDER'].map({'M': 0, 'F': 1})

# Xử lý target (Label Encoding)
label_encoder = LabelEncoder()
df['target'] = label_encoder.fit_transform(df['target'])

# Rời rạc hóa AGE (tùy chọn)
bins = [0, 30, 40, 50, 60, 70, 100]
labels = ['<30', '30-40', '40-50', '50-60', '60-70', '>70']
df['AGE_GROUP'] = pd.cut(df['AGE'], bins=bins, labels=labels, right=False)
df.drop('AGE', axis=1, inplace=True)

# Tách features và target
X = df.drop('target', axis=1)
y = df['target']

# One-hot Encoding và MinMaxScaler
ct = ColumnTransformer(
    transformers=[
        ('ohe', OneHotEncoder(handle_unknown='ignore'), ['GENDER', 'AGE_GROUP']),
        ('scaler', MinMaxScaler(), ['SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
                                       'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ',
                                       'ALLERGY ', 'WHEEZING', 'ALCOHOL CONSUMING',
                                       'COUGHING', 'SHORTNESS OF BREATH',
                                       'SWALLOWING DIFFICULTY', 'CHEST PAIN'])
    ],
    remainder='passthrough'
)

# Chia dữ liệu thành tập train và test (có stratify)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Tạo pipeline
pipeline = Pipeline([('transformer', ct)])
X_train_transformed = pipeline.fit_transform(X_train)
X_test_transformed = pipeline.transform(X_test)

# ID3 (sử dụng DecisionTreeClassifier với criterion='entropy')
id3_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
id3_model.fit(X_train_transformed, y_train)
y_pred_id3 = id3_model.predict(X_test_transformed)
print("ID3 Accuracy:", accuracy_score(y_test, y_pred_id3))

# KNN - Chỉ sử dụng các đặc trưng triệu chứng
# Lấy các cột từ cột thứ 8 (SMOKING)
X_train_knn = X_train_transformed[:, 8:]
X_test_knn = X_test_transformed[:, 8:]
knn_model = KNeighborsClassifier(n_neighbors=7)
knn_model.fit(X_train_knn, y_train)
y_pred_knn = knn_model.predict(X_test_knn)
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_transformed, y_train)
y_pred_rf = rf_model.predict(X_test_transformed)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# In ra 5 dòng đầu tiên của X_train_transformed để kiểm tra
print("X_train_transformed (5 dòng đầu):")
print(pd.DataFrame(X_train_transformed).head())

# In ra 5 dòng đầu tiên của X_test_transformed để kiểm tra
print("X_test_transformed (5 dòng đầu):")
print(pd.DataFrame(X_test_transformed).head())

# In ra 5 dòng đầu tiên của X_train_knn để kiểm tra
print("X_train_knn (5 dòng đầu):")
print(pd.DataFrame(X_train_knn).head())

# In ra 5 dòng đầu tiên của X_test_knn để kiểm tra
print("X_test_knn (5 dòng đầu):")
print(pd.DataFrame(X_test_knn).head())

# Lưu mô hình
with open('models/id3_model.pkl', 'wb') as f:
    pickle.dump(id3_model, f)

with open('models/knn_model.pkl', 'wb') as f:
    pickle.dump(knn_model, f)

with open('models/rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# Lưu pipeline
with open('models/pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
