import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load pipeline và mô hình
try:
    with open('models/pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    with open('models/id3_model.pkl', 'rb') as f:
        id3_model = pickle.load(f)
    with open('models/knn_model.pkl', 'rb') as f:
        knn_model = pickle.load(f)
    with open('models/rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
except FileNotFoundError:
    st.error("Không tìm thấy tệp mô hình. Vui lòng kiểm tra đường dẫn.")
    st.stop()  # Dừng ứng dụng nếu không tìm thấy mô hình

# Tiền xử lý dữ liệu đầu vào
def preprocess_input(data):
    df = pd.DataFrame([data])
    df['GENDER'] = df['GENDER'].map({'Nam': 0, 'Nữ': 1})
    df['target'] = 0

    bins = [0, 30, 40, 50, 60, 70, 100]
    labels = ['<30', '30-40', '40-50', '50-60', '60-70', '>70']
    df['AGE_GROUP'] = pd.cut(df['AGE'], bins=bins, labels=labels, right=False)
    df.drop('AGE', axis=1, inplace=True)

    processed_data = pipeline.transform(df.drop('target', axis=1))
    return processed_data

# Giao diện Streamlit
st.set_page_config(page_title="Dự đoán ung thư phổi", page_icon=":lungs:")

st.markdown("<h1 style='text-align: center; color: #1E90FF;'>ỨNG DỤNG DỰ ĐOÁN UNG THƯ PHỔI</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Nhập thông tin")

# Sử dụng form để nhóm các input và xử lý sự kiện submit
with st.sidebar.form(key='input_form'):
    gender = st.selectbox('Giới tính', ['Nam', 'Nữ'])
    age = st.number_input('Tuổi', min_value=0, max_value=100, value=30)
    smoking = st.selectbox('Hút thuốc', ['Không', 'Có'])
    yellow_fingers = st.selectbox('Vàng ngón tay', ['Không', 'Có'])
    anxiety = st.selectbox('Lo lắng', ['Không', 'Có'])
    peer_pressure = st.selectbox('Áp lực từ bạn bè', ['Không', 'Có'])
    chronic_disease = st.selectbox('Bệnh mãn tính', ['Không', 'Có'])
    fatigue = st.selectbox('Mệt mỏi', ['Không', 'Có'])
    allergy = st.selectbox('Dị ứng', ['Không', 'Có'])
    wheezing = st.selectbox('Thở khò khè', ['Không', 'Có'])
    alcohol_consuming = st.selectbox('Uống rượu bia', ['Không', 'Có'])
    coughing = st.selectbox('Ho', ['Không', 'Có'])
    shortness_of_breath = st.selectbox('Khó thở', ['Không', 'Có'])
    swallowing_difficulty = st.selectbox('Khó nuốt', ['Không', 'Có'])
    chest_pain = st.selectbox('Đau ngực', ['Không', 'Có'])
    model_name = st.selectbox('Chọn mô hình', ['ID3', 'KNN', 'Random Forest'])
    submit_button = st.form_submit_button(label='Dự đoán')

# Xử lý dự đoán khi form được submit
if submit_button:
    data = {
        'GENDER': gender,
        'AGE': age,
        'SMOKING': 2 if smoking == 'Có' else 1,  # Sửa thành 2 nếu 'Có', 1 nếu 'Không'
        'YELLOW_FINGERS': 2 if yellow_fingers == 'Có' else 1,  # Sửa thành 2 nếu 'Có', 1 nếu 'Không'
        'ANXIETY': 2 if anxiety == 'Có' else 1,  # Sửa thành 2 nếu 'Có', 1 nếu 'Không'
        'PEER_PRESSURE': 2 if peer_pressure == 'Có' else 1,  # Sửa thành 2 nếu 'Có', 1 nếu 'Không'
        'CHRONIC DISEASE': 2 if chronic_disease == 'Có' else 1,  # Sửa thành 2 nếu 'Có', 1 nếu 'Không'
        'FATIGUE ': 2 if fatigue == 'Có' else 1,  # Sửa thành 2 nếu 'Có', 1 nếu 'Không'
        'ALLERGY ': 2 if allergy == 'Có' else 1,  # Sửa thành 2 nếu 'Có', 1 nếu 'Không'
        'WHEEZING': 2 if wheezing == 'Có' else 1,  # Sửa thành 2 nếu 'Có', 1 nếu 'Không'
        'ALCOHOL CONSUMING': 2 if alcohol_consuming == 'Có' else 1,  # Sửa thành 2 nếu 'Có', 1 nếu 'Không'
        'COUGHING': 2 if coughing == 'Có' else 1,  # Sửa thành 2 nếu 'Có', 1 nếu 'Không'
        'SHORTNESS OF BREATH': 2 if shortness_of_breath == 'Có' else 1,  # Sửa thành 2 nếu 'Có', 1 nếu 'Không'
        'SWALLOWING DIFFICULTY': 2 if swallowing_difficulty == 'Có' else 1,  # Sửa thành 2 nếu 'Có', 1 nếu 'Không'
        'CHEST PAIN': 2 if chest_pain == 'Có' else 1,  # Sửa thành 2 nếu 'Có', 1 nếu 'Không'
        'target': 0
    }

    processed_data = preprocess_input(data)

    if model_name == 'ID3':
        model = id3_model
        prediction = model.predict(processed_data)
        try:
            proba = model.predict_proba(processed_data)
            probability = round(proba[0][prediction[0]] * 100, 2)
            st.markdown(f"<h4 style='text-align: center;'>Xác suất: {probability}%</h4>", unsafe_allow_html=True)
        except AttributeError:
            st.markdown("<h4 style='text-align: center;'>Mô hình này không có xác suất</h4>", unsafe_allow_html=True)
    elif model_name == 'KNN':
        model = knn_model
        prediction = model.predict(processed_data[:, 8:]) # Chỉ sử dụng các đặc trưng triệu chứng
        try:
            proba = model.predict_proba(processed_data[:, 8:])
            probability = round(proba[0][prediction[0]] * 100, 2)
            st.markdown(f"<h4 style='text-align: center;'>Xác suất: {probability}%</h4>", unsafe_allow_html=True)
        except AttributeError:
            st.markdown("<h4 style='text-align: center;'>Mô hình này không có xác suất</h4>", unsafe_allow_html=True)
    else:
        model = rf_model
        prediction = model.predict(processed_data)
        try:
            proba = model.predict_proba(processed_data)
            probability = round(proba[0][prediction[0]] * 100, 2)
            st.markdown(f"<h4 style='text-align: center;'>Xác suất: {probability}%</h4>", unsafe_allow_html=True)
        except AttributeError:
            st.markdown("<h4 style='text-align: center;'>Mô hình này không có xác suất</h4>", unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: center; color: green;'>Kết quả dự đoán</h2>", unsafe_allow_html=True)
    if prediction[0] == 1:
        st.markdown(f"<h3 style='text-align: center;'>Mô hình <span style='color: red;'>{model_name}</span> dự đoán nguy cơ ung thư phổi <span style='color: red;'>cao</span>.</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='text-align: center;'>Mô hình <span style='color: green;'>{model_name}</span> dự đoán nguy cơ ung thư phổi <span style='color: green;'>thấp</span>.</h3>", unsafe_allow_html=True)

