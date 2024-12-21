import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load pipeline
with open('models/pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Load mô hình
with open('models/id3_model.pkl', 'rb') as f:
    id3_model = pickle.load(f)

with open('models/knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)

with open('models/rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# Tiền xử lý dữ liệu đầu vào
def preprocess_input(data):
    df = pd.DataFrame([data])
    df['GENDER'] = df['GENDER'].map({'Nam': 0, 'Nữ': 1})
    df['target'] = 0  # Giá trị mặc định, không dùng trong dự đoán

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
st.sidebar.header("Nhập những dấu hiệu của bạn")

# Initialize session state for form inputs if they don't exist
if 'gender' not in st.session_state:
    st.session_state.gender = 'Nam'
if 'age' not in st.session_state:
    st.session_state.age = 30
if 'smoking' not in st.session_state:
    st.session_state.smoking = 'Không'
if 'yellow_fingers' not in st.session_state:
    st.session_state.yellow_fingers = 'Không'
if 'anxiety' not in st.session_state:
    st.session_state.anxiety = 'Không'
if 'peer_pressure' not in st.session_state:
    st.session_state.peer_pressure = 'Không'
if 'chronic_disease' not in st.session_state:
    st.session_state.chronic_disease = 'Không'
if 'fatigue' not in st.session_state:
    st.session_state.fatigue = 'Không'
if 'allergy' not in st.session_state:
    st.session_state.allergy = 'Không'
if 'wheezing' not in st.session_state:
    st.session_state.wheezing = 'Không'
if 'alcohol_consuming' not in st.session_state:
    st.session_state.alcohol_consuming = 'Không'
if 'coughing' not in st.session_state:
    st.session_state.coughing = 'Không'
if 'shortness_of_breath' not in st.session_state:
    st.session_state.shortness_of_breath = 'Không'
if 'swallowing_difficulty' not in st.session_state:
    st.session_state.swallowing_difficulty = 'Không'
if 'chest_pain' not in st.session_state:
    st.session_state.chest_pain = 'Không'
if 'model_name' not in st.session_state:
    st.session_state.model_name = 'ID3'

gender = st.sidebar.selectbox('Giới tính', ['Nam', 'Nữ'], key='gender')
age = st.sidebar.number_input('Tuổi', min_value=0, max_value=100, value=st.session_state.age, key='age')
smoking = st.sidebar.selectbox('Hút thuốc', ['Không', 'Có'], key = 'smoking')
yellow_fingers = st.sidebar.selectbox('Vàng ngón tay', ['Không', 'Có'], key='yellow_fingers')
anxiety = st.sidebar.selectbox('Lo lắng', ['Không', 'Có'], key='anxiety')
peer_pressure = st.sidebar.selectbox('Áp lực từ bạn bè', ['Không', 'Có'], key='peer_pressure')
chronic_disease = st.sidebar.selectbox('Bệnh mãn tính', ['Không', 'Có'], key='chronic_disease')
fatigue = st.sidebar.selectbox('Mệt mỏi', ['Không', 'Có'], key='fatigue')
allergy = st.sidebar.selectbox('Dị ứng', ['Không', 'Có'], key='allergy')
wheezing = st.sidebar.selectbox('Thở khò khè', ['Không', 'Có'], key='wheezing')
alcohol_consuming = st.sidebar.selectbox('Uống rượu bia', ['Không', 'Có'], key='alcohol_consuming')
coughing = st.sidebar.selectbox('Ho', ['Không', 'Có'], key='coughing')
shortness_of_breath = st.sidebar.selectbox('Khó thở', ['Không', 'Có'], key='shortness_of_breath')
swallowing_difficulty = st.sidebar.selectbox('Khó nuốt', ['Không', 'Có'], key='swallowing_difficulty')
chest_pain = st.sidebar.selectbox('Đau ngực', ['Không', 'Có'], key='chest_pain')

model_name = st.sidebar.selectbox('Chọn mô hình', ['ID3', 'KNN', 'Random Forest'], key='model_name')

# Nút dự đoán
if st.sidebar.button('Dự đoán'):
    # Check if all fields are filled
    if not gender or not age or not smoking or not yellow_fingers or not anxiety or not peer_pressure or not chronic_disease or not fatigue or not allergy or not wheezing or not alcohol_consuming or not coughing or not shortness_of_breath or not swallowing_difficulty or not chest_pain:
        st.sidebar.warning("Vui lòng điền đầy đủ thông tin.")
    else:
        data = {
            'GENDER': gender,
            'AGE': age,
            'SMOKING': 1 if smoking == 'Có' else 0,
            'YELLOW_FINGERS': 1 if yellow_fingers == 'Có' else 0,
            'ANXIETY': 1 if anxiety == 'Có' else 0,
            'PEER_PRESSURE': 1 if peer_pressure == 'Có' else 0,
            'CHRONIC DISEASE': 1 if chronic_disease == 'Có' else 0,
            'FATIGUE ': 1 if fatigue == 'Có' else 0,
            'ALLERGY ': 1 if allergy == 'Có' else 0,
            'WHEEZING': 1 if wheezing == 'Có' else 0,
            'ALCOHOL CONSUMING': 1 if alcohol_consuming == 'Có' else 0,
            'COUGHING': 1 if coughing == 'Có' else 0,
            'SHORTNESS OF BREATH': 1 if shortness_of_breath == 'Có' else 0,
            'SWALLOWING DIFFICULTY': 1 if swallowing_difficulty == 'Có' else 0,
            'CHEST PAIN': 1 if chest_pain == 'Có' else 0,
            'target': 0
        }
        
        processed_data = preprocess_input(data)

        if model_name == 'ID3':
            prediction = id3_model.predict(processed_data)
            try:
                proba = id3_model.predict_proba(processed_data)
                probability = round(proba[0][prediction[0]] * 100, 2)
                st.markdown(f"<h4 style='text-align: center;'>Xác suất: {probability}%</h4>", unsafe_allow_html=True)
            except AttributeError:
                st.markdown("<h4 style='text-align: center;'>Mô hình này không có xác suất</h4>", unsafe_allow_html=True)
        elif model_name == 'KNN':
            prediction = knn_model.predict(processed_data)
            try:
                proba = knn_model.predict_proba(processed_data)
                probability = round(proba[0][prediction[0]] * 100, 2)
                st.markdown(f"<h4 style='text-align: center;'>Xác suất: {probability}%</h4>", unsafe_allow_html=True)
            except AttributeError:
                st.markdown("<h4 style='text-align: center;'>Mô hình này không có xác suất</h4>", unsafe_allow_html=True)
        else:
            prediction = rf_model.predict(processed_data)
            try:
                proba = rf_model.predict_proba(processed_data)
                probability = round(proba[0][prediction[0]] * 100, 2)
                st.markdown(f"<h4 style='text-align: center;'>Xác suất: {probability}%</h4>", unsafe_allow_html=True)
            except AttributeError:
                st.markdown("<h4 style='text-align: center;'>Mô hình này không có xác suất</h4>", unsafe_allow_html=True)

        # Hiển thị kết quả
        st.markdown("<h2 style='text-align: center; color: green;'>Kết quả dự đoán</h2>", unsafe_allow_html=True)
        if prediction[0] == 1:
            st.markdown(f"<h3 style='text-align: center;'>Mô hình <span style='color: red;'>{model_name}</span> dự đoán nguy cơ ung thư phổi <span style='color: red;'>cao</span>.</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='text-align: center;'>Mô hình <span style='color: green;'>{model_name}</span> dự đoán nguy cơ ung thư phổi <span style='color: green;'>thấp</span>.</h3>", unsafe_allow_html=True)

