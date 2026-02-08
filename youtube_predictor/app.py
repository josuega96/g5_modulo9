import streamlit as st
import pandas as pd
import numpy as np
import joblib

# CONFIGURACIÓN DE LA INTERFAZ
st.set_page_config(page_title="YouTube ML Predictor", page_icon="📺")

# Cargar los artefactos del modelo
@st.cache_resource
def load_model_assets():
    model = joblib.load('best_youtube_model.pkl')
    scaler_X = joblib.load('scaler_X.pkl')
    scaler_y = joblib.load('scaler_y.pkl')
    features_list = joblib.load('features_order.pkl')
    return model, scaler_X, scaler_y, features_list

model, scaler_X, scaler_y, features_list = load_model_assets()

st.title("📺 YouTube Views Predictor API")
st.markdown("Estima el éxito de tu video usando inteligencia artificial.")

# FORMULARIO DE ENTRADA
with st.form("prediction_form"):
    st.subheader("Datos del Video")
    
    # Inputs numéricos
    subscribers = st.number_input("Suscriptores del canal", min_value=0, value=100000)
    likes = st.number_input("Likes esperados", min_value=0, value=5000)
    comments = st.number_input("Comentarios esperados", min_value=0, value=500)
    duration = st.number_input("Duración (segundos)", min_value=1, value=600)
    
    # Selección de categoría (IDs basados en tu dataset)
    # 10=Music, 20=Gaming, 24=Entertainment, 28=Tech, etc.
    category_id = st.selectbox("Categoría (ID)", options=[10, 15, 17, 20, 22, 23, 24, 25, 26, 27, 28])
    
    submit_button = st.form_submit_button("🚀 Calcular Vistas")

if submit_button:
    # 1. Crear DataFrame vacío con las columnas del entrenamiento
    input_df = pd.DataFrame(0.0, index=[0], columns=features_list)
    
    # 2. Llenar variables numéricas
    input_df['subscriber_count'] = float(subscribers)
    input_df['likes'] = float(likes)
    input_df['comments'] = float(comments)
    input_df['duration_sec'] = float(duration)
    
    # 3. Aplicar One-Hot Encoding manual (ID.0 como sale en Sklearn)
    cat_col_name = f"{category_id}.0"
    if cat_col_name in features_list:
        input_df[cat_col_name] = 1.0

    # 4. Escalar los datos de entrada
    input_scaled = scaler_X.transform(input_df.values)
    
    # 5. Predicción
    prediction_scaled = model.predict(input_scaled)
    
    # 6. Revertir el escalado del Target (y)
    # El modelo devuelve el precio/vistas en el formato 'pequeño'
    prediction_real = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))
    
    # 7. Revertir la división por 1,000,000 (si lo hiciste en el paso 3)
    final_result = prediction_real[0][0] * 1000000
    
    st.success(f"### Resultado Estimado: {int(final_result):,} vistas")