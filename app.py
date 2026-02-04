import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(
    page_title="Prediksi Inflasi Indonesia",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_models():

    try:

        model_dir = "model_deployment"
        
        if not os.path.exists(model_dir):
            st.error(f"❌ Directory {model_dir} tidak ditemukan!")
            return None
        

        svr_model = joblib.load(os.path.join(model_dir, 'svr_model.pkl'))
        ann_model = joblib.load(os.path.join(model_dir, 'ann_model.pkl'))
        scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        
        # Load metadata
        with open(os.path.join(model_dir, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        # Load config
        with open(os.path.join(model_dir, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        
        return {
            'svr': svr_model,
            'ann': ann_model,
            'scaler': scaler,
            'metadata': metadata,
            'config': config
        }
    
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None

# ------------------------------------------------------------
# FUNGSI PREDIKSI
# ------------------------------------------------------------
def predict_inflation(models, year, month_num, province, subcategory):
    """Lakukan prediksi menggunakan kedua model"""
    
    # Get encodings
    province_encoded = models['metadata']['province_mapping_reverse'][province]
    subcategory_encoded = models['metadata']['subcategory_mapping_reverse'][subcategory]
    
    # Prepare input data
    input_data = pd.DataFrame({
        'year': [year],
        'month_num': [month_num],
        'province_encoded': [province_encoded],
        'subcategory_encoded': [subcategory_encoded],
        'is_annual': [0]  # False untuk prediksi bulanan
    })
    
    # Scale features
    input_scaled = models['scaler'].transform(input_data)
    
    # Predict dengan kedua model
    pred_svr = models['svr'].predict(input_scaled)[0]
    pred_ann = models['ann'].predict(input_scaled)[0]
    
    # Hitung ensemble (rata-rata)
    pred_ensemble = (pred_svr + pred_ann) / 2
    
    return {
        'svr': pred_svr,
        'ann': pred_ann,
        'ensemble': pred_ensemble,
        'input': input_data.iloc[0].to_dict()
    }

# ------------------------------------------------------------
# FUNGSI EVALUASI MODEL (opsional)
# ------------------------------------------------------------
def evaluate_models(models):
    """Evaluasi performa model menggunakan test data"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    try:
        # Load test data
        test_data = joblib.load(os.path.join('model_deployment', 'test_data.pkl'))
        X_test = test_data['X_test']
        y_test = test_data['y_test']
        
        # Scale
        X_test_scaled = models['scaler'].transform(X_test)
        
        # Predict
        y_pred_svr = models['svr'].predict(X_test_scaled)
        y_pred_ann = models['ann'].predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {}
        
        for model_name, y_pred in [('SVR', y_pred_svr), ('ANN', y_pred_ann)]:
            metrics[model_name] = {
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MAE': mean_absolute_error(y_test, y_pred),
                'R2': r2_score(y_test, y_pred)
            }
        
        return metrics
    
    except Exception as e:
        st.warning(f"⚠️ Tidak dapat mengevaluasi model: {str(e)}")
        return None

# ------------------------------------------------------------
# MAIN APP
# ------------------------------------------------------------
def main():
    # Load models
    with st.spinner("🔄 Loading models..."):
        models = load_models()
    
    if models is None:
        st.stop()
    
    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------
    st.title("📈 Prediksi Inflasi Bulanan 38 Provinsi Indonesia")
    st.markdown("**Menggunakan Support Vector Regression (SVR) dan Artificial Neural Network (ANN)**")
    st.divider()
    
    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Pengaturan Prediksi")
        
        # Pilih Provinsi
        province_list = sorted(list(models['metadata']['province_mapping_reverse'].keys()))
        selected_province = st.selectbox(
            "📍 Pilih Provinsi",
            options=province_list,
            index=province_list.index("JAWA BARAT") if "JAWA BARAT" in province_list else 0
        )
        
        # Pilih Subkategori
        subcategory_list = sorted(list(models['metadata']['subcategory_mapping_reverse'].keys()))
        selected_subcategory = st.selectbox(
            "📋 Pilih Subkategori",
            options=subcategory_list,
            index=0  # Default "Total"
        )
        
        # Pilih Tahun
        current_year = datetime.now().year
        selected_year = st.selectbox(
            "📅 Tahun",
            options=list(range(2024, 2030)),
            index=list(range(2024, 2030)).index(current_year)
        )
        
        # Pilih Bulan
        month_names = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", 
                       "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
        selected_month = st.selectbox(
            "📆 Bulan",
            options=month_names,
            index=datetime.now().month - 1 if datetime.now().year == selected_year else 0
        )
        month_num = month_names.index(selected_month) + 1
        
        st.divider()
        
        # Tombol Prediksi
        if st.button("🔮 Prediksi Sekarang", type="primary", use_container_width=True):
            st.session_state['predict'] = True
            st.session_state['inputs'] = {
                'province': selected_province,
                'subcategory': selected_subcategory,
                'year': selected_year,
                'month': selected_month,
                'month_num': month_num
            }
    
    # --------------------------------------------------------
    # MAIN CONTENT
    # --------------------------------------------------------
    
    # Tab Navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Prediksi", 
        "📊 Perbandingan Model", 
        "📈 Visualisasi", 
        "ℹ️ Informasi"
    ])
    
    # --------------------------------------------------------
    # TAB 1: PREDIKSI
    # --------------------------------------------------------
    with tab1:
        st.header("🎯 Hasil Prediksi Inflasi")
        
        if 'predict' in st.session_state and st.session_state['predict']:
            inputs = st.session_state['inputs']
            
            # Tampilkan input summary
            with st.expander("📋 Detail Input Prediksi", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Provinsi", inputs['province'])
                with col2:
                    st.metric("Subkategori", inputs['subcategory'])
                with col3:
                    st.metric("Periode", f"{inputs['month']} {inputs['year']}")
            
            # Lakukan prediksi
            with st.spinner("🔮 Memproses prediksi..."):
                results = predict_inflation(
                    models,
                    inputs['year'],
                    inputs['month_num'],
                    inputs['province'],
                    inputs['subcategory']
                )
            
            st.divider()
            
            # Tampilkan hasil prediksi
            st.subheader("📊 Hasil Prediksi dari Kedua Model")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="🤖 SVR (RBF Kernel)",
                    value=f"{results['svr']:.4f}%",
                    delta=None,
                    help="Support Vector Regression dengan kernel RBF"
                )
            
            with col2:
                st.metric(
                    label="🧠 ANN (MLP)",
                    value=f"{results['ann']:.4f}%",
                    delta=None,
                    help="Artificial Neural Network dengan 2 hidden layers"
                )
            
            with col3:
                st.metric(
                    label="⭐ Ensemble (Rata-rata)",
                    value=f"{results['ensemble']:.4f}%",
                    delta=None,
                    help="Rata-rata dari kedua model untuk hasil lebih robust"
                )
            
            st.divider()
            
            # Interpretasi hasil
            st.subheader("📝 Interpretasi")
            
            pred_value = results['ensemble']
            
            if pred_value > 0.5:
                status = "🔴 **TINGGI** - Mengalami inflasi signifikan"
                color = "#ff4444"
            elif pred_value > 0.1:
                status = "🟠 **SEDANG** - Mengalami inflasi moderat"
                color = "#ff9800"
            elif pred_value > -0.1:
                status = "🟢 **RENDAH/STABIL** - Inflasi sangat rendah atau stabil"
                color = "#4caf50"
            else:
                status = "🔵 **DEFLASI** - Mengalami penurunan harga"
                color = "#2196f3"
            
            st.markdown(f"""
            <div style="background-color: {color}15; padding: 20px; border-radius: 10px; border-left: 5px solid {color};">
                <h4>Periode: {inputs['month']} {inputs['year']}</h4>
                <p><strong>Provinsi:</strong> {inputs['province']}</p>
                <p><strong>Subkategori:</strong> {inputs['subcategory']}</p>
                <p><strong>Prediksi Inflasi (Ensemble):</strong> <span style="font-size: 24px; font-weight: bold;">{pred_value:.4f}%</span></p>
                <p><strong>Status:</strong> {status}</p>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.info("👈 Silakan atur parameter di sidebar dan klik 'Prediksi Sekarang' untuk melihat hasil.")
    
    # --------------------------------------------------------
    # TAB 2: PERBANDINGAN MODEL
    # --------------------------------------------------------
    with tab2:
        st.header("📊 Perbandingan Performa Model")
        
        # Evaluasi model
        metrics = evaluate_models(models)
        
        if metrics:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("SVR (RBF Kernel)")
                st.metric("RMSE", f"{metrics['SVR']['RMSE']:.4f}")
                st.metric("MAE", f"{metrics['SVR']['MAE']:.4f}")
                st.metric("R² Score", f"{metrics['SVR']['R2']:.4f}")
                
                if metrics['SVR']['R2'] > 0.7:
                    st.success("✅ Model performa BAIK")
                elif metrics['SVR']['R2'] > 0.5:
                    st.warning("⚠️ Model performa CUKUP")
                else:
                    st.error("❌ Model performa KURANG BAIK")
            
            with col2:
                st.subheader("ANN (MLPRegressor)")
                st.metric("RMSE", f"{metrics['ANN']['RMSE']:.4f}")
                st.metric("MAE", f"{metrics['ANN']['MAE']:.4f}")
                st.metric("R² Score", f"{metrics['ANN']['R2']:.4f}")
                
                if metrics['ANN']['R2'] > 0.7:
                    st.success("✅ Model performa BAIK")
                elif metrics['ANN']['R2'] > 0.5:
                    st.warning("⚠️ Model performa CUKUP")
                else:
                    st.error("❌ Model performa KURANG BAIK")
            
            st.divider()
            
            # Bar chart perbandingan
            st.subheader("Perbandingan Metrik Evaluasi")
            
            metric_df = pd.DataFrame({
                'Model': ['SVR', 'ANN'],
                'RMSE': [metrics['SVR']['RMSE'], metrics['ANN']['RMSE']],
                'MAE': [metrics['SVR']['MAE'], metrics['ANN']['MAE']],
                'R2': [metrics['SVR']['R2'], metrics['ANN']['R2']]
            })
            
            # RMSE comparison
            fig_rmse = px.bar(
                metric_df,
                x='Model',
                y='RMSE',
                color='Model',
                title='Perbandingan RMSE (Semakin Rendah Semakin Baik)',
                color_discrete_sequence=['steelblue', 'coral']
            )
            fig_rmse.update_layout(yaxis_title="RMSE")
            st.plotly_chart(fig_rmse, use_container_width=True)
            
            # R2 comparison
            fig_r2 = px.bar(
                metric_df,
                x='Model',
                y='R2',
                color='Model',
                title='Perbandingan R² Score (Semakin Tinggi Semakin Baik)',
                color_discrete_sequence=['steelblue', 'coral']
            )
            fig_r2.update_layout(yaxis_title="R² Score")
            st.plotly_chart(fig_r2, use_container_width=True)
        
        else:
            st.info("ℹ️ Data evaluasi tidak tersedia.")
    
    # --------------------------------------------------------
    # TAB 3: VISUALISASI
    # --------------------------------------------------------
    with tab3:
        st.header("📈 Visualisasi Prediksi")
        
        if 'predict' in st.session_state and st.session_state['predict']:
            inputs = st.session_state['inputs']
            
            # Prediksi untuk 12 bulan ke depan
            st.subheader(f"Prediksi 12 Bulan ke Depan - {inputs['province']}")
            
            future_predictions = []
            current_year = inputs['year']
            current_month = inputs['month_num']
            
            for i in range(12):
                month_offset = current_month + i
                pred_year = current_year + (month_offset - 1) // 12
                pred_month = ((month_offset - 1) % 12) + 1
                
                pred = predict_inflation(
                    models,
                    pred_year,
                    pred_month,
                    inputs['province'],
                    inputs['subcategory']
                )
                
                future_predictions.append({
                    'Bulan': month_names[pred_month - 1],
                    'Tahun': pred_year,
                    'SVR': pred['svr'],
                    'ANN': pred['ann'],
                    'Ensemble': pred['ensemble']
                })
            
            pred_df = pd.DataFrame(future_predictions)
            
            # Plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=pred_df['Bulan'],
                y=pred_df['SVR'],
                mode='lines+markers',
                name='SVR',
                line=dict(color='steelblue', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=pred_df['Bulan'],
                y=pred_df['ANN'],
                mode='lines+markers',
                name='ANN',
                line=dict(color='coral', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=pred_df['Bulan'],
                y=pred_df['Ensemble'],
                mode='lines+markers',
                name='Ensemble',
                line=dict(color='green', width=3, dash='dash')
            ))
            
            fig.update_layout(
                title=f'Tren Prediksi Inflasi {inputs["province"]} - {inputs["subcategory"]}',
                xaxis_title='Bulan',
                yaxis_title='Inflasi Rate (%)',
                hovermode='x unified',
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tampilkan tabel
            st.subheader("Tabel Prediksi")
            st.dataframe(pred_df.style.format({
                'SVR': '{:.4f}%',
                'ANN': '{:.4f}%',
                'Ensemble': '{:.4f}%'
            }), use_container_width=True)
        
        else:
            st.info("👈 Lakukan prediksi terlebih dahulu untuk melihat visualisasi.")
    
    # --------------------------------------------------------
    # TAB 4: INFORMASI
    # --------------------------------------------------------
    with tab4:
        st.header("ℹ️ Tentang Aplikasi")
        
        st.markdown("""
        ### 📚 Deskripsi
        
        Aplikasi ini merupakan sistem prediksi inflasi bulanan untuk 38 provinsi di Indonesia 
        pada kategori **Perlengkapan, Peralatan dan Pemeliharaan Rutin Rumah Tangga**.
        
        ### 🤖 Model Machine Learning
        
        Aplikasi ini menggunakan dua model machine learning:
        
        1. **Support Vector Regression (SVR)** dengan kernel RBF
           - Efektif untuk data dengan pola non-linear
           - Robust terhadap outliers
        
        2. **Artificial Neural Network (ANN)** dengan arsitektur MLP
           - 2 hidden layers (100 dan 50 neurons)
           - Activation function: ReLU
           - Optimizer: Adam
        
        ### 📊 Fitur yang Digunakan
        
        - Tahun
        - Bulan (1-12)
        - Provinsi (encoded)
        - Subkategori (encoded)
        - Tipe data (bulanan/tahunan)
        
        ### 🎯 Subkategori yang Tersedia
        
        1. Total
        2. Furnitur Perlengkapan dan Karpet
        3. Tekstil Rumah Tangga
        4. Peralatan Rumah Tangga
        5. Barang Pecah Belah dan Peralatan Makan Minum
        6. Peralatan dan Perlengkapan Perumahan dan Kebun
        7. Barang dan Layanan untuk Pemeliharaan Rumah Tangga Rutin
        
        ### 📈 Metrik Evaluasi
        
        - **RMSE** (Root Mean Squared Error): Semakin rendah semakin baik
        - **MAE** (Mean Absolute Error): Rata-rata kesalahan absolut
        - **R² Score**: Koefisien determinasi (0-1, semakin tinggi semakin baik)
        
        ### 👨‍💻 Dikembangkan untuk Skripsi
        
        Aplikasi ini dikembangkan sebagai bagian dari penelitian skripsi 
        di bidang Data Science dan Machine Learning.
        """)
        
        # Model info
        st.divider()
        st.subheader("📊 Informasi Teknis Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### SVR Hyperparameters")
            st.code("""
kernel='rbf'
C=100
gamma=0.1
epsilon=0.01
max_iter=10000
""")
        
        with col2:
            st.markdown("#### ANN Hyperparameters")
            st.code("""
hidden_layer_sizes=(100, 50)
activation='relu'
solver='adam'
alpha=0.0001
learning_rate='adaptive'
max_iter=1000
early_stopping=True
""")
        
        # Dataset info
        st.divider()
        st.subheader("📂 Informasi Dataset")
        
        st.markdown(f"""
        - **Sumber Data**: BPS (Badan Pusat Statistik)
        - **Periode**: {models['metadata']['date_range']['start']} s/d {models['metadata']['date_range']['end']}
        - **Jumlah Provinsi**: {len(models['metadata']['province_mapping'])}
        - **Jumlah Subkategori**: {len(models['metadata']['subcategory_mapping'])}
        - **Total Observasi**: {len(models['metadata']['province_mapping']) * 12 * 3:,}
        - **Tipe Data**: Inflasi Month-to-Month (M-to-M)
        """)
    
    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------
    st.divider()
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: #666;">
        <p>📈 Prediksi Inflasi Bulanan 38 Provinsi Indonesia</p>
        <p><small>Dikembangkan dengan Python, Streamlit, Scikit-learn | © 2026</small></p>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------
# RUN APP
# ------------------------------------------------------------
if __name__ == "__main__":
    main()