"""
Iris Flower Classification - Streamlit Web Application
Author: AI Assistant
Description:
    An interactive, premium web application built with Streamlit.
    Allows users to input iris flower measurements, choose from five different
    trained ML models, view predictions with confidence percentages, browse
    botanical details, and inspect model performance and dataset EDA.
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import joblib

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="Iris Species Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# Custom Premium Styling (CSS)
# ----------------------------------------------------
st.markdown("""
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
    }
    
    /* Main App Header Accent */
    .header-container {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #EC4899 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(124, 58, 237, 0.3);
        text-align: center;
    }
    .header-title {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.025em;
    }
    .header-subtitle {
        font-size: 1.15rem;
        font-weight: 300;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Metric Card Styling */
    .custom-card {
        background-color: #1E1E1E;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
        padding: 1.5rem;
        border: 1px solid #334155;
        transition: all 0.3s ease;
        margin-bottom: 1rem;
        color: #FFFFFF;
    }
    .custom-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
    }
    
    /* Feature Value Badge */
    .feature-badge {
        display: inline-block;
        background-color: #334155;
        color: #FFFFFF;
        font-weight: 600;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        margin-right: 0.5rem;
    }
    
    /* Botanical Fact Card */
    .botanical-card {
        background: linear-gradient(to right, #1E1E1E, #111111);
        border-left: 5px solid #10B981;
        padding: 1.25rem;
        border-radius: 0 12px 12px 0;
        margin-top: 1rem;
        color: #FFFFFF;
    }
    
    /* Prediction Banner */
    .pred-setosa {
        background: linear-gradient(135deg, #064E3B 0%, #065F46 100%);
        border: 1px solid #10B981;
        color: #ECFDF5;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .pred-versicolor {
        background: linear-gradient(135deg, #1E3A8A 0%, #1E40AF 100%);
        border: 1px solid #3B82F6;
        color: #EFF6FF;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .pred-virginica {
        background: linear-gradient(135deg, #581C87 0%, #701A75 100%);
        border: 1px solid #D946EF;
        color: #FDF4FF;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .pred-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    .pred-subtitle {
        font-size: 0.95rem;
        opacity: 0.85;
        margin-top: 0.25rem;
    }
    
    /* Hide default Streamlit footer */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Helper Functions
# ----------------------------------------------------
@st.cache_resource
def load_ml_assets():
    """Loads and caches the scaler and trained classification models."""
    scaler = joblib.load(os.path.join('models', 'scaler.joblib'))
    
    models = {
        'Logistic Regression': joblib.load(os.path.join('models', 'logistic_regression.joblib')),
        'K-Nearest Neighbors': joblib.load(os.path.join('models', 'knn.joblib')),
        'Decision Tree': joblib.load(os.path.join('models', 'decision_tree.joblib')),
        'Random Forest': joblib.load(os.path.join('models', 'random_forest.joblib')),
        'Support Vector Machine': joblib.load(os.path.join('models', 'svm.joblib'))
    }
    return scaler, models

@st.cache_data
def load_dataset_and_metrics():
    """Loads and caches raw dataset and training metrics JSON."""
    df = pd.read_csv(os.path.join('dataset', 'iris.csv'))
    
    with open(os.path.join('models', 'metrics.json'), 'r') as f:
        metrics = json.load(f)
        
    return df, metrics

# Load assets (handles exceptions if models aren't trained yet)
try:
    scaler, models = load_ml_assets()
    df, metrics = load_dataset_and_metrics()
    assets_loaded = True
except Exception as e:
    assets_loaded = False
    st.error(f"Error loading models or dataset. Please run 'model_training.py' first. Details: {e}")

# ----------------------------------------------------
# App Layout
# ----------------------------------------------------
# Premium Header
st.markdown("""
    <div class="header-container">
        <h1 class="header-title">Iris Flower Classification Dashboard</h1>
        <p class="header-subtitle">Analyze, classify, and compare machine learning models on Scikit-Learn's classical Iris dataset.</p>
    </div>
""", unsafe_allow_html=True)

if assets_loaded:
    # ----------------------------------------------------
    # Sidebar: Model and Feature Inputs
    # ----------------------------------------------------
    st.sidebar.markdown("### 🛠️ Configuration")
    
    # 1. Model Selection
    selected_model_name = st.sidebar.selectbox(
        "Choose Machine Learning Model",
        options=list(models.keys()),
        index=4  # SVM as default
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📏 Flower Measurements")
    
    # Sliders based on min/max of dataset
    sepal_length = st.sidebar.slider(
        "Sepal Length (cm)",
        min_value=float(df['sepal length (cm)'].min()),
        max_value=float(df['sepal length (cm)'].max()),
        value=5.8,
        step=0.1,
        help="Length of the flower's sepal in centimeters."
    )
    
    sepal_width = st.sidebar.slider(
        "Sepal Width (cm)",
        min_value=float(df['sepal width (cm)'].min()),
        max_value=float(df['sepal width (cm)'].max()),
        value=3.0,
        step=0.1,
        help="Width of the flower's sepal in centimeters."
    )
    
    petal_length = st.sidebar.slider(
        "Petal Length (cm)",
        min_value=float(df['petal length (cm)'].min()),
        max_value=float(df['petal length (cm)'].max()),
        value=4.35,
        step=0.1,
        help="Length of the flower's petal in centimeters."
    )
    
    petal_width = st.sidebar.slider(
        "Petal Width (cm)",
        min_value=float(df['petal width (cm)'].min()),
        max_value=float(df['petal width (cm)'].max()),
        value=1.3,
        step=0.1,
        help="Width of the flower's petal in centimeters."
    )
    
    # ----------------------------------------------------
    # Main Tabs
    # ----------------------------------------------------
    tab1, tab2, tab3 = st.tabs([
        "🌸 Species Predictor", 
        "📈 Model Performance Dashboard", 
        "📊 Exploratory Data Analysis"
    ])
    
    # ----------------------------------------------------
    # Tab 1: Species Predictor
    # ----------------------------------------------------
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Input Features Summary")
            
            # Interactive visualization of input feature cards
            st.markdown(f"""
                <div class="custom-card">
                    <p style='margin-bottom:0.8rem; font-weight:500; color:#64748B;'>Current Input Values:</p>
                    <span class="feature-badge">Sepal Length: {sepal_length} cm</span>
                    <span class="feature-badge">Sepal Width: {sepal_width} cm</span>
                    <br><br>
                    <span class="feature-badge">Petal Length: {petal_length} cm</span>
                    <span class="feature-badge">Petal Width: {petal_width} cm</span>
                </div>
            """, unsafe_allow_html=True)
            
            # Model prediction pipeline
            input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            scaled_features = scaler.transform(input_features)
            
            active_model = models[selected_model_name]
            
            # Make prediction
            pred_class_idx = active_model.predict(scaled_features)[0]
            species_names = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']
            predicted_species = species_names[pred_class_idx]
            
            # Predict probability if supported
            if hasattr(active_model, "predict_proba"):
                probabilities = active_model.predict_proba(scaled_features)[0]
            else:
                # Fallback for models without native probabilities (like Decision Tree with certain settings, or SVM without probability=True)
                # But our SVC has probability=True, and others have predict_proba
                probabilities = np.zeros(3)
                probabilities[pred_class_idx] = 1.0
                
            # Class-specific card styling class
            css_class = "pred-setosa" if pred_class_idx == 0 else ("pred-versicolor" if pred_class_idx == 1 else "pred-virginica")
            
            st.markdown(f"""
                <div class="{css_class}">
                    <p class="pred-subtitle">CLASSIFIED BY {selected_model_name.upper()}</p>
                    <h2 class="pred-title">{predicted_species}</h2>
                    <p class="pred-subtitle">Confidence Level: {probabilities[pred_class_idx]*100:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Display Prediction Probability Breakdown
            st.markdown("#### Class Probability Breakdown")
            for name, prob in zip(species_names, probabilities):
                st.write(f"**{name}**")
                st.progress(float(prob))
                st.markdown(f"<p style='text-align:right; margin-top:-1.8rem; font-weight:600;'>{prob*100:.1f}%</p>", unsafe_allow_html=True)
                
        with col2:
            st.markdown("### Botanical Profile")
            
            # Botanical details dictionary
            botanical_info = {
                'Iris Setosa': {
                    'description': "Iris setosa, the <b>bristle-pointed iris</b>, is characterized by its small size, striking deep blue-purple color, and extremely narrow petals (known as standards) compared to its large sepals (known as falls).",
                    'habitat': "Coastal beaches, meadows, sand dunes, and subarctic regions of Alaska, Canada, and Northern Asia.",
                    'distinguishing_feature': "Very short standards (petals) that are reduced to bristles, making it easily distinguishable from the other two species."
                },
                'Iris Versicolor': {
                    'description': "Iris versicolor, the <b>blue flag</b> or <b>harlequin blueflag</b>, features beautiful medium-sized violet-blue flowers. The sepals have a distinct yellow and white patch at the base with intricate purple veins.",
                    'habitat': "Wet meadows, marshes, swamps, and along riverbanks and shorelines across North America.",
                    'distinguishing_feature': "Moderate size with prominent yellow-white patches at the throat of the sepals, displaying a classic 'blue flag' shape."
                },
                'Iris Virginica': {
                    'description': "Iris virginica, the <b>Virginia iris</b>, has large, showy, light violet to lavender-blue flowers. The sepals are highly developed, with a bright golden-yellow slash (or signal patch) on their crest.",
                    'habitat': "Swamps, wet savannas, marshes, and ditches in the southeastern and midwestern United States.",
                    'distinguishing_feature': "Typically taller plants with larger petals/sepals and a very prominent bright orange-yellow signal on the sepals."
                }
            }
            
            info = botanical_info[predicted_species]
            
            # Display botanical description
            st.markdown(f"""
                <div class="botanical-card">
                    <p>🌱 <b>Description:</b> {info['description']}</p>
                    <p>📍 <b>Natural Habitat:</b> {info['habitat']}</p>
                    <p>🔍 <b>Key Identification Feature:</b> <i>{info['distinguishing_feature']}</i></p>
                </div>
            """, unsafe_allow_html=True)
            
            # Display Generated Image illustration
            img_filename = "setosa.png" if pred_class_idx == 0 else ("versicolor.png" if pred_class_idx == 1 else "virginica.png")
            img_path = os.path.join('images', img_filename)
            
            if os.path.exists(img_path):
                img = Image.open(img_path)
                st.image(img, caption=f"Botanical Illustration of {predicted_species}", use_container_width=True)
            else:
                st.warning(f"Illustration file '{img_path}' not found. Verify project file structure.")
                
    # ----------------------------------------------------
    # Tab 2: Model Performance Dashboard
    # ----------------------------------------------------
    with tab2:
        st.markdown("### 🏆 Model Comparison & Metrics")
        st.markdown("Explore how each classification algorithm performed on the 20% validation/test subset.")
        
        # Display Metrics Table
        metrics_df = pd.DataFrame(metrics).T
        # Format as percentages for presentation
        formatted_metrics = metrics_df.style.format({
            'Accuracy': '{:.2%}',
            'Precision': '{:.2%}',
            'Recall': '{:.2%}',
            'F1-Score': '{:.2%}'
        })
        
        col_tbl, col_chart = st.columns([2, 3])
        
        with col_tbl:
            st.markdown("#### Performance Metrics Table")
            st.table(formatted_metrics)
            
            st.markdown("""
                > **Note on Evaluation Metrics:**
                > - **Accuracy:** Overall percentage of correctly predicted flowers.
                > - **Precision:** Out of all predicted positives, how many were actually positive (weighted by class size).
                > - **Recall:** Out of all actual positives, how many did the model identify.
                > - **F1-Score:** The harmonic mean of Precision and Recall. The optimal metric for overall classifier health.
            """)
            
        with col_chart:
            st.markdown("#### Model Accuracy Comparison")
            comp_chart_path = os.path.join('images', 'model_comparison.png')
            if os.path.exists(comp_chart_path):
                st.image(Image.open(comp_chart_path), caption="Model Accuracy Bar Chart", use_container_width=True)
            else:
                st.warning("Model comparison chart not found.")
                
        st.markdown("---")
        st.markdown("### 🧩 Confusion Matrices")
        st.markdown("The confusion matrices below illustrate the exact prediction errors made by each model (Actual vs. Predicted).")
        
        cm_path = os.path.join('images', 'confusion_matrices.png')
        if os.path.exists(cm_path):
            st.image(Image.open(cm_path), caption="Confusion Matrices for the 5 Algorithms", use_container_width=True)
        else:
            st.warning("Confusion matrix compilation plot not found.")
            
    # ----------------------------------------------------
    # Tab 3: Exploratory Data Analysis
    # ----------------------------------------------------
    with tab3:
        st.markdown("### 📊 Dataset Exploration & Visualizations")
        st.markdown("Analyze the distribution, relationships, and correlations between the iris measurements.")
        
        # Summary statistics
        if st.checkbox("Show Summary Statistics"):
            st.write(df.describe())
            
        col_dist, col_heat = st.columns([1, 1])
        
        with col_dist:
            st.markdown("#### Feature Distributions")
            dist_img_path = os.path.join('images', 'feature_distribution.png')
            if os.path.exists(dist_img_path):
                st.image(Image.open(dist_img_path), caption="Distribution histograms with KDE curves", use_container_width=True)
            else:
                st.warning("Feature distribution plot not found.")
                
        with col_heat:
            st.markdown("#### Feature Correlation Heatmap")
            heat_img_path = os.path.join('images', 'correlation_heatmap.png')
            if os.path.exists(heat_img_path):
                st.image(Image.open(heat_img_path), caption="Pearson Correlation between measurements", use_container_width=True)
            else:
                st.warning("Correlation heatmap plot not found.")
                
        st.markdown("---")
        st.markdown("#### Pairwise Relationships Pair-Plot")
        pair_img_path = os.path.join('images', 'pair_plot.png')
        if os.path.exists(pair_img_path):
            st.image(Image.open(pair_img_path), caption="Pair-wise features scattered by species classification", use_container_width=True)
        else:
            st.warning("Dataset pair plot not found.")
            
        st.markdown("---")
        st.markdown("#### Raw Dataset Table Viewer")
        st.dataframe(df.drop(columns=['target']), height=300)
else:
    st.info("The application requires trained models and dataset exports. Please ensure the training pipeline runs successfully first.")
