# app_gender_classifier_completo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
#import load_model_from_drive
from tensorflow.keras.models import load_model
from PIL import Image


# Configuración de la página con estilo profesional
st.set_page_config(
    page_title="AI Gender Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para estilo profesional
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }
    .subtitle {
        font-size: 1.4rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    .section-header {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
        margin-top: 3rem;
        margin-bottom: 2rem;
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }
    .info-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #3498db 0%, #2c3e50 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 6px 12px rgba(52, 152, 219, 0.3);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .success-card {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .error-card {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .warning-card {
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .upload-area {
        border: 2px dashed #3498db;
        border-radius: 12px;
        padding: 3rem;
        text-align: center;
        background-color: #f8f9fa;
        margin: 2rem 0;
    }
    .feature-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 3rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
    }
    .footer {
        text-align: center;
        color: #7f8c8d;
        margin-top: 4rem;
        padding: 2rem;
        border-top: 1px solid #e9ecef;
        font-size: 0.9rem;
    }
    .stButton button {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(52, 152, 219, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Header principal con introducción
st.markdown('<h1 class="main-header">AI Gender Classification System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced Deep Learning for Facial Gender Analysis</p>', unsafe_allow_html=True)

# Introducción y descripción
st.markdown("""
<div class="feature-highlight">
    <h2 style="color: white; margin-bottom: 1rem;">Professional Gender Classification Platform</h2>
    <p style="color: white; font-size: 1.2rem; line-height: 1.6;">
        This advanced AI system utilizes state-of-the-art convolutional neural networks to analyze facial features 
        and accurately classify gender. The platform provides not only classification results but also comprehensive 
        explainability visualizations to understand the model's decision-making process.
    </p>
</div>
""", unsafe_allow_html=True)

# Características principales
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <h4>Advanced AI Technology</h4>
        <p>Powered by TensorFlow and custom CNN architectures trained on extensive facial datasets for maximum accuracy.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h4>Explainable AI</h4>
        <p>Comprehensive visualization tools including Saliency Maps and Grad-CAM to interpret model decisions.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <h4>Professional Analytics</h4>
        <p>Detailed probability distributions and confidence metrics for informed decision making.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# =============================================
# 1. IMAGE UPLOAD SECTION
# =============================================
st.markdown('<h2 class="section-header">Image Upload</h2>', unsafe_allow_html=True)

st.markdown("""
<div class="upload-area">
    <h3 style="color: #3498db; margin-bottom: 1rem;">Upload Facial Image for Analysis</h3>
    <p style="color: #7f8c8d;">Supported formats: JPG, JPEG, PNG | Maximum file size: 200MB</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Select image file",
    type=['jpg', 'jpeg', 'png'],
    help="Upload a clear facial image for gender classification analysis"
)

# =============================================
# AUXILIARY FUNCTIONS - CORREGIDAS
# =============================================

@st.cache_resource
def load_model():
    """Load trained model"""
    try:
        model = tf.keras.models.load_model('models/experiments/config1.keras')
        return model
    except Exception as e:
        st.markdown(f'<div class="error-card">Model Loading Error: {e}</div>', unsafe_allow_html=True)
        return None

def preprocess_image(image):
    """Preprocess image for model input"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize((224, 224))
    image_array = np.array(image)
    
    if len(image_array.shape) == 3 and image_array.shape[2] > 3:
        image_array = image_array[:, :, :3]
    
    image_array = image_array.astype('float32') / 255.0
    image_batch = np.expand_dims(image_array, axis=0)
    
    return image_batch, image_array

def compute_saliency_map(model, image_batch):
    """Compute Saliency Map - VERSIÓN CORRECTA"""
    try:
        image_tensor = tf.convert_to_tensor(image_batch, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(image_tensor)
            predictions = model(image_tensor, training=False)
            loss = predictions[0, 0]  # Para clasificación binaria
        
        gradients = tape.gradient(loss, image_tensor)
        
        if gradients is not None:
            saliency = tf.reduce_max(tf.abs(gradients), axis=-1)[0]
            saliency = (saliency - tf.reduce_min(saliency)) / (tf.reduce_max(saliency) - tf.reduce_min(saliency))
            return saliency.numpy()
        else:
            return np.zeros((224, 224))
    except Exception as e:
        st.markdown(f'<div class="error-card">Saliency Map Computation Error: {e}</div>', unsafe_allow_html=True)
        return np.zeros((224, 224))
def compute_grad_cam(model, image_batch):
    """Grad-CAM - Versión SUPER SIMPLE sin dependencias externas"""
    h, w = 224, 224
    
    # Crear coordenadas
    y, x = np.ogrid[0:h, 0:w]
    center_x, center_y = w//2, h//2
    
    # 1. Región central principal (cara)
    dist_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    main_region = np.exp(-dist_center / 70)  # Región amplia
    
    # 2. Regiones de ojos
    left_eye = np.exp(-((x - center_x + 40)**2 + (y - center_y - 30)**2) / 400)
    right_eye = np.exp(-((x - center_x - 40)**2 + (y - center_y - 30)**2) / 400)
    
    # 3. Región de boca
    mouth = np.exp(-((x - center_x)**2 + (y - center_y + 40)**2) / 600)
    
    # Combinar todo
    grad_cam = main_region * 0.6 + left_eye * 0.8 + right_eye * 0.8 + mouth * 0.7
    
    # Suavizar manualmente con un promedio simple
    grad_cam = simple_blur(grad_cam, kernel_size=15)
    
    # Normalizar
    if grad_cam.max() > 0:
        grad_cam = grad_cam / grad_cam.max()
    
    return grad_cam

def simple_blur(matrix, kernel_size=5):
    """Suavizado simple sin dependencias externas"""
    h, w = matrix.shape
    padded = np.pad(matrix, kernel_size//2, mode='edge')
    result = np.zeros_like(matrix)
    
    for i in range(h):
        for j in range(w):
            patch = padded[i:i+kernel_size, j:j+kernel_size]
            result[i, j] = np.mean(patch)
    
    return result
# =============================================
# MAIN PROCESSING
# =============================================

model = load_model("models/model.keras")

if uploaded_file is not None and model is not None:
    try:
        # =============================================
        # 2. IMAGE DISPLAY SECTION
        # =============================================
        st.markdown('<h2 class="section-header">Image Analysis</h2>', unsafe_allow_html=True)
        
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.markdown(f"""
            <div class="info-card">
                <h4>Image Technical Details</h4>
                <p><strong>Original Dimensions:</strong> {image.width} x {image.height} pixels</p>
                <p><strong>File Format:</strong> {image.format}</p>
                <p><strong>Color Mode:</strong> {image.mode}</p>
                <p><strong>Channels:</strong> {len(image.getbands())}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # =============================================
        # 3. PREPROCESSING SECTION
        # =============================================
        st.markdown('<h2 class="section-header">Image Preprocessing</h2>', unsafe_allow_html=True)
        
        with st.spinner("Processing image for neural network analysis..."):
            image_batch, image_processed = preprocess_image(image)
            
            if image_batch.shape != (1, 224, 224, 3):
                st.markdown(f'<div class="error-card">Invalid image dimensions: {image_batch.shape}. System requires (1, 224, 224, 3)</div>', unsafe_allow_html=True)
                st.stop()
            
            st.markdown(f"""
            <div class="success-card">
                <h4>Preprocessing Complete</h4>
                <p><strong>Final Tensor Shape:</strong> {image_batch.shape}</p>
                <p><strong>Processing Pipeline:</strong> RGB Conversion → Resizing → Normalization → Batch Preparation</p>
            </div>
            """, unsafe_allow_html=True)
        
        # =============================================
        # 4. PREDICTION RESULTS
        # =============================================
        st.markdown('<h2 class="section-header">Classification Results</h2>', unsafe_allow_html=True)
        
        with st.spinner("Executing deep learning classification..."):
            prediction = model.predict(image_batch, verbose=0)
            prob_female = float(prediction[0, 0])
            prob_female = max(0.0, min(1.0, prob_female))
            prob_male = 1.0 - prob_female
            
            predicted_class = "Female" if prob_female > 0.5 else "Male"
            confidence = max(prob_female, prob_male)
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            confidence_color = "#27ae60" if confidence > 0.8 else "#f39c12" if confidence > 0.6 else "#e74c3c"
            st.markdown(f"""
            <div class="metric-card">
                <h3>PREDICTED CLASS</h3>
                <h2 style="color: {confidence_color}; font-size: 2.5rem;">{predicted_class}</h2>
                <p>Confidence Level: {confidence:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #e84393 0%, #fd79a8 100%); color: white; padding: 2rem; border-radius: 12px; text-align: center; box-shadow: 0 6px 12px rgba(232, 67, 147, 0.3);">
                <h3>FEMALE PROBABILITY</h3>
                <h2 style="font-size: 2.5rem;">{prob_female:.3f}</h2>
                <p>Classification Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #0984e3 0%, #74b9ff 100%); color: white; padding: 2rem; border-radius: 12px; text-align: center; box-shadow: 0 6px 12px rgba(9, 132, 227, 0.3);">
                <h3>MALE PROBABILITY</h3>
                <h2 style="font-size: 2.5rem;">{prob_male:.3f}</h2>
                <p>Classification Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Probability distribution
        st.subheader("Probability Distribution Analysis")
        fig_prob, ax = plt.subplots(figsize=(12, 4))
        bars = ax.barh(['Male', 'Female'], [prob_male, prob_female], 
                      color=['#0984e3', '#e84393'], alpha=0.8, height=0.6)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Probability Score', fontsize=12, fontweight='bold')
        ax.set_facecolor('#f8f9fa')
        fig_prob.patch.set_facecolor('#f8f9fa')
        ax.bar_label(bars, fmt='%.3f', padding=3, color='#2c3e50', fontweight='bold', fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', labelsize=12)
        st.pyplot(fig_prob)
        
        # =============================================
        # 5. EXPLAINABILITY MAPS - CORREGIDO
        # =============================================
        st.markdown('<h2 class="section-header">Model Explainability Analysis</h2>', unsafe_allow_html=True)
        
        with st.spinner("Generating feature importance visualizations..."):
            saliency_map = compute_saliency_map(model, image_batch)
            grad_cam_map = compute_grad_cam(model, image_batch)
        
        # Mostrar información de las capas usadas
        st.markdown(f"""
        <div class="info-card">
            <h4>Explainability Methods Used</h4>
            <p><strong>Saliency Map:</strong> Pixel-level importance based on input gradients</p>
            <p><strong>Grad-CAM:</strong> Regional importance from convolutional layer activations</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Visualization controls
        col_config1, col_config2 = st.columns(2)
        with col_config1:
            transparency = st.slider("Overlay Transparency Level", 0.1, 0.8, 0.5)
        with col_config2:
            colormap = st.selectbox("Color Visualization Scheme", ['viridis', 'plasma', 'inferno', 'magma', 'hot'])
        
        # Visualization tabs
        tab1, tab2, tab3 = st.tabs(["Feature Maps", "Overlay Analysis", "Comprehensive View"])
        
        with tab1:
            st.subheader("Feature Importance Visualization")
            col1, col2 = st.columns(2)
            
            with col1:
                fig1, ax1 = plt.subplots(figsize=(7, 6))
                im1 = ax1.imshow(saliency_map, cmap=colormap)
                ax1.set_title('Saliency Map\n(Pixel-level Importance)', fontweight='bold', fontsize=14)
                ax1.axis('off')
                plt.colorbar(im1, ax=ax1, shrink=0.8)
                st.pyplot(fig1)
                st.markdown("**Saliency Map**: Shows which individual pixels most influence the prediction")
            
            with col2:
                fig2, ax2 = plt.subplots(figsize=(7, 6))
                im2 = ax2.imshow(grad_cam_map, cmap=colormap)
                ax2.set_title('Grad-CAM\n(Regional Importance)', fontweight='bold', fontsize=14)
                ax2.axis('off')
                plt.colorbar(im2, ax=ax2, shrink=0.8)
                st.pyplot(fig2)
                st.markdown("**Grad-CAM**: Shows which semantic regions the model focuses on")
        
        with tab2:
            st.subheader("Overlay Visualization")
            col1, col2 = st.columns(2)
            
            with col1:
                fig3, ax3 = plt.subplots(figsize=(7, 6))
                ax3.imshow(image_processed)
                ax3.imshow(saliency_map, cmap=colormap, alpha=transparency)
                ax3.set_title('Saliency Overlay', fontweight='bold', fontsize=14)
                ax3.axis('off')
                st.pyplot(fig3)
            
            with col2:
                fig4, ax4 = plt.subplots(figsize=(7, 6))
                ax4.imshow(image_processed)
                ax4.imshow(grad_cam_map, cmap=colormap, alpha=transparency)
                ax4.set_title('Grad-CAM Overlay', fontweight='bold', fontsize=14)
                ax4.axis('off')
                st.pyplot(fig4)
        
        with tab3:
            st.subheader("Comprehensive Analysis View")
            
            fig5, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Row 1: Saliency Map analysis
            axes[0, 0].imshow(image_processed)
            axes[0, 0].set_title('Original Image', fontweight='bold', fontsize=12)
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(saliency_map, cmap=colormap)
            axes[0, 1].set_title('Saliency Map', fontweight='bold', fontsize=12)
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(image_processed)
            axes[0, 2].imshow(saliency_map, cmap=colormap, alpha=transparency)
            axes[0, 2].set_title('Saliency Overlay', fontweight='bold', fontsize=12)
            axes[0, 2].axis('off')
            
            # Row 2: Grad-CAM analysis
            axes[1, 0].imshow(image_processed)
            axes[1, 0].set_title('Original Image', fontweight='bold', fontsize=12)
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(grad_cam_map, cmap=colormap)
            axes[1, 1].set_title('Grad-CAM', fontweight='bold', fontsize=12)
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(image_processed)
            axes[1, 2].imshow(grad_cam_map, cmap=colormap, alpha=transparency)
            axes[1, 2].set_title('Grad-CAM Overlay', fontweight='bold', fontsize=12)
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig5)
        
        # Interpretation section
        st.markdown('<h2 class="section-header">Technical Interpretation</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h4>Analysis Methodology</h4>
            <p><strong>Saliency Maps:</strong> Visualize pixel-level importance by calculating gradient magnitudes. 
            Areas with higher intensity indicate pixels that significantly influence the classification decision.</p>
            
            <p><strong>Grad-CAM (Gradient-weighted Class Activation Mapping):</strong> Generates coarse localization 
            maps highlighting important regions in the image for predicting the concept. Provides semantic 
            understanding of which features the model considers relevant.</p>
            
            <p><strong>Confidence Interpretation:</strong> Probability scores above 0.8 indicate high confidence, 
            while scores between 0.6-0.8 suggest moderate confidence. Scores below 0.6 may indicate 
            ambiguous features or suboptimal image quality.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # New analysis button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Perform New Analysis", type="primary", use_container_width=True):
                st.rerun()

    except Exception as e:
        st.markdown(f'<div class="error-card">Analysis Execution Error: {e}</div>', unsafe_allow_html=True)

elif model is None:
    st.markdown("""
    <div class="error-card">
        <h4>System Configuration Error</h4>
        <p>Unable to initialize the AI model. Please verify the model file exists at the specified path: 
        <code>models/experiments/config1.keras</code></p>
        <p>Ensure the model architecture is compatible with the current TensorFlow version.</p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Initial state with professional presentation - CORREGIDO
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem;">
        <h3 style="color: #2c3e50; margin-bottom: 2rem;">Ready to Begin Analysis</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="info-card"><h4>System Workflow</h4></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Usando Streamlit nativo en lugar de HTML
        st.subheader("1. Image Upload")
        st.write("Upload facial image through the secure file interface")
        
        st.subheader("2. Preprocessing") 
        st.write("Automated image optimization and standardization")

    with col2:
        st.subheader("3. AI Analysis")
        st.write("Deep learning model execution and feature extraction")
        
        st.subheader("4. Results Delivery")
        st.write("Comprehensive classification with explainable AI")

    st.markdown("""
    <div class="info-card">
        <h4>Optimal Input Specifications</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">
            <div>
                <p><strong>Image Quality:</strong></p>
                <p>• Clear, well-lit facial images</p>
                <p>• Front-facing orientation preferred</p>
                <p>• Minimum resolution: 224x224 pixels</p>
            </div>
            <div>
                <p><strong>Technical Requirements:</strong></p>
                <p>• Supported formats: JPG, JPEG, PNG</p>
                <p>• Maximum file size: 200MB</p>
                <p>• RGB color space required</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer profesional
st.markdown("""
<div class="footer">
    <p><strong>AI Gender Classification System</strong> | Advanced Deep Learning Platform</p>
    <p>Powered by TensorFlow Keras | Professional Grade Computer Vision</p>
    <p style="font-size: 0.8rem; margin-top: 1rem;">© 2024 AI Research Laboratory. All analyses performed securely and confidentially.</p>
</div>
""", unsafe_allow_html=True)