"""
Page 4: 🏋️ Train Model — GUI-based model training interface
Provides controls for dataset selection, hyperparameters, and training progress.
"""

import streamlit as st
import numpy as np
import os
from io import StringIO

# TensorFlow/Keras imports
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback

from utils.model_utils import EMOTIONS

# Set matplotlib for plots
import matplotlib.pyplot as plt
plt.style.use('dark_background')


def show():
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 1rem;">
            <h1>🏋️ Train Emotion Model</h1>
            <p style="color: #8B949E; font-size: 1rem;">
                Configure and train a CNN model on the FER2013 dataset
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ─── Dataset Source ───────────────────────────────────────
    st.markdown("### 📁 Dataset Source")
    
    dataset_source = st.radio(
        "Select dataset source",
        ["Kaggle (auto-download)", "Local Folder"],
        help="Kaggle option automatically downloads FER2013. Local folder lets you point to your own dataset.",
        horizontal=True,
    )
    
    local_path = None
    if dataset_source == "Local Folder":
        local_path = st.text_input(
            "Path to dataset folder",
            placeholder="e.g., /path/to/fer2013/",
            help="Folder should contain 'train' and 'test' subdirectories with emotion class subfolders.",
        )
        if local_path and not os.path.exists(local_path):
            st.error(f"❌ Path does not exist: {local_path}")
            local_path = None

    # ─── Architecture Selection ───────────────────────────────
    st.markdown("### 🧠 Architecture")
    
    arch_type = st.selectbox(
        "Model Architecture",
        ["Standard (3 Conv blocks)", "Lightweight (2 Conv blocks)", "Deep (5 Conv blocks)"],
        help="Standard is the default 3-block CNN. Lightweight is faster. Deep has more capacity.",
    )

    # ─── Hyperparameters ──────────────────────────────────────
    st.markdown("### ⚙️ Hyperparameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        epochs = st.slider(
            "Epochs",
            min_value=1, max_value=200, value=30, step=1,
            help="Number of complete passes through the training dataset.",
        )
        batch_size = st.select_slider(
            "Batch Size",
            options=[16, 32, 64, 128],
            value=64,
            help="Number of samples per gradient update.",
        )
    
    with col2:
        learning_rate = st.slider(
            "Learning Rate",
            min_value=0.0001, max_value=0.01, value=0.001, step=0.0001,
            format="%.4f",
            help="Step size for optimizer. Lower = more precise but slower.",
        )
        dropout_rate = st.slider(
            "Dropout Rate",
            min_value=0.1, max_value=0.5, value=0.25, step=0.05,
            help="Fraction of neurons to drop during training (regularization).",
        )

    # ─── Data Augmentation ────────────────────────────────────
    st.markdown("### 🔄 Data Augmentation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        use_flip = st.checkbox("Horizontal Flip", value=True, help="Randomly flip images horizontally.")
    
    with col2:
        rotation_range = st.slider(
            "Rotation Range (°)",
            min_value=0, max_value=20, value=10, step=1,
            help="Random rotation range in degrees.",
        )
    
    with col3:
        zoom_range = st.slider(
            "Zoom Range",
            min_value=0.0, max_value=0.2, value=0.1, step=0.05,
            help="Random zoom range.",
        )

    # ─── Model Save ───────────────────────────────────────────
    st.markdown("### 💾 Save Options")
    
    model_name = st.text_input(
        "Model filename",
        value="emotion_model.h5",
        help="Name of the saved model file (will be saved to project root).",
    )
    
    # Add .h5 if not present
    if not model_name.endswith('.h5'):
        model_name += '.h5'

    # ─── Train Button ─────────────────────────────────────────
    st.markdown("---")
    
    train_disabled = (dataset_source == "Local Folder" and local_path is None)
    
    if st.button("🚀 Start Training", type="primary", use_container_width=True, disabled=train_disabled):
        _run_training(
            dataset_source=dataset_source,
            local_path=local_path,
            arch_type=arch_type,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            use_flip=use_flip,
            rotation_range=rotation_range,
            zoom_range=zoom_range,
            model_name=model_name,
        )


def _build_model(arch_type, input_shape=(48, 48, 1), num_classes=7, dropout_rate=0.25):
    """Build a CNN model based on the selected architecture type."""
    model = Sequential()
    
    if arch_type == "Lightweight (2 Conv blocks)":
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout_rate))
        
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout_rate))
    
    elif arch_type == "Deep (5 Conv blocks)":
        filter_sizes = [32, 64, 128, 256, 512]
        for i, filters in enumerate(filter_sizes):
            model.add(Conv2D(filters, kernel_size=(3, 3), activation='relu', 
                            input_shape=input_shape if i == 0 else None))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(dropout_rate if i < 4 else 0.5))
    
    else:  # Standard (3 Conv blocks)
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout_rate))
        
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout_rate))
        
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout_rate))
    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model


def _run_training(dataset_source, local_path, arch_type, epochs, batch_size,
                  learning_rate, dropout_rate, use_flip, rotation_range,
                  zoom_range, model_name):
    """Execute the training process with live Streamlit updates."""
    
    # Training status placeholders
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()
    
    status_placeholder.info("🔄 Initializing training...")
    
    try:
        # Download/download dataset
        if dataset_source == "Kaggle (auto-download)":
            status_placeholder.info("📥 Downloading FER2013 dataset from Kaggle...")
            import kagglehub
            path = kagglehub.dataset_download("msambare/fer2013")
            train_path = os.path.join(path, "train")
            test_path = os.path.join(path, "test")
        else:
            train_path = os.path.join(local_path, "train")
            test_path = os.path.join(local_path, "test")
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            status_placeholder.error(f"❌ Dataset paths not found. Check: {train_path} and {test_path}")
            return
        
        status_placeholder.info("📊 Loading data generators...")
        
        # Data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=use_flip,
            rotation_range=rotation_range,
            zoom_range=zoom_range,
        )
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=(48, 48),
            batch_size=batch_size,
            color_mode="grayscale",
            class_mode='categorical',
        )
        
        test_generator = test_datagen.flow_from_directory(
            test_path,
            target_size=(48, 48),
            batch_size=batch_size,
            color_mode="grayscale",
            class_mode='categorical',
        )
        
        status_placeholder.info("🧠 Building model...")
        
        # Build model
        model = _build_model(arch_type, dropout_rate=dropout_rate)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=learning_rate),
            metrics=['accuracy'],
        )
        
        # Model summary
        summary_buffer = StringIO()
        model.summary(print_fn=lambda x: summary_buffer.write(x + '\n'))
        model_summary_str = summary_buffer.getvalue()
        
        with st.expander("Model Architecture Summary"):
            st.code(model_summary_str)
        
        status_placeholder.info("🏋️ Training in progress...")
        
        # Training metrics storage (shared with callback)
        history_data = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        steps_per_epoch = max(1, train_generator.n // train_generator.batch_size)
        validation_steps = max(1, test_generator.n // test_generator.batch_size)
        
        # Custom Keras Callback for live Streamlit updates
        class StreamlitCallback(Callback):
            def __init__(self, epochs):
                super().__init__()
                self.total_epochs = epochs
            
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                
                # Store metrics
                for key in ['loss', 'accuracy', 'val_loss', 'val_accuracy']:
                    if key in logs:
                        history_data[key].append(logs[key])
                
                # Update progress
                progress = (epoch + 1) / self.total_epochs
                progress_bar.progress(progress)
                
                # Update metrics display
                current_loss = history_data['loss'][-1] if history_data['loss'] else 0
                current_acc = history_data['accuracy'][-1] if history_data['accuracy'] else 0
                current_val_loss = history_data['val_loss'][-1] if history_data['val_loss'] else 0
                current_val_acc = history_data['val_accuracy'][-1] if history_data['val_accuracy'] else 0
                
                metrics_placeholder.markdown(
                    f"""
                    <div style="display: flex; gap: 1rem; margin: 1rem 0;">
                        <div style="background: #1C2128; border: 1px solid #30363D; border-radius: 8px; padding: 0.8rem; flex: 1; text-align: center;">
                            <p style="color: #8B949E; font-size: 0.8rem; margin: 0;">Epoch</p>
                            <p style="color: #00D4AA; font-size: 1.5rem; font-weight: 700; margin: 0;">{epoch+1}/{self.total_epochs}</p>
                        </div>
                        <div style="background: #1C2128; border: 1px solid #30363D; border-radius: 8px; padding: 0.8rem; flex: 1; text-align: center;">
                            <p style="color: #8B949E; font-size: 0.8rem; margin: 0;">Loss</p>
                            <p style="color: #FF6B6B; font-size: 1.5rem; font-weight: 700; margin: 0;">{current_loss:.4f}</p>
                        </div>
                        <div style="background: #1C2128; border: 1px solid #30363D; border-radius: 8px; padding: 0.8rem; flex: 1; text-align: center;">
                            <p style="color: #8B949E; font-size: 0.8rem; margin: 0;">Accuracy</p>
                            <p style="color: #2ECC71; font-size: 1.5rem; font-weight: 700; margin: 0;">{current_acc:.2%}</p>
                        </div>
                        <div style="background: #1C2128; border: 1px solid #30363D; border-radius: 8px; padding: 0.8rem; flex: 1; text-align: center;">
                            <p style="color: #8B949E; font-size: 0.8rem; margin: 0;">Val Loss</p>
                            <p style="color: #F39C12; font-size: 1.5rem; font-weight: 700; margin: 0;">{current_val_loss:.4f}</p>
                        </div>
                        <div style="background: #1C2128; border: 1px solid #30363D; border-radius: 8px; padding: 0.8rem; flex: 1; text-align: center;">
                            <p style="color: #8B949E; font-size: 0.8rem; margin: 0;">Val Accuracy</p>
                            <p style="color: #3498DB; font-size: 1.5rem; font-weight: 700; margin: 0;">{current_val_acc:.2%}</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
                # Update chart every 5 epochs
                if (epoch + 1) % 5 == 0 or (epoch + 1) == self.total_epochs:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    
                    axes[0].plot(history_data['loss'], label='Training Loss', color='#FF6B6B')
                    axes[0].plot(history_data['val_loss'], label='Validation Loss', color='#F39C12')
                    axes[0].set_title('Loss', color='white', fontsize=14)
                    axes[0].set_xlabel('Epoch', color='#8B949E')
                    axes[0].set_ylabel('Loss', color='#8B949E')
                    axes[0].legend(loc='upper right')
                    axes[0].grid(True, alpha=0.3)
                    axes[0].set_facecolor('#161B22')
                    axes[0].tick_params(colors='#8B949E')
                    
                    axes[1].plot(history_data['accuracy'], label='Training Accuracy', color='#2ECC71')
                    axes[1].plot(history_data['val_accuracy'], label='Validation Accuracy', color='#3498DB')
                    axes[1].set_title('Accuracy', color='white', fontsize=14)
                    axes[1].set_xlabel('Epoch', color='#8B949E')
                    axes[1].set_ylabel('Accuracy', color='#8B949E')
                    axes[1].legend(loc='lower right')
                    axes[1].grid(True, alpha=0.3)
                    axes[1].set_facecolor('#161B22')
                    axes[1].tick_params(colors='#8B949E')
                    
                    plt.tight_layout()
                    chart_placeholder.pyplot(fig)
                    plt.close()
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, verbose=1),
            StreamlitCallback(epochs),
        ]
        
        # Single fit call - efficient, allows TF to optimize the full training run
        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=test_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=0,
        )
        
        # ─── Training Complete ────────────────────────────────
        progress_bar.progress(1.0)
        
        # Save the model
        model.save(model_name)
        
        status_placeholder.success(f"✅ Training complete! Model saved as '{model_name}'")
        
        # Final metrics
        final_val_acc = history_data['val_accuracy'][-1]
        final_val_loss = history_data['val_loss'][-1]
        
        st.balloons()
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #0A2D15, #1C2128);
                border: 2px solid #2ECC71;
                border-radius: 16px;
                padding: 2rem;
                text-align: center;
                margin-top: 1rem;
            ">
                <h2 style="color: #2ECC71;">🎉 Training Complete!</h2>
                <p style="font-size: 1.2rem; color: #E6EDF3;">
                    Final Validation Accuracy: <strong>{final_val_acc:.2%}</strong><br>
                    Final Validation Loss: <strong>{final_val_loss:.4f}</strong>
                </p>
                <p style="color: #00D4AA;">Model saved as: {model_name}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Download button
        if os.path.exists(model_name):
            with open(model_name, 'rb') as f:
                st.download_button(
                    label="📥 Download Trained Model",
                    data=f,
                    file_name=model_name,
                    mime="application/octet-stream",
                    use_container_width=True,
                    type="primary",
                )
        
        # Option to navigate to other pages
        st.markdown("---")
        st.markdown("### 🔀 Next Steps")
        step_col1, step_col2, step_col3 = st.columns(3)
        with step_col1:
            st.page_link("streamlit_app.py", label="🎥 Test with Live Camera", icon="🎥")
        with step_col2:
            st.page_link("streamlit_app.py", label="🖼️ Try Image Analysis", icon="🖼️")
        with step_col3:
            st.page_link("streamlit_app.py", label="🧠 Inspect Model", icon="🧠")
    
    except Exception as e:
        status_placeholder.error(f"❌ Training failed: {e}")
        st.exception(e)
