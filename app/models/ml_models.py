# app/models/ml_models.py
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, concatenate, Input
from tensorflow.keras.models import Model

# --- START STAGE 1 LOGIC ---

def create_stage1_preprocessor(data_path):
    """Creates and fits the preprocessor specifically for the Stage 1 model's data."""
    try:
        df = pd.read_csv(data_path)
        # These features must match EXACTLY what you used in your Stage 1 Colab notebook
        categorical_features = ['gender', 'smoking_status', 'alcohol_consumption', 'hpv_status']
        numerical_features = ['age', 'lesion_size_mm']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        preprocessor.fit(df)
        return preprocessor
    except Exception as e:
        print(f"--- FATAL ERROR in create_stage1_preprocessor: {e} ---")
        raise

def create_stage1_model(num_tabular_features):
    """Rebuilds the complete architecture for the Stage 1 multi-modal model."""
    # Image Branch
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    image_input = Input(shape=(224, 224, 3), name='image_input')
    x = base_model(image_input, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    image_features = Dense(128, activation='relu')(x)
    image_branch = Model(inputs=image_input, outputs=image_features, name='image_branch')

    # Tabular Branch
    tabular_input = Input(shape=(num_tabular_features,), name='tabular_input')
    y = Dense(64, activation='relu')(tabular_input)
    y = Dense(32, activation='relu')(y)
    tabular_features = Dense(16, activation='relu')(y)
    tabular_branch = Model(inputs=tabular_input, outputs=tabular_features, name='tabular_branch')

    # Head
    head_input = Input(shape=(128 + 16,), name='head_input')
    z = Dense(64, activation='relu')(head_input)
    z = Dropout(0.5)(z)
    output = Dense(2, activation='softmax', name='output')(z)
    head_branch = Model(inputs=head_input, outputs=output, name='head_branch')

    # Combine
    combined = concatenate([image_branch.output, tabular_branch.output])
    final_output = head_branch(combined)
    
    final_model = Model(inputs=[image_input, tabular_input], outputs=final_output)
    
    # Return all the pieces
    return final_model, image_branch, tabular_branch, head_branch

class ModelLoader:
    def __init__(self, weights_paths, preprocessor_data_path):
        self.model = None
        self.preprocessor = None
        self._load_resources(weights_paths, preprocessor_data_path)

    def _load_resources(self, weights_paths, preprocessor_data_path):
        try:
            print("--- Loading Stage 1 Resources ---")
            print(f"--- Creating Stage 1 preprocessor from: {preprocessor_data_path} ---")
            self.preprocessor = create_stage1_preprocessor(preprocessor_data_path)
            
            one_hot_encoder = self.preprocessor.named_transformers_['cat']
            num_cat_features = sum(len(cats) for cats in one_hot_encoder.categories_)
            num_num_features = len(self.preprocessor.named_transformers_['num'].get_feature_names_out())
            num_total_tabular_features = num_num_features + num_num_features
            
            print("--- Re-creating Stage 1 model architecture ---")
            final_model, image_b, tabular_b, head_b = create_stage1_model(num_total_tabular_features)

            print("--- Loading Stage 1 weights into branches ---")
            image_b.load_weights(weights_paths['image'])
            tabular_b.load_weights(weights_paths['tabular'])
            head_b.load_weights(weights_paths['head'])
            
            print("--- Compiling the final Stage 1 model ---")
            final_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.model = final_model
            print("--- Stage 1 Model and preprocessor loaded successfully! ---")

        except Exception as e:
            print(f"--- FATAL ERROR loading Stage 1 resources: {e} ---")
            # Clear the variables on failure
            self.model = None
            self.preprocessor = None
            raise # Re-raise the exception to make sure the server stops

# Create the single instance of the loader
model_loader = ModelLoader(
    weights_paths={
        'image': 'models_store/stage1/image_branch.weights.h5',
        'tabular': 'models_store/stage1/tabular_branch.weights.h5',
        'head': 'models_store/stage1/head_model.weights.h5'
    },
    preprocessor_data_path='models_store/stage1/training_data_for_preprocessor.csv'
)