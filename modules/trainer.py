import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
import os

LABEL_KEY = 'Churn'
def transformed_name(key): return key + '_xf'

def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=64):
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY)
    )
    return dataset

def build_model(hp):
    NUMERIC_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']
    CATEGORICAL_FEATURES = {
        'gender': 2, 'SeniorCitizen': 2, 'Partner': 2,
        'StreamingTV': 3, 'PhoneService': 2, 'InternetService': 3,
        'PaperlessBilling': 2
    }
    
    input_features = []
    for key in NUMERIC_FEATURES:
        input_features.append(tf.keras.Input(shape=(1,), name=transformed_name(key), dtype=tf.float32))
    for key in CATEGORICAL_FEATURES.keys():
        input_features.append(tf.keras.Input(shape=(1,), name=transformed_name(key), dtype=tf.int64))
        
    concatenated_inputs = tf.keras.layers.Concatenate()([tf.cast(inp, tf.float32) for inp in input_features])
    
    dense_units = hp.get('units') if hp else 32
    dropout_rate = hp.get('dropout_rate') if hp else 0.2
    learning_rate = hp.get('learning_rate') if hp else 1e-3

    x = tf.keras.layers.Dense(dense_units, activation='relu')(concatenated_inputs)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=input_features, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def _get_serve_tf_examples_fn(model, tf_transform_output):
    model.tft_layer = tf_transform_output.transform_features_layer()
    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)
    return serve_tf_examples_fn

def run_fn(fn_args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    train_dataset = input_fn(fn_args.train_files, tf_transform_output, num_epochs=10)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=10)
    
    hp = fn_args.hyperparameters.get('values') if fn_args.hyperparameters else None
    model = build_model(hp)
    
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        epochs=10
    )
    
    signatures = {
        'serving_default': _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
        )
    }
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
