import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from typing import NamedTuple, Dict, Any
from tfx.components.trainer.fn_args_utils import FnArgs

LABEL_KEY = 'Churn'

def transformed_name(key):
    return key + '_xf'

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
    
    dense_units = hp.Int('units', min_value=16, max_value=64, step=16)
    x = tf.keras.layers.Dense(dense_units, activation='relu')(concatenated_inputs)
    x = tf.keras.layers.Dropout(hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1))(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model = tf.keras.Model(inputs=input_features, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

TunerFnResult = NamedTuple('TunerFnResult', [('tuner', Any), ('fit_kwargs', Dict[str, Any])])

def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_dataset = input_fn(fn_args.train_files, tf_transform_output, num_epochs=5)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=5)
    
    tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=3,
        directory=fn_args.working_dir,
        project_name='churn_kt'
    )
    
    fit_kwargs = {
        'x': train_dataset,
        'validation_data': eval_dataset,
        'steps_per_epoch': fn_args.train_steps,
        'validation_steps': fn_args.eval_steps
    }
    
    return TunerFnResult(tuner=tuner, fit_kwargs=fit_kwargs)
