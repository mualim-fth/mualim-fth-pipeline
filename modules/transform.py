import tensorflow as tf
import tensorflow_transform as tft

CATEGORICAL_FEATURES = {
    'gender': 2, 'SeniorCitizen': 2, 'Partner': 2,
    'StreamingTV': 3, 'PhoneService': 2, 'InternetService': 3,
    'PaperlessBilling': 2
}
NUMERIC_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']
LABEL_KEY = 'Churn'

def transformed_name(key):
    return key + '_xf'

def preprocessing_fn(inputs):
    outputs = {}
    
    for key in CATEGORICAL_FEATURES:
        dim = CATEGORICAL_FEATURES[key]
        int_value = tft.compute_and_apply_vocabulary(inputs[key], top_k=dim + 1)
        outputs[transformed_name(key)] = tf.cast(int_value, tf.int64)
        
    for key in NUMERIC_FEATURES:
        if inputs[key].dtype == tf.string:
            tensor_float = tf.strings.to_number(inputs[key], out_type=tf.float32)
            outputs[transformed_name(key)] = tft.scale_to_z_score(tensor_float)
        else:
            outputs[transformed_name(key)] = tft.scale_to_z_score(inputs[key])
            
    outputs[transformed_name(LABEL_KEY)] = tf.cast(
        tft.compute_and_apply_vocabulary(inputs[LABEL_KEY], top_k=2),
        tf.int64
    )
    
    return outputs
