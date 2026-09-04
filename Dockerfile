FROM tensorflow/serving:latest

# Variabel ini otomatis dibaca oleh sistem bawaan TF Serving
ENV MODEL_NAME=churn-model
ENV MODEL_BASE_PATH=/models
ENV PORT=8501

# Menyalin isi folder pipeline (yang berisi folder angka) tepat ke tempat yang dicari sistem
COPY ./serving_model/mualim-fth-pipeline /models/churn-model