FROM tensorflow/serving:latest
ENV MODEL_NAME=churn-model
ENV PORT=8501

# Salin model dan file konfigurasi metrik
COPY ./serving_model/mualim-fth-pipeline /models/churn-model
COPY ./monitoring_config.txt /models/monitoring_config.txt

# Tambahkan parameter --monitoring_config_file
CMD tensorflow_model_server --rest_api_port=${PORT} --model_name=${MODEL_NAME} --model_base_path=/models/churn-model --monitoring_config_file=/models/monitoring_config.txt