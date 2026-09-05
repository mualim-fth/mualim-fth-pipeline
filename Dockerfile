FROM tensorflow/serving:latest
ENV MODEL_NAME=churn-model
ENV PORT=8501
COPY ./serving_model/mualim-fth-pipeline /models/churn-model
CMD tensorflow_model_server --rest_api_port=${PORT} --model_name=${MODEL_NAME} --model_base_path=/models/churn-model