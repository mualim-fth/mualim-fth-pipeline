FROM tensorflow/serving:latest

COPY ./serving_model/mualim-fth-pipeline /models/churn-model

ENV MODEL_NAME=churn-model
ENV PORT=8501

CMD ["sh", "-c", "tensorflow_model_server --rest_api_port=${PORT} --model_name=${MODEL_NAME} --model_base_path=/models/${MODEL_NAME}"]