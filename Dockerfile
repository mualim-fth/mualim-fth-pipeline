FROM tensorflow/serving:latest
ENV MODEL_NAME=churn-model
ENV MODEL_BASE_PATH=/models
ENV PORT=8501

COPY ./serving_model/mualim-fth-pipeline /models/churn-model
COPY ./monitoring_config.txt /models/monitoring_config.txt

CMD ["--monitoring_config_file=/models/monitoring_config.txt"]