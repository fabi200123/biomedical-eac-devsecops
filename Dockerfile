FROM python:3.11-slim
WORKDIR /app
COPY scripts/ scripts/
COPY configs/ configs/
RUN pip install --no-cache-dir pyyaml requests pytz python-dateutil kubernetes prometheus-client
CMD ["python","/app/scripts/carla_scheduler.py"]
