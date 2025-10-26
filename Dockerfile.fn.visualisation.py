# Use official Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /feature_engineering

# Install dependencies
RUN pip3 install poetry

COPY poetry.lock pyproject.toml ./


# Copy app files
COPY . .

RUN poetry config virtualenvs.in-project true

RUN poetry install --no-root


# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
ENTRYPOINT ["poetry", "run", "streamlit", "run", "feature_visualisation/feature_visualisation.py", "--server.port=8501", "--server.enableCORS=false"]

# ENTRYPOINT ["poetry", "run", "streamlit", "run", "eda_platform/main.py", "--server.port=8080", "--server.address=0.0.0.0"]
