# Use official Python image
FROM python:3.12

# Set working directory
# WORKDIR /feature_engineering

# Install dependencies
RUN pip3 install poetry

COPY poetry.lock pyproject.toml ./

RUN poetry config virtualenvs.in-project true
RUN poetry config installer.max-workers 10

RUN poetry install --no-root


# Copy app files
COPY . .


# WORKDIR /

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
#ENTRYPOINT ["ls"]
ENTRYPOINT ["poetry", "run", "streamlit", "run", "feature_engineering/feature_visualisation/fe_visualisation.py", "--server.port=8501", "--server.enableCORS=false"]
#ENTRYPOINT ["poetry", "run"]

# ENTRYPOINT ["poetry", "run", "streamlit", "run", "eda_platform/main.py", "--server.port=8080", "--server.address=0.0.0.0"]
