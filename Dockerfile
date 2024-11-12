FROM python:3.10-slim

# Set the working directory
WORKDIR /arewetoxic

# Define a build argument for the credentials path
ARG GCP_SERVICE_ACCOUNT

# Set the environment variable for Google Cloud authentication
ENV GOOGLE_APPLICATION_CREDENTIALS=${GCP_SERVICE_ACCOUNT}

# Copy the service account key into the container
COPY ${GCP_SERVICE_ACCOUNT} ${GCP_SERVICE_ACCOUNT}

# Install required Python packages (including Google Cloud SDK)
RUN pip install --no-cache-dir google-cloud-storage google-cloud

# Copy the requirements file into the container
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the source code into the container
COPY . .

# Run the cleaning script
CMD ["python", "tests/cleaning.py"]