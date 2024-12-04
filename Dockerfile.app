FROM python:3.10-slim

# Set the working directory
WORKDIR /arewetoxic

# Copy the requirements file into the container
COPY requirements_app.txt .
RUN pip install --no-cache-dir -r requirements_app.txt

# Copy the source code into the container
COPY .env .env
COPY . .

# Expose the port that Streamlit will run on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]