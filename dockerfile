ARG TARGETPLATFORM=linux/x86_64
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt ./

RUN apt-get update
RUN apt-get install -y tesseract-ocr libtesseract-dev tesseract-ocr-eng tesseract-ocr-hin
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_md

# Copy the rest of the application code
COPY . .

# Command to run the Pathway script
CMD ["sh", "-c", "python -u ./test.py && python -u ./KYC.py"]