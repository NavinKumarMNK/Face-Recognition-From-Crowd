# Use NVIDIA's PyTorch container as the base image
FROM nvcr.io/nvidia/pytorch:20.03-py3

# Set the working directory
WORKDIR /app

COPY requirements.txt .

RUN mkdir ./weights
RUN mkdir ./temp

WORKDIR /app/weights
RUN wget "https://drive.google.com/uc?export=download&id=FILE_ID" -O weights.zip
RUN unzip weights.zip
RUN rm weights.zip

WORKDIR /app

RUN pip install -r requirements.txt

COPY . .

EXPOSE 5005

CMD ["python", "run.py" "--source", "live"]