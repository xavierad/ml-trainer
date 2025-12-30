FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . ./modelling

RUN mkdir /models 
COPY main.py main.py
CMD ["python", "main.py"]