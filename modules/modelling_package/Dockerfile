FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

COPY modelling_package/requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY modelling_package /modelling_package
RUN pip install /modelling_package

RUN mkdir /models 
COPY scripts/main_modelling.py main.py
CMD ["python", "main.py"]