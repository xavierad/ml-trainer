FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY modelling_package /modelling_package
RUN uv pip install --system /modelling_package

RUN mkdir /models 
COPY scripts/main_modelling.py main.py
CMD ["python", "main.py"]