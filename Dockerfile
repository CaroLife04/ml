FROM python:3.11

WORKDIR /machine_learning_python

COPY . /machine_learning_python/

RUN pip install -r requirements.txt

CMD ["python", "r_l.py"]
