FROM python:3
LABEL project="Hippo Survey"

WORKDIR /app
COPY requirements.txt /app/
COPY ./src /app/src/

RUN pip install -r requirements.txt

RUN chmod a+x src/run.py

CMD ["./src/run.py"]