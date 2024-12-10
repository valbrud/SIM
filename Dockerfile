FROM python:3.8

WORKDIR /SIM

COPY . /SIM

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "input_parser.py"]
