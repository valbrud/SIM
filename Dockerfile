FROM python:3.12

WORKDIR /SIM

COPY . /SIM

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "input_parser.py"]
