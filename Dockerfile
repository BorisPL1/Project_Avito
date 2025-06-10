FROM python:3.12

WORKDIR /app

COPY requirements.txt model_class.py project_bk.py ./
RUN pip install -r requirements.txt
RUN ["model_class.py"]

CMD [ "python", "./project_bk.py" ]