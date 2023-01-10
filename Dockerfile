FROM python:3.9.6-slim-buster

COPY . /app
#COPY Pipfile* /app/
WORKDIR /app

RUN pip install --upgrade pip && \
    pip install pipenv && \
    pipenv install --system --deploy

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]