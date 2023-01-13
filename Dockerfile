FROM python:3.9.6-slim-buster

RUN pip install --upgrade pip
RUN pip install pipenv

COPY Pipfile* /app/
WORKDIR /app
RUN pipenv install --system --deploy

COPY . /app

WORKDIR /app

EXPOSE 8000 8501

#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# run both fastapi and streamlit
CMD ["sh", "-c", "streamlit run streamlit.py & uvicorn main:app --host 0.0.0.0 --port 8000"]



#run streamlit
# CMD ["streamlit", "run", "streamlit.py"]
