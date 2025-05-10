FROM python
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["gunicorn", "-t", "15", "-b", "0.0.0.0:5000", "app:app"]