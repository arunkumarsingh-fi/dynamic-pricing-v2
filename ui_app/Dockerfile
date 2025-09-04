FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Ensure data directory exists
RUN mkdir -p /app/data
EXPOSE 8502
# Copy progressive UI design
COPY ui_progressive.py .
CMD ["streamlit", "run", "ui_progressive_fixed.py", "--server.port=8503", "--server.address=0.0.0.0"]
