FROM python:3.10-slim

WORKDIR /app

# Instalar dependências do sistema (necessário para algumas libs de ML)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o código de monitorização
COPY monitor.py .

# O comando de arranque (o script tem o loop de agendamento)
CMD ["python", "monitor.py"]
