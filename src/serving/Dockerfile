WORKDIR /app

# Copy full src
COPY src/ .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose port 8080 to match your tests
EXPOSE 8080

# Start server on 0.0.0.0:8080
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8080
CMD ["python", "server.py"]