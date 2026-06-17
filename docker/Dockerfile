FROM ghcr.io/ggerganov/llama.cpp:server

# This assumes you will put the downloaded GGUF file in a 'models' folder at the root
COPY models/model-q4_k_m.gguf /models/model.gguf

EXPOSE 8080

ENTRYPOINT ["/llama-server", "-m", "/models/model.gguf", "--host", "0.0.0.0", "--port", "8080", "-c", "2048"]