# service-llm

# Docker

docker build --build-arg PIP_CACHE_DIR=/pip-cache -t my-python-api .

docker build -t my-python-api .

docker run --gpus all -p 5001:5000 -v D:/Projects/test-angular/Llama-2-7b-chat-hf:/app/models my-python-api

curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"text": "Hello, how are you?"}'



## Utils
https://www.datacamp.com/fr/tutorial/fine-tuning-llama-2

https://huggingface.co/NousResearch/Llama-2-7b-chat-hf