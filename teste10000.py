import requests

def baixar_arquivo(nome_arquivo):
    url = f"https://huggingface.co/citizenlab/twitter-xlm-roberta-base-sentiment-finetunned/resolve/main/{nome_arquivo}"
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        with open(nome_arquivo, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"{nome_arquivo} baixado com sucesso!")
    else:
        print(f"Erro ao baixar {nome_arquivo}: {response.status_code}")

arquivos = ["config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json"]

for arquivo in arquivos:
    baixar_arquivo(arquivo)
