import pandas as pd
from deep_translator import GoogleTranslator

# Configuração
arquivo_entrada = "comentarios.xlsx"  # Substitua pelo nome do seu arquivo
coluna_texto = "comentario"  # Nome da coluna a ser traduzida
idioma_origem = "pt"
idioma_destino = "en"
arquivo_saida = "comentarios_traduzidos.xlsx"

# Carregar o arquivo
if arquivo_entrada.endswith(".csv"):
    df = pd.read_csv(arquivo_entrada)
elif arquivo_entrada.endswith(".xlsx"):
    df = pd.read_excel(arquivo_entrada)
else:
    raise ValueError("Formato de arquivo não suportado. Use CSV ou Excel.")

# Verificar se a coluna existe
if coluna_texto not in df.columns:
    raise ValueError(f"A coluna '{coluna_texto}' não foi encontrada no arquivo.")

# Traduzir os comentários
tradutor = GoogleTranslator(source=idioma_origem, target=idioma_destino)
df["traducao"] = df[coluna_texto].astype(str).apply(lambda x: tradutor.translate(x) if x.strip() else "")

# Salvar o novo arquivo
df.to_excel(arquivo_saida, index=False)
print(f"Tradução concluída! Arquivo salvo como {arquivo_saida}")
