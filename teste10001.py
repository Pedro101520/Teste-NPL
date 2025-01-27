from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

# Caminho para o diretório onde o modelo foi baixado
model_path = "model"  # Substitua com o caminho correto

# Carregar o tokenizador e o modelo
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

comentarios = [
    "Eu não tou seguindo acessar o aplicativo",
    "Estou sem aplicado. Tem algum número WhatsApp que eu possa falar ?",
    "Que demais! Essa é a dupla vencedora.",
    "Eu quero entender pq o Bradesco bloqueou meus DOIS cartões , sendo que eu recebo meu salário por lá e pago TUDO em dia, ANTES mesmo do VENCIMENTO ! E agora como que irei USAR ? Sinceramente eu quero que resolvam esse problema AGORA",
    "Como eu consigo fazer o curso online da Bradesco?",
    "Boa noite,não consigo acessar o app,perdi meu cartão,já fui no auto atendimento e não consegui resolver,idem pelo telefone,preciso acessar minha conta",
    "Bradesco tinha ter um Cartão de crédito zero anuidade, mais zero mesmo.",
    "Oi,fazem dias que tento abrir conta pelo app e não consigo. Da um erro 404 mesmo tentando de outros aparelhos,e em horários distintos.",
    "Curiosa pra saber",
    "O comercial ficou maravilhoso! Parabéns!!!",
    "E aí Bradesco? Mais de 3 dias que paguei a fatura e nada do valor em limite! Vai ser preciso acionar a justiça para vocês agirem pelo correto?",
    "Não estou conseguindo acessar minha conta por NADA... ligo para a central de atendimento e desligam na minha cara e não resolvem o meu problema... achei que fosse um erro geral, mas pelo visto não. O aplicativo não tem nenhuma atualização para ser feita e já fiz de tudo, mas dá erro toda vez...",
    "Alguém mais está com problemas no app? Desde a última atualização só vive dando problemas.",
    "A mais de um mês venho tentando resolver um problema que tive junto ao banco, mas o @bradesco está tratando o caso com desprezo, descaso. Será a última vez que tento resolver extrajudicialmente. Se tiverem interesse em resolver esse transtorno, informei meus dados no direct.",
    "Telefone Fácil de vocês não está atendendo. A pessoa passar por uma urgência, perde tudo e ninguém atende.",
    "Pix não funcionando novamente",
    "Gostaria de alertar a todos os interessados que, ao lidarem com consórcios no Bradesco, é fundamental estar extremamente atento aos detalhes, pois a falta de orientação clara e a negligência de algumas gerentes pode resultar em prejuízos inesperados.",
    "Ajeitar esse app, não dá para ativar a chave de segurança pelo app, sempre fala que não é possível ativar a Chave de segurança. Pqp",
    "Antes a gente desbloqueava o cartão novo só fazendo uma ligação e agora não dá porque?",
    "Não sou cliente do Bradesco há anos, fiz portabilidade para o Itaú, mas o Bradesco me cobrou uma dívida de R$ 600, que foi reconhecida como indevida pela gerente. Mesmo assim, desde dezembro do ano passado, recebo 7 ligações diárias cobrando essa dívida. Não aguento mais esse constrangimento. Alguém do banco pode resolver?",
    "Em meus lançamentos futuros está um valor a ser descontado do qual não sei oq é desconta mês sim mês não,e outro de pserv que eu também não faço a mínima idéia do que seja,preciso que entrem em contato para que eu tenha uma solução,já foram descontados de outros meses...",
    "Pior banco do mundo roubou dinheiro meu e não quer devolver",
    "Legal mesmo é o seguro residencial que fizeram SEM AUTORIZAÇÃO na conta da minha mãe idosa! Faz alguns meses já tinham feito um seguro de vida e só cancelaram porque descobrimos, e não restituíram o valor que foi descontado! Se querem vender e bater meta que seja de forma honesta e profissional e não se aproveitando de pessoas idosas que ganham 1 salário mínimo! Parabéns Bradesco! 👏",
    "@bradesco eu tenho uma fatura pendente para janeiro, ja liguei e os atendentes informaram que essa fatura não tinha código de barras. O meu cartão é casas bahia o qual cancelei hoje, preciso de um parecer de vcs. Se foi gerado uma pendência em janeiro eu preciso do codigo de barras para pagar",
    "Vou ter que sair e emcerrar a minha conta no Bradesco pois todos os Pix que faço não são mais automáticos, só que o dinheiro é debitado instantaneamente da minha conta. Já liguei, mandei email, falei com gerente, NADA acontece. Não recomendo",
    "Minha senha foi cancelada pq eu tentei entrar no gov através do Bradesco! Que aplicativo ruim é esses que vocês têm? Mas, não tem problema. Amanhã vou na agência e cancelo minha conta 🙌",
    "O que está acontecendo com o Bradesco? Uma desorganização geral: primeiro a empresa não envia fatura para os clientes, nem por meio de DDA. Depois, negocia uma quitação total do cartão, recebe e agora quer incluir meu nome no Serasa. E pra fechar com chave de ouro, misteriosamente surgiu uma fatura que vence esse mês no meu DDA, do mesmo cartão que já foi quitado! E SIM, eu já entrei em contato com todos os meios de comunicação da Bradescard, inclusive Reclame Aqui! Estou esperando as cenas do próximo capítulo pra saber o que mais vcs vão inventar! Não dou dois anos pra esse banco quebrar!",
    "Tenho esse cartão a anos, utilizo em 7k por mês e nunca tive problemas. Passei por algumas dificuldades e acabei atrasando algumas faturas, mas que regularizei tudo em Dezembro, ou seja, se não pagava todo valor, esse valor constava na próxima fatura + juros e bla bla, poré, colocaram não sei quantos parcelamentos automáticos em 12x que somam mais 7k, mas nas faturas já constava o saldo remanescente + juros e ainda continuam me cobrando isso. Reclamei na central e falaram que seria estornado. Olhei a fatura agora e consta estorno, porém no mesmo valor de um novo lançamento que colocaram. Não aguento mais essa patifaria, um monte de lançamento que nos confundem todo. Quero urgente uma resolução, ou cancelarei tudo, cartão, conta, tudo.",
    "Não sou cliente do Bradesco há anos, fiz portabilidade para o Itaú, mas o Bradesco me cobrou uma dívida de R$ 600, que foi reconhecida como indevida pela gerente. Mesmo assim, desde dezembro do ano passado, recebo 7 ligações diárias cobrando essa dívida. Não aguento mais esse constrangimento. Alguém do banco pode resolver?",
    "Pior banco do mundo app não abre nem que dê a mulesta",
    "Lindaaaaa!!!",
    "Preciso de ajuda sofri um golpe e já pedi a devolução do que o Bradesco não respondeu nada ainda tenho provas que sofri um golpe preciso de ajuda Bradesco ❤️",
    "Boa tarde",
    "Pois é meu amor!!! Estou relax!!! Um novo Boninho!!! Me conta o que vem por ai…",
    "Ahhhhhhh, tô sabendo…. Rs",
    "Mais de 15 dias pra liberar consórcio contemplado e nada ainda, se Deus quiser bradesco nunca mais",
    "Rapaz, Big Boss Bradesco",
    "Lá vem coisa aí!!!!",
    "Iiih! Ansioso!!",
    "Expectativas foram criadas",
    "Sou cliente do Bradesco e estou super animada e ansiosa pelas dicas para facilitar minha vida",
    "AAA já quero ver o que ele vai aprontar!",
    "Amei, gente! Tô aqui só de olho no que vem por aí haha",
    "Surpresas, curiosos estamos. @aanafurtado",
    "Esse casal està demais. Que bacana a gnt ver eles nessa alegria toda e passando Boas Energias. Se é Boninho, tinha que ter algo de Bonzinho",
    "cadê o Boninho",
    "Ansiosaaaaaa",
    "Ana, Boninho e Bradesco. Que encontro",
    "Contaí Boninho!!!",
    "Tudo Bem, Pessoal?",
    "Entrei na minha conta hoje e roubaram 240,70 da minha conta devolvam meu dinheiro",
    "A gente é Bra de Bradesco",
    "A melhor pra propaganda do Bradesco!",
    "Veeeem muito aí!!",
    "Coisa boa vem por aiii",
    "Ansioso!!!",
    "Oi Bradesco, se eu movimentar 800 reais na minha conta por mês eu vou ser taxado??",
    "Ana é luz",
    "Uma volta à velha propaganda criativa do Bradesco, agora com um toque de redes sociais. Sempre uma boa sacada.",
    "Eu já fui Bradesco a vida toda, mas hoje me sinto frustrada e quero mais é encerrar minha conta. Só não faço isso ainda por conta do tempo, trabalho de perda de saldo de conta, entre outros. São absurdos o que fazem comigo. Não repasso o dinheiro nas datas, e esse mês veio cobrando juros do mesmo valor por 6 meses consecutivos. Não dou mais.",
    "Nosso diretor está sendo super atencioso, espero que ajude mais! Recebi feedback bom!"
]
# Classes do modelo
labels = ['negative', 'neutral', 'positive']

# Definir um limiar para ajustar as classificações
threshold = 0.32  # Limite de probabilidade

# Processar os comentários
resultados = []
for comentario in comentarios:
    # Tokenização
    inputs = tokenizer(comentario, return_tensors="pt", truncation=True, padding=True)

    # Fazer a previsão
    with torch.no_grad():
        outputs = model(**inputs)

    # Obter probabilidades
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Verificar se a probabilidade máxima está acima do limiar
    max_prob = probs.max().item()
    if max_prob < threshold:
        # Se a probabilidade máxima for menor que o limiar, classifique como neutro
        sentimento = 'neutral'
    else:
        # Caso contrário, pegue a classificação normal
        sentimento = labels[probs.argmax().item()]
    
    resultados.append({"Comentário": comentario, "Sentimento": sentimento, "P": max_prob})

# Criar um DataFrame
df = pd.DataFrame(resultados)

# Salvar em um arquivo CSV
df.to_csv("avaliacoes_sentimentos1.csv", index=False, encoding="utf-8-sig")

print("Arquivo 'avaliacoes_sentimentos.csv' criado com sucesso!")
