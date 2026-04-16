# Documentação Pessoal do Projeto

Este arquivo foi escrito para servir como uma documentação de estudo e consulta rápida para mim mesmo. A ideia aqui não é fazer uma documentação curta para GitHub, e sim um guia completo em português explicando o que cada parte do projeto faz, por que foi feita desse jeito e como alterar o comportamento do treino sem me perder.

---

## 1. Objetivo do projeto

O objetivo deste projeto é classificar lesões de pele em múltiplas classes usando o dataset HAM10000, aproveitando GPU local e transfer learning com PyTorch.

Neste projeto, o script principal é:

- `train_skin_lesion_classifier.py`

Ele faz, em ordem:

1. configura o dispositivo de execução
2. prepara os dados e confere arquivos
3. gera gráficos iniciais do dataset
4. divide os dados em treino, validação e teste
5. cria os `DataLoaders`
6. monta o modelo EfficientNet
7. treina em duas etapas
8. avalia no conjunto de teste
9. salva gráficos e um exemplo de Grad-CAM

---

## 2. Estrutura das pastas

As pastas mais importantes são:

- `all_images/`
Contém as imagens extraídas do HAM10000.

- `Saved_Model/`
Contém os modelos treinados salvos em checkpoint.

- `plots/`
Contém gráficos gerados automaticamente pelo script durante treino e avaliação.

- `Resultados/`
É uma pasta organizada para reunir os principais arquivos finais de resultado.

- `venv_skincancer/`
É o ambiente virtual com as bibliotecas do projeto.

---

## 3. Arquivo principal e visão geral do fluxo

O arquivo `train_skin_lesion_classifier.py` foi pensado para rodar localmente na GTX 1650.

O fluxo geral do `main()` é:

1. `set_seed(SEED)`
Deixa o comportamento mais reprodutível.

2. `configure_device()`
Detecta se há CUDA disponível e imprime as configurações de GPU.

3. `prepare_data()`
Confere se metadata e imagens existem, extrai zips se necessário e monta um `DataFrame` com caminho da imagem, nome da classe e índice numérico.

4. `visualize_frequency(df)` e `visualize_samples(df)`
Gera gráficos do dataset.

5. `train_test_split(...)`
Divide o dataset em treino, validação e teste com estratificação por classe.

6. `create_dataloaders(...)`
Cria os `DataLoaders` do PyTorch.

7. `build_model(device)`
Monta o modelo base e troca a cabeça final.

8. `create_criterion(...)`
Cria a função de perda.

9. `train_stage(...)`
Treina em duas fases:
- treino da cabeça
- fine-tuning parcial do backbone

10. `run_epoch(..., training=False, tta_passes=TTA_PASSES)`
Faz a avaliação final usando TTA.

11. `plot_confusion_matrix(...)`, curva ROC e `generate_grad_cam(...)`
Cria os artefatos finais de interpretação e análise.

---

## 4. Configurações do topo do script

No começo do arquivo existe a seção:

```python
# ================= CONFIGURATION =================
```

Essa é a parte mais importante para ajustar comportamento sem precisar mexer na lógica interna.

### 4.1 Caminhos principais

- `MAIN_ZIP_NAME`
Nome do zip principal do dataset.

- `BASE_PATH`
Pega a pasta atual do projeto.

- `IMAGE_DIR`
Pasta onde ficam as imagens extraídas.

- `SAVE_DIR`
Pasta onde os checkpoints do modelo são salvos.

- `PLOTS_DIR`
Pasta onde os gráficos gerados são salvos.

### 4.2 Configurações do modelo e treino

- `MODEL_NAME = "efficientnet_b2"`
Escolhe o backbone principal.

Atualmente o script aceita:
- `efficientnet_b0`
- `efficientnet_b2`

Na prática:
- `b0` é mais leve
- `b2` é mais forte, mas consome mais memória

- `BATCH_SIZE = 4`
Quantidade de imagens processadas de uma vez.

Como a GTX 1650 tem pouca VRAM, o batch foi reduzido para 4.

- `GRADIENT_ACCUMULATION_STEPS = 2`
Faz acumulação de gradiente para simular um batch efetivo maior.

Nesse caso:
- batch real = 4
- batch efetivo = 8

- `HEAD_EPOCHS = 5`
Quantidade de épocas para treinar só a cabeça do modelo.

- `FINETUNE_EPOCHS = 20`
Quantidade máxima de épocas no fine-tuning.

- `IMG_HEIGHT = 260` e `IMG_WIDTH = 260`
Tamanho das imagens de entrada.

- `HEAD_LEARNING_RATE = 3e-4`
Taxa de aprendizado da primeira fase.

- `FINETUNE_LEARNING_RATE = 1e-5`
Taxa de aprendizado menor para fine-tuning.

- `WEIGHT_DECAY = 1e-4`
Regularização do otimizador.

- `NUM_WORKERS = 2`
Número de subprocessos para carregar dados.

Se o PC travar ou der bug no Windows, uma alternativa é baixar para `0`.

- `UNFREEZE_LAST_BLOCKS = 3`
Quantidade de blocos finais do backbone que são destravados no fine-tuning.

### 4.3 Controle de estabilidade

- `SEED = 42`
Ajuda a reproduzir o experimento.

- `EARLY_STOPPING_PATIENCE = 6`
Número de épocas sem melhora relevante antes de interromper.

- `EARLY_STOPPING_MIN_DELTA = 1e-4`
Melhora mínima exigida para contar como avanço real.

- `USE_AMP = False`
Mixed precision desligado por estabilidade numérica.

Na GTX 1650, isso foi deixado desligado porque apareceram `NaN` em testes anteriores.

- `GRAD_CLIP_NORM = 1.0`
Limite do gradient clipping.

Isso ajuda a evitar explosão de gradiente.

### 4.4 Controle de loss e avaliação

- `USE_CLASS_WEIGHTS = False`
Desligado para favorecer `accuracy` geral.

Quando pesos de classe ficam muito agressivos:
- melhora classes raras
- mas pode derrubar accuracy global

- `LABEL_SMOOTHING = 0.05`
Suaviza a loss para reduzir overconfidence.

- `TTA_PASSES = 4`
Número de versões da imagem usadas na avaliação com TTA.

### 4.5 Saídas visuais

- `SHOW_PLOTS = False`
Não abre janelas na tela.

- `SAVE_PLOTS = True`
Salva gráficos em disco.

- `RUN_GRAD_CAM = True`
Gera um exemplo de Grad-CAM ao final.

---

## 5. Mapeamento das classes

O dicionário:

```python
LESION_TYPE_DICT = {...}
```

traduz os códigos do HAM10000 em nomes legíveis.

Exemplo:

- `nv` -> `Melanocytic nevi`
- `mel` -> `Melanoma`
- `bcc` -> `Basal cell carcinoma`

Depois disso:

- `CLASS_NAMES = list(LESION_TYPE_DICT.values())`

gera a lista com a ordem oficial das classes usadas no projeto.

Isso é importante porque:
- o índice da classe precisa ser consistente
- o modelo aprende por índice numérico
- o relatório final volta a converter para nomes humanos

---

## 6. Funções iniciais do script

### 6.1 `set_seed(seed)`

Essa função fixa sementes em:

- `random`
- `numpy`
- `torch`
- `torch.cuda`

Objetivo:
- reduzir variação aleatória entre execuções

Observação:
- não garante repetição perfeita de tudo
- mas ajuda bastante

### 6.2 `configure_device()`

Essa função detecta:

- se CUDA está disponível
- nome da GPU
- memória total

Também imprime no console:

- modelo
- resolução
- batch
- acumulação de gradiente
- se AMP está ligado ou não

Além disso:

```python
torch.backends.cudnn.benchmark = True
```

Isso acelera a execução quando o tamanho das entradas é estável.

---

## 7. Preparação dos dados

### 7.1 `ensure_dirs()`

Garante que as pastas necessárias existam:

- `IMAGE_DIR`
- `SAVE_DIR`
- `PLOTS_DIR`

### 7.2 `prepare_data()`

Essa função é uma das mais importantes do projeto.

Ela faz:

1. chama `ensure_dirs()`
2. verifica se a metadata já existe
3. se não existir, tenta extrair do zip principal
4. verifica quantas imagens já existem
5. se estiver faltando muita imagem, extrai os zips internos
6. lê o arquivo de metadata
7. detecta a extensão da imagem (`.jpg`, por exemplo)
8. cria a coluna `path`
9. cria a coluna `cell_type`
10. remove linhas com classe nula
11. remove linhas cujo arquivo de imagem não existe
12. cria `class_to_idx`
13. cria a coluna `label`

No final ela retorna:

- `df`
- `class_to_idx`

### 7.3 Colunas principais do DataFrame

Depois da preparação, o `df` importante contém:

- `image_id`
- `dx`
- `path`
- `cell_type`
- `label`

Onde:

- `path` é o caminho completo da imagem
- `cell_type` é o nome legível da classe
- `label` é o índice inteiro da classe

---

## 8. Visualização inicial dos dados

### 8.1 `visualize_frequency(df)`

Gera um gráfico horizontal de frequência por classe.

Serve para:
- enxergar desbalanceamento
- confirmar que o dataset foi lido corretamente

### 8.2 `visualize_samples(df)`

Mostra uma imagem exemplo por classe.

Serve para:
- checar se as imagens estão carregando
- ter uma noção visual das classes

Os gráficos são salvos em:

- `plots/frequency_plot.png`
- `plots/sample_images.png`

---

## 9. Classe personalizada do dataset

### `SkinCancerDataset(Dataset)`

Essa classe adapta o `DataFrame` para o formato esperado pelo PyTorch.

Ela implementa:

- `__len__()`
Retorna o tamanho do dataset.

- `__getitem__(index)`
Retorna:
- a imagem transformada
- o label da imagem

Fluxo interno:

1. pega a linha do DataFrame
2. abre a imagem com `PIL`
3. converte para RGB
4. aplica transformações
5. retorna `(image, label)`

---

## 10. Transformações das imagens

### `create_transforms()`

Retorna dois pipelines:

- `train_transform`
- `eval_transform`

### 10.1 Transformações de treino

No treino são usadas:

- `RandomResizedCrop`
- `RandomHorizontalFlip`
- `RandomVerticalFlip`
- `RandomRotation`
- `RandomAffine`
- `ColorJitter`
- `ToTensor`
- `Normalize`

Objetivo:
- gerar variações plausíveis
- reduzir overfitting
- melhorar generalização

### 10.2 Transformações de validação/teste

Na avaliação, o pipeline é mais simples:

- `Resize`
- `ToTensor`
- `Normalize`

Aqui a ideia é:
- não distorcer demais a imagem
- manter comparação consistente

### 10.3 Normalização

Foi usada a normalização padrão do ImageNet:

- média: `[0.485, 0.456, 0.406]`
- desvio: `[0.229, 0.224, 0.225]`

Isso é importante porque o backbone veio pré-treinado no ImageNet.

---

## 11. DataLoaders

### `create_dataloaders(train_df, val_df, test_df, device)`

Essa função cria:

- `train_loader`
- `val_loader`
- `test_loader`

Parâmetros importantes:

- `batch_size = BATCH_SIZE`
- `num_workers = NUM_WORKERS`
- `pin_memory = use_cuda`

`pin_memory=True` ajuda transferência CPU -> GPU quando CUDA está ativa.

---

## 12. Construção do modelo

### 12.1 `get_model_builder()`

Escolhe a função correta do `torchvision.models` conforme `MODEL_NAME`.

Hoje:

- `efficientnet_b0`
- `efficientnet_b2`

### 12.2 `build_model(device)`

Essa função:

1. escolhe o construtor do modelo
2. tenta carregar pesos pré-treinados
3. substitui a cabeça final
4. manda o modelo para GPU ou CPU

### 12.3 Cabeça final

A cabeça foi substituída por:

- `Dropout(0.35)`
- `Linear(in_features, 256)`
- `ReLU()`
- `Dropout(0.25)`
- `Linear(256, len(CLASS_NAMES))`

Motivo:
- adaptar o backbone ao número de classes do HAM10000
- aumentar capacidade da cabeça
- adicionar regularização

---

## 13. Congelar e descongelar camadas

### 13.1 `freeze_feature_extractor(model)`

Congela todo o backbone:

- `model.features.parameters() -> requires_grad = False`

E mantém treinável:

- `model.classifier.parameters() -> requires_grad = True`

Uso:
- primeira fase do treino

### 13.2 `unfreeze_last_blocks(model, blocks_to_unfreeze)`

Primeiro congela tudo com `freeze_feature_extractor(model)`.

Depois:
- pega os blocos finais de `model.features`
- destrava os últimos `blocks_to_unfreeze`

Uso:
- segunda fase do treino

Motivo:
- fazer fine-tuning só da parte mais alta da rede
- melhorar performance sem estourar custo computacional

---

## 14. Loss e pesos de classe

### `build_class_weights(train_df, device)`

Cria pesos baseados na frequência das classes, mas usando raiz quadrada para suavizar:

```python
weights = 1.0 / np.sqrt(freq)
```

### `create_criterion(train_df, device)`

Se `USE_CLASS_WEIGHTS=True`, usa pesos.

Se `False`, não usa pesos.

Também aplica:

- `label_smoothing=LABEL_SMOOTHING`

Loss usada:

- `nn.CrossEntropyLoss(...)`

No estado atual, para maximizar accuracy geral:

- `USE_CLASS_WEIGHTS = False`

---

## 15. TTA na avaliação

### 15.1 `build_tta_variants(images, passes)`

Cria variações da mesma imagem:

- original
- flip horizontal
- flip vertical
- flip horizontal + vertical

### 15.2 `predict_probabilities(model, images, tta_passes=1)`

Passa as variações no modelo, soma as probabilidades e faz média.

Objetivo:
- reduzir sensibilidade a orientação
- melhorar a estabilidade da predição final

---

## 16. Loop principal de época

### `run_epoch(...)`

Essa função serve tanto para treino quanto para avaliação.

Ela:

1. define modo `train()` ou `eval()`
2. percorre os batches
3. envia imagens/labels para o device
4. roda forward
5. calcula loss
6. faz backward e update se estiver em treino
7. calcula probabilidades e predições
8. acumula métricas
9. retorna loss, accuracy, probabilidades, preds e targets

### 16.1 Proteções numéricas

O código usa:

- `torch.nan_to_num(...)`
- verificação de `torch.isfinite(loss)`
- `clip_grad_norm_`

Essas proteções foram adicionadas porque durante testes anteriores apareceram `NaN` em loss e ROC.

### 16.2 Acumulação de gradiente

No treino:

```python
loss_to_backprop = loss / GRADIENT_ACCUMULATION_STEPS
```

Depois o otimizador só faz `step()` quando chega no número certo de mini-batches.

Isso permite:
- reduzir uso de VRAM
- simular batch maior

---

## 17. Salvamento e carregamento do modelo

### `save_checkpoint(...)`

Salva:

- época
- pesos do modelo
- estado do otimizador
- melhor val_acc
- nomes das classes
- tamanho da imagem
- nome da fase
- nome do modelo

### `load_checkpoint(...)`

Carrega os pesos salvos no modelo.

Arquivo salvo em:

- `Saved_Model/best_skin_lesion_<modelo>_1650.pth`

No treino atual:

- `Saved_Model/best_skin_lesion_efficientnet_b2_1650.pth`

---

## 18. Grad-CAM

### 18.1 `denormalize_image(tensor)`

Desfaz a normalização do ImageNet para a imagem ficar visualizável de novo.

### 18.2 `generate_grad_cam(...)`

Essa função:

1. coloca o modelo em `eval()`
2. registra hooks no último bloco de features
3. faz forward da imagem
4. encontra a classe predita
5. faz backward dessa classe
6. coleta ativações e gradientes
7. monta o mapa de calor
8. redimensiona e sobrepõe na imagem
9. salva em `plots/grad_cam_example.png`

Objetivo:
- entender visualmente onde o modelo está olhando

---

## 19. Treino em duas etapas

### `train_stage(...)`

Essa função executa uma fase completa de treino.

Ela:

1. cria o otimizador
2. cria scheduler
3. cria scaler
4. roda épocas
5. salva melhor checkpoint
6. controla early stopping
7. atualiza o histórico

### 19.1 Fase 1: `head_training`

No `main()`, primeiro o backbone é congelado:

```python
freeze_feature_extractor(model)
```

Depois roda:

- `HEAD_EPOCHS`
- `HEAD_LEARNING_RATE`

Objetivo:
- treinar apenas a cabeça nova
- preservar conhecimento do backbone pré-treinado

### 19.2 Fase 2: `fine_tuning`

Depois:

1. carrega o melhor checkpoint da fase anterior
2. destrava os últimos blocos com `unfreeze_last_blocks`
3. roda nova fase com `FINETUNE_LEARNING_RATE`

Objetivo:
- adaptar parte do backbone ao dataset
- melhorar accuracy final

---

## 20. Histórico de treino

### `plot_training_history(history)`

Gera:

- curva de loss
- curva de accuracy

Salva em:

- `plots/training_history.png`

O dicionário `history` guarda:

- fase
- época
- train_loss
- train_acc
- val_loss
- val_acc

---

## 21. O que acontece no `main()`, passo a passo

Aqui está a leitura prática do `main()`:

### Etapa 1

```python
set_seed(SEED)
```

Torna o experimento mais reprodutível.

### Etapa 2

```python
device = configure_device()
```

Detecta GPU e imprime as configs.

### Etapa 3

```python
df, class_to_idx = prepare_data()
```

Prepara o DataFrame principal.

### Etapa 4

```python
visualize_frequency(df)
visualize_samples(df)
```

Gera gráficos do dataset.

### Etapa 5

```python
train_df, test_df = train_test_split(...)
train_df, val_df = train_test_split(...)
```

Divide em treino, validação e teste.

### Etapa 6

```python
train_loader, val_loader, test_loader = create_dataloaders(...)
```

Cria os loaders.

### Etapa 7

```python
model = build_model(device)
criterion = create_criterion(train_df, device)
```

Monta o modelo e a função de perda.

### Etapa 8

Treino da cabeça:

```python
freeze_feature_extractor(model)
train_stage(..., stage_name="head_training")
```

### Etapa 9

Fine-tuning:

```python
load_checkpoint(...)
unfreeze_last_blocks(model, UNFREEZE_LAST_BLOCKS)
train_stage(..., stage_name="fine_tuning")
```

### Etapa 10

Gera o gráfico do histórico.

### Etapa 11

Carrega o melhor checkpoint e roda avaliação com TTA.

### Etapa 12

Imprime:

- test loss
- test accuracy
- classification report

### Etapa 13

Gera:

- matriz de confusão
- curva ROC
- Grad-CAM

---

## 22. Arquivos gerados ao final

Arquivos mais importantes:

- `Saved_Model/best_skin_lesion_efficientnet_b2_1650.pth`
- `plots/training_history.png`
- `plots/confusion_matrix.png`
- `plots/roc_curve.png`
- `plots/grad_cam_example.png`
- `plots/frequency_plot.png`
- `plots/sample_images.png`

Depois eles também podem ser organizados em:

- `Resultados/`

---

## 23. Resultado atual do projeto

No estado atual, o melhor resultado atingido foi:

- Melhor validação: `0.8215`
- Accuracy de teste: `0.8224`

Isso aconteceu com:

- `EfficientNet-B2`
- entrada `260x260`
- treino em duas fases
- fine-tuning parcial
- TTA

---

## 24. Como rodar

### 24.1 Ativar ambiente

```powershell
cd 'C:\Users\nicolas\OneDrive\Área de Trabalho\projeto fic'
.\venv_skincancer\Scripts\Activate.ps1
```

### 24.2 Rodar treino

```powershell
python .\train_skin_lesion_classifier.py
```

---

## 25. Como alterar o comportamento do projeto

### 25.1 Quero usar menos memória

Alterar:

- `BATCH_SIZE = 4` -> `2`
- `IMG_HEIGHT/IMG_WIDTH = 260` -> `224`
- `MODEL_NAME = "efficientnet_b2"` -> `"efficientnet_b0"`

### 25.2 Quero tentar mais accuracy

Possíveis ajustes:

- aumentar `FINETUNE_EPOCHS`
- aumentar `UNFREEZE_LAST_BLOCKS`
- testar `USE_CLASS_WEIGHTS = True`
- testar mais `TTA_PASSES`
- testar outro backbone

### 25.3 Quero treinar mais rápido

Possíveis ajustes:

- usar `efficientnet_b0`
- reduzir resolução
- reduzir número de épocas
- desligar geração de gráficos

### 25.4 Quero abrir os gráficos na tela

Alterar:

```python
SHOW_PLOTS = True
```

### 25.5 Não quero salvar gráficos

Alterar:

```python
SAVE_PLOTS = False
```

### 25.6 Não quero Grad-CAM

Alterar:

```python
RUN_GRAD_CAM = False
```

---

## 26. Possíveis erros e interpretação

### Erro: GPU não detectada

Se `torch.cuda.is_available()` retornar `False`, o script cai para CPU.

Conferir:

- driver NVIDIA
- CUDA
- versão do PyTorch com CUDA

### Erro: falta de memória

Sinais:

- crash
- `CUDA out of memory`

Soluções:

- diminuir `BATCH_SIZE`
- diminuir `IMG_HEIGHT/IMG_WIDTH`
- trocar para `efficientnet_b0`

### Erro: `NaN`

Esse projeto já teve esse problema antes.

Proteções atuais:

- `USE_AMP = False`
- `nan_to_num`
- gradient clipping
- ignorar batch com loss não finita

---

## 27. Interpretação das métricas

### Accuracy

Percentual total de acerto.

É a métrica principal que eu estava tentando maximizar.

### Precision

Entre o que o modelo previu como uma classe, quanto estava correto.

### Recall

Entre os exemplos reais da classe, quantos o modelo conseguiu encontrar.

### F1-score

Equilíbrio entre precision e recall.

### Matriz de confusão

Mostra onde o modelo mais confunde classes.

### ROC

Mostra a separação por classe com base nas probabilidades.

### Grad-CAM

Ajuda a interpretar visualmente em que região da imagem o modelo se baseou.

---

## 28. O que foi mais importante para subir a accuracy

Os pontos que mais ajudaram nesse projeto foram:

- trocar para `EfficientNet-B2`
- aumentar a resolução para `260x260`
- treino em duas etapas
- destravar parte do backbone
- usar TTA no teste
- não usar pesos de classe agressivos quando o alvo era accuracy global

---

## 29. Resumo mental rápido

Se eu quiser lembrar do projeto em 20 segundos:

- o script principal é `train_skin_lesion_classifier.py`
- ele treina uma EfficientNet-B2 em PyTorch
- usa HAM10000
- roda em GTX 1650
- treina em duas fases
- salva modelo e gráficos
- faz TTA no teste
- gera Grad-CAM
- melhor accuracy de teste atual: `82.24%`

---

## 30. Próximos passos possíveis

Se eu quiser continuar evoluindo este projeto no futuro, as melhores linhas são:

- melhorar o README com seção de reprodutibilidade
- criar inferência para imagem única
- criar script separado só para teste/inferência
- exportar resultados em CSV
- fazer comparação entre `efficientnet_b0` e `efficientnet_b2`
- testar `class weights` mais suaves
- salvar logs de treino em arquivo

---

## 31. Observação final para mim mesmo

Se eu parar de mexer nisso por um tempo, a ordem segura para retomar é:

1. abrir este arquivo
2. abrir `train_skin_lesion_classifier.py`
3. conferir parâmetros do topo
4. ativar `venv_skincancer`
5. rodar o script
6. olhar `Resultados/` e `plots/`

Se eu seguir isso, eu provavelmente volto ao contexto sem me enrolar.
