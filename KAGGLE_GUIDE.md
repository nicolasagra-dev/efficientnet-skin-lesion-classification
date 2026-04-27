# Como treinar no Kaggle

Este guia roda o treino do projeto em uma GPU do Kaggle, sem depender da sua GTX 1650 local.

## 1. Criar um Dataset privado no Kaggle

1. Entre em `https://www.kaggle.com/datasets`.
2. Clique em `New Dataset`.
3. Crie um dataset privado, por exemplo `skin-lesion-project-code`.
4. Envie estes arquivos/pastas do projeto:

```text
train_skin_lesion_classifier.py
kaggle_train.py
skin_lesion_classifier/
```

Nao envie:

```text
venv_skincancer/
Saved_Model/
plots/
all_images/
*.pth
```

## 2. Adicionar o dataset HAM10000

Voce precisa adicionar ao notebook um dataset que contenha:

```text
HAM10000_metadata.csv
HAM10000_images_part_1.zip
HAM10000_images_part_2.zip
```

Tambem funciona se o dataset ja tiver as imagens `.jpg` extraidas.

## 3. Criar o Notebook

1. Entre em `https://www.kaggle.com/code`.
2. Clique em `New Notebook`.
3. No painel direito, clique em `Add Input`.
4. Adicione:
   - o dataset privado com o codigo do projeto
   - o dataset HAM10000
5. Em `Settings`, mude `Accelerator` para `GPU`.

## 4. Copiar o codigo para a sessao

Na primeira celula do notebook, rode:

```python
import shutil
from pathlib import Path

working = Path("/kaggle/working/project")
working.mkdir(parents=True, exist_ok=True)

code_root = None
for candidate in Path("/kaggle/input").rglob("kaggle_train.py"):
    code_root = candidate.parent
    break

if code_root is None:
    raise FileNotFoundError("kaggle_train.py nao encontrado nos inputs.")

for name in ["kaggle_train.py", "train_skin_lesion_classifier.py", "skin_lesion_classifier"]:
    source = code_root / name
    destination = working / name
    if source.is_dir():
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)

print("Projeto copiado para", working)
```

## 5. Rodar o treino

Na segunda celula:

```python
%cd /kaggle/working/project
!python kaggle_train.py
```

## 6. Baixar resultados

Ao final, o Kaggle vai deixar estes arquivos em `/kaggle/working`:

```text
best_skin_lesion_efficientnet_b2_1650.pth
plots/
```

Baixe o `.pth` e coloque no app privado:

```text
C:\Users\nicolas\OneDrive\Área de Trabalho\skin-lesion-inference-app\models\
```

Depois reinicie o Streamlit.

## Observacoes

- O `kaggle_train.py` usa `batch_size=16` e `AMP=True`, porque a GPU do Kaggle costuma ter mais memoria que a GTX 1650.
- O treino salva o melhor checkpoint pelo criterio configurado no projeto.
- Se der erro de memoria, reduza `batch_size` em `kaggle_train.py` para `8`.
- Se a sessao cair, rode novamente; o Kaggle tem limite de tempo por sessao.
