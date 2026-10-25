# Transformer

В этом репозитории реализован GPT в оригинальном виде, как это было в статье [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
При реализации активно использовалась информация из книги С. Рашки "Строим LLM с нуля"

GPT реализован в виде python пакета. Для просмотра можно посмотреть примеры юпитер-ноутбуков ниже, либо подключить пакет напрямую

ТОКЕНИЗАТОР: при данной реализации текенизатор не реализован самостоятельно, а взять tiktoken для gpt2

ВАЙБКОДИНГ: некоторые фрагменты кода были написаны нейросетью, некоторые были взяты из книги Рашки. Но не один фрагмент не был вставлен без понимания. Все комментарии были написаны человеком
## 1. Запуск модели.ipynb
В этом ноутбуке можно запустить модель из python пакета, загрузив в нее веса из .pth файла

## 2. Обучение модели.ipynb
В этом ноутбуке можно обучить модель на текстах из датасета wikitext-103(пока что он захардкожен) и выгрузить веса в .pth файл

## 3. Использование Python пакета transformer_lab

GPT реализован в виде python пакета.

### Установка

```bash
pip install git+https://github.com/tereshkovyv/transformer_lab
```

### Использование

```python
from transformer_lab.model.gpt_model import GPTModel
from transformer_lab.model.generate import generate
from transformer_lab.training.data import tokenize_split, TokenDataset
from transformer_lab.training.train import train_model_simple
```

## 3. Состав модуля

### model

**attenеion.py** - блок внимания

**feed_forward.py** - блок feed forward

**layer_norm.py** - слой нормализации

**transformer_block.py** - весь блок трансформера

**gpt_model.py** - сама GPT модель

**generate.py** - методы генерации

### training
**data.py** - в нем есть класс TokenDataset - который можно просто брать и использовать(там внутри датасет **wikitext-103**)
**train.py** - код обучения модели. Взят из Raschka, ch. 5.2.


### Тесты

Тесты были завайбкожены. Они созданы для того, чтобы убедиться, чтобы все, что мы сделали - ничего не ломает. `python tests/run_all.py`.
