# lagchain-pdf-rag

Projeto de estudo de busca semântica com LangChain e embeddings locais. O script carrega um índice vetorial já existente, compara consultas com os trechos indexados e exibe os resultados mais relevantes, incluindo metadados como a página do documento.

> **Escopo atual:** apesar do nome, a versão atual não lê nem divide arquivos PDF, não cria o índice vetorial e não usa um LLM para gerar respostas. Ela espera encontrar o índice pronto em `vector_store/react_store.json` e executa consultas de exemplo definidas em `main.py`.

## Requisitos

- Python 3.12 ou superior.
- Git para clonar o repositório.
- [uv](https://docs.astral.sh/uv/) para criar o ambiente e instalar as dependências a partir de `pyproject.toml` e `uv.lock`.
- Acesso à internet na primeira execução para baixar o modelo de embeddings `BAAI/bge-small-en-v1.5`, caso ele ainda não esteja em cache.

Não é necessária uma chave de API para as consultas implementadas atualmente: os embeddings são gerados localmente pelo FastEmbed.

## Baixar e executar

Clone o repositório e entre na pasta:

```bash
git clone https://github.com/rrguedes/lagchain-pdf-rag.git
cd lagchain-pdf-rag
```

Sincronize as dependências:

```bash
uv sync
```

Execute o script:

```bash
uv run python main.py
```

O comando deve ser executado na raiz do projeto, onde o caminho relativo `vector_store/react_store.json` pode ser encontrado. O modelo baixado pelo FastEmbed é armazenado no diretório `models`.

## O que o script demonstra

- Carregamento do índice vetorial em memória.
- Geração de embedding para uma consulta com FastEmbed.
- Busca por similaridade dos três trechos mais relevantes.
- Busca com filtro de metadados de página.
- Uso de um retriever do LangChain.

As consultas e o índice são definidos/selecionados no próprio código; para consultar outros documentos, é necessário preparar um índice compatível e ajustar a consulta em `main.py`.
