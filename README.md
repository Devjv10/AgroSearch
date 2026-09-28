# AgroSearch

Aplicação didática de Recuperação de Informação com Python e Streamlit, usando uma pequena base de documentos sobre agricultura.

## Funcionalidades

- Tokenização, normalização, remoção de stopwords e stemming.
- Visualização do vocabulário e do índice invertido.
- Busca com ranking TF-IDF e similaridade de cosseno.
- Exibição dos detalhes dos cálculos.

## Como executar

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run AgroSearch.py
```

No Windows, ative o ambiente com `.venv\Scripts\activate`.

Experimente a consulta `irrigação soja`.

## Relatório

[Relatório do projeto](Relatorio_AgroSearch_Joao_Victor.pdf)
