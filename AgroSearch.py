"""AgroSearch — exemplo didático de Recuperação de Informação com Streamlit.

Execute com: streamlit run AgroSearch.py
"""

import math
import re
import unicodedata
from collections import Counter, defaultdict

import pandas as pd
import streamlit as st


DOCUMENTOS = {
    "Doc 1": "A soja requer irrigação constante durante o período de floração para garantir a produtividade.",
    "Doc 2": "O controle biológico de lagartas na soja pode ser feito com a vespa Trichogramma.",
    "Doc 3": "A adubação verde com leguminosas melhora o nitrogênio no solo para o milho.",
    "Doc 4": "Lagartas desfolhadoras causam grande prejuízo na cultura da soja e do algodão.",
    "Doc 5": "A irrigação por gotejamento economiza água e é ideal para o cultivo orgânico.",
}

STOPWORDS = {
    "a", "o", "as", "os", "de", "da", "do", "das", "dos", "e", "em",
    "para", "por", "com", "na", "no", "nas", "nos", "um", "uma",
}

def tokenizar(texto):
    """Separa o texto em palavras; a pontuação não entra nos tokens."""
    return re.findall(r"\b[^\W\d_]+\b", texto, flags=re.UNICODE)


def normalizar_tokens(tokens):
    """Converte para minúsculas e retira marcas de acentuação."""
    resultado = []
    for token in tokens:
        sem_acentos = "".join(
            caractere
            for caractere in unicodedata.normalize("NFD", token.lower())
            if unicodedata.category(caractere) != "Mn"
        )
        resultado.append(sem_acentos)
    return resultado


def remover_stopwords(tokens):
    return [token for token in tokens if token not in STOPWORDS]


def aplicar_stemmer(token):
    """Stemmer rudimentar com as mesmas regras usadas na prática da disciplina."""
    if token.endswith("entos"):
        return token[:-4]
    if token.endswith("amente"):
        return token[:-5]
    if token.endswith("ados"):
        return token[:-3]
    return token


def processar_texto(texto, usar_stopwords=True, usar_stemming=True):
    """Executa o pipeline e devolve cada estágio para fins de visualização."""
    tokenizados = tokenizar(texto)
    normalizados = normalizar_tokens(tokenizados)
    sem_stopwords = (
        remover_stopwords(normalizados) if usar_stopwords else normalizados.copy()
    )
    stems = (
        [aplicar_stemmer(token) for token in sem_stopwords]
        if usar_stemming
        else sem_stopwords.copy()
    )
    return {
        "tokenizacao": tokenizados,
        "normalizacao": normalizados,
        "stopwords": sem_stopwords,
        "stemming": stems,
        "final": stems,
    }


def construir_indice(documentos_processados):
    """Cria manualmente o índice termo -> documentos, sem IDs repetidos."""
    indice = defaultdict(list)
    for doc_id, tokens in documentos_processados.items():
        for termo in sorted(set(tokens)):
            indice[termo].append(doc_id)
    return dict(sorted(indice.items()))


def calcular_ranking_tfidf(query_tokens, documentos_processados, indice):
    """Soma TF(t,d) * IDF(t) para cada ocorrência única na consulta."""
    quantidade_documentos = len(documentos_processados)
    termos_query = list(dict.fromkeys(query_tokens))
    detalhes = []
    pontuacoes = {doc_id: 0.0 for doc_id in documentos_processados}

    for termo in termos_query:
        df = len(indice.get(termo, []))
        idf = math.log(quantidade_documentos / df) if df else 0.0
        for doc_id, tokens in documentos_processados.items():
            frequencia = tokens.count(termo)
            tf = frequencia / len(tokens) if tokens else 0.0
            tfidf = tf * idf
            pontuacoes[doc_id] += tfidf
            detalhes.append({
                "Termo": termo,
                "Documento": doc_id,
                "Frequência": frequencia,
                "Total de termos": len(tokens),
                "TF": tf,
                "DF": df,
                "IDF": idf,
                "TF-IDF": tfidf,
            })

    ranking = sorted(pontuacoes.items(), key=lambda item: (-item[1], item[0]))
    return ranking, detalhes


def calcular_ranking_cosseno(query_tokens, documentos_processados, indice):
    """Calcula manualmente o cosseno entre vetores TF-IDF da query e docs."""
    quantidade_documentos = len(documentos_processados)
    vocabulario = sorted(indice)
    query_contagem = Counter(query_tokens)
    total_query = len(query_tokens)
    idfs = {
        termo: math.log(quantidade_documentos / len(indice[termo]))
        for termo in vocabulario
    }

    vetor_query = {
        termo: (query_contagem[termo] / total_query) * idfs[termo]
        if total_query else 0.0
        for termo in vocabulario
    }
    norma_query = math.sqrt(sum(valor ** 2 for valor in vetor_query.values()))
    resultado = []

    for doc_id, tokens in documentos_processados.items():
        contagem_doc = Counter(tokens)
        total_doc = len(tokens)
        vetor_doc = {
            termo: (contagem_doc[termo] / total_doc) * idfs[termo]
            if total_doc else 0.0
            for termo in vocabulario
        }
        produto = sum(vetor_query[t] * vetor_doc[t] for t in vocabulario)
        norma_doc = math.sqrt(sum(valor ** 2 for valor in vetor_doc.values()))
        denominador = norma_query * norma_doc
        similaridade = produto / denominador if denominador else 0.0
        resultado.append((doc_id, similaridade))

    return sorted(resultado, key=lambda item: (-item[1], item[0]))


def formatar_tokens_por_documento(estagios, chave):
    return pd.DataFrame([
        {"Documento": doc_id, "Tokens": " · ".join(dados[chave]) or "—"}
        for doc_id, dados in estagios.items()
    ])


def tabela_ranking(ranking, coluna_pontuacao):
    return pd.DataFrame([
        {
            "Posição": posicao,
            "Documento": doc_id,
            coluna_pontuacao: pontuacao,
            "Texto original": DOCUMENTOS[doc_id],
        }
        for posicao, (doc_id, pontuacao) in enumerate(ranking, start=1)
    ])


st.set_page_config(page_title="AgroSearch", page_icon="🌱", layout="wide")
st.title("🌱 AgroSearch")
st.subheader("Motor de Busca Inteligente")
st.write(
    "Aplicação didática de Recuperação de Informação: pré-processamento textual, "
    "índice invertido e ranqueamento são implementados manualmente."
)

st.header("📚 Base de documentos")
st.dataframe(
    pd.DataFrame(
        [{"Documento": doc_id, "Texto": texto} for doc_id, texto in DOCUMENTOS.items()]
    ),
    hide_index=True,
    use_container_width=True,
)

st.header("⚙️ Configurações")
coluna_1, coluna_2 = st.columns(2)
with coluna_1:
    usar_stopwords = st.checkbox("Remover Stopwords", value=True)
with coluna_2:
    usar_stemming = st.checkbox("Aplicar Stemming", value=True)

estagios_documentos = {
    doc_id: processar_texto(texto, usar_stopwords, usar_stemming)
    for doc_id, texto in DOCUMENTOS.items()
}
documentos_processados = {
    doc_id: dados["final"] for doc_id, dados in estagios_documentos.items()
}
indice = construir_indice(documentos_processados)

st.header("🧹 Pipeline de Pré-processamento")
with st.expander("1. Tokenização"):
    st.caption("Separa cada texto em palavras e descarta a pontuação.")
    st.dataframe(formatar_tokens_por_documento(estagios_documentos, "tokenizacao"), hide_index=True, use_container_width=True)
with st.expander("2. Normalização"):
    st.caption("Converte os tokens para minúsculas e remove acentos.")
    st.dataframe(formatar_tokens_por_documento(estagios_documentos, "normalizacao"), hide_index=True, use_container_width=True)
with st.expander("3. Remoção de Stopwords"):
    st.caption("Etapa ativa." if usar_stopwords else "Etapa desativada: os tokens foram mantidos.")
    st.dataframe(formatar_tokens_por_documento(estagios_documentos, "stopwords"), hide_index=True, use_container_width=True)
with st.expander("4. Stemming"):
    st.caption("Etapa ativa." if usar_stemming else "Etapa desativada: os tokens foram mantidos.")
    st.dataframe(formatar_tokens_por_documento(estagios_documentos, "stemming"), hide_index=True, use_container_width=True)

st.subheader("📖 Vocabulário")
vocabulario = sorted(indice)
st.write(f"**{len(vocabulario)} termos:** " + ", ".join(vocabulario))

st.header("🔗 Índice Invertido")
st.dataframe(
    pd.DataFrame([
        {"Termo": termo, "IDs dos documentos": ", ".join(documentos)}
        for termo, documentos in indice.items()
    ]),
    hide_index=True,
    use_container_width=True,
)

st.header("🔎 Busca")
consulta = st.text_input("Digite sua consulta:", placeholder="Ex.: irrigação soja")

if consulta.strip():
    etapas_query = processar_texto(consulta, usar_stopwords, usar_stemming)
    query_tokens = etapas_query["final"]
    st.write("**Consulta processada:**", " · ".join(query_tokens) if query_tokens else "—")

    if not query_tokens:
        st.warning("A consulta ficou vazia após o pré-processamento.")
    else:
        ranking_tfidf, detalhes = calcular_ranking_tfidf(
            query_tokens, documentos_processados, indice
        )
        st.header("📊 Ranking TF-IDF")
        if ranking_tfidf[0][1] > 0:
            st.success(f"🏆 Documento mais relevante: {ranking_tfidf[0][0]}")
        else:
            st.info("Nenhum termo da consulta foi encontrado na base.")

        tabela_tfidf = tabela_ranking(ranking_tfidf, "TF-IDF acumulado")
        st.dataframe(
            tabela_tfidf.style.format({"TF-IDF acumulado": "{:.6f}"}),
            hide_index=True,
            use_container_width=True,
        )

        with st.expander("Ver detalhes do cálculo: TF, DF, IDF e TF-IDF"):
            st.caption("TF = frequência ÷ total de termos | IDF = ln(N ÷ DF)")
            tabela_detalhes = pd.DataFrame(detalhes)
            st.dataframe(
                tabela_detalhes.style.format({
                    "TF": "{:.6f}", "IDF": "{:.6f}", "TF-IDF": "{:.6f}"
                }),
                hide_index=True,
                use_container_width=True,
            )

        st.header("⭐ Bônus — Similaridade de Cosseno")
        st.write(
            "A similaridade de cosseno compara a direção dos vetores TF-IDF. "
            "Ela é especialmente útil em consultas com múltiplas palavras."
        )
        ranking_cosseno = calcular_ranking_cosseno(
            query_tokens, documentos_processados, indice
        )
        st.dataframe(
            tabela_ranking(ranking_cosseno, "Similaridade").style.format(
                {"Similaridade": "{:.6f}"}
            ),
            hide_index=True,
            use_container_width=True,
        )
else:
    st.info("Digite uma consulta para gerar os rankings TF-IDF e de cosseno.")
