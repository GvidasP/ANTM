import pandas as pd
from antm.ctfidf import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, List, Any
import numpy as np
from scipy.sparse import csr_matrix
import openai
from antm.openai import OpenAI


def extract_representative_docs(
    c_tf_idf: csr_matrix,
    documents: pd.Series,
    vectorizer_model: CountVectorizer,
    ctfidf_model: ClassTfidfTransformer,
    index: int,
    nr_repr_docs: int = 5,
) -> Tuple[List[str], List[int]]:
    """Extract representative documents based on cosine similarity."""
    selected_docs = documents.tolist()
    selected_docs_ids = documents.index.tolist()

    # Transform documents into the cTF-IDF space
    bow = vectorizer_model.transform(selected_docs)
    doc_ctfidf = ctfidf_model.transform(bow)

    # Compute similarity with the topic's cTF-IDF vector
    topic_vector = c_tf_idf[index].reshape(1, -1)
    sim_matrix = cosine_similarity(doc_ctfidf, topic_vector)

    # Select the most similar documents
    top_indices = np.argsort(sim_matrix.flatten())[-nr_repr_docs:][::-1]

    return [selected_docs[i] for i in top_indices]


def ctf_idf_topics(
    docs_per_class: pd.Series,
    words: List[str],
    ctfidf: np.ndarray,
    num_terms: int,
) -> List[List[str]]:
    """Extract topics from cTF-IDF matrix."""
    return [
        [words[i] for i in ctfidf[int(label)].argsort()[-num_terms:]]
        for label in docs_per_class.unique()
    ]


def rep_prep(cluster_df):
    clusters_df = pd.concat(cluster_df)

    clusters_df_copy = clusters_df.copy()
    clusters_df_copy.loc[:, "num_doc"] = 1
    clusters_df = clusters_df_copy

    documents_per_topic_per_time = clusters_df.groupby(
        ["slice_num", "C"], as_index=False
    ).agg({"content": " ".join, "num_doc": "count"})
    documents_per_topic_per_time = (
        documents_per_topic_per_time.reset_index().rename(
            columns={"index": "cluster"}
        )
    )

    return documents_per_topic_per_time


def ctfidf_rp(
    dictionary: Any,
    documents_per_topic_per_time: pd.DataFrame,
    cluster_df: List[pd.DataFrame],
    azure_endpoint: str,
    api_key: str,
    num_words: int = 10,
    num_doc: int = 10,
):
    """Compute cTF-IDF topics and extract representative topics using OpenAI."""
    # Step 1: Compute cTF-IDF
    count_vectorizer = CountVectorizer(vocabulary=dictionary.token2id)
    count = count_vectorizer.fit_transform(documents_per_topic_per_time.content)
    words = count_vectorizer.get_feature_names_out()
    ctfidf_model = ClassTfidfTransformer().fit(count)
    ctfidf = ctfidf_model.transform(count).toarray()

    # Step 2: Extract topics
    topics_representations_ctf_idf = ctf_idf_topics(
        documents_per_topic_per_time.cluster, words, ctfidf, num_words
    )

    # Step 3: OpenAI Client Setup
    client = openai.AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version="2023-06-01-preview",
    )
    openai_model = OpenAI(
        client=client,
        chat=True,
        generator_kwargs={"model": "rs-gpt-3"},
        model="gpt-3.5-turbo",
    )

    clusters_df = pd.concat(cluster_df)
    grouped = clusters_df.groupby(["slice_num", "C"])
    group_topic_mappings = {}

    for index, ((slice_num, cluster_label), group) in enumerate(grouped):
        repr_docs = extract_representative_docs(
            c_tf_idf=ctfidf,
            documents=group["content"],
            vectorizer_model=count_vectorizer,
            ctfidf_model=ctfidf_model,
            index=index,
            nr_repr_docs=num_doc,
        )

        topics_openai = openai_model.extract_topics(
            documents=repr_docs, keywords=topics_representations_ctf_idf[index]
        )

        group_topic_mappings[(slice_num, cluster_label)] = topics_openai

    documents_per_topic_per_time["topics_representations_openai"] = (
        documents_per_topic_per_time.apply(
            lambda row: group_topic_mappings.get(
                (row["slice_num"], row["C"]), ""
            ),
            axis=1,
        )
    )
    output = documents_per_topic_per_time.assign(
        topics_representations_ctf_idf=topics_representations_ctf_idf
    )

    return output


def topic_evolution(list_tm, output):
    evolving_topics = []
    for et in list_tm:
        evolving_topic = []
        for topic in et:
            cl = int(float(topic.split("-")[1]))
            win = int(float(topic.split("-")[0]))
            t = output[output["slice_num"] == win]
            t = t[t["C"] == cl]
            evolving_topic.append(t.topics_representations_ctf_idf.to_list()[0])
        evolving_topics.append(evolving_topic)
    evolving_topics_df = pd.DataFrame({"evolving_topics": evolving_topics})
    return evolving_topics_df
