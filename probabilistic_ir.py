import pandas as pd
import pyterrier as pt


def main(significance=False, tune_baseline=False, analyze_errors=False):
    if analyze_errors and not significance and not tune_baseline:
        brs, topics, qrels, names = pt_br_all(only_baseline=True)
        error_analysis(brs[0], topics, qrels, names[0])
        return

    brs, topics, qrels, names = pt_br_all()
    if significance:
        perform_experiment_significance(brs, topics, qrels, names)
    if tune_baseline:
        perform_bm25_tuning(brs[0].index, topics, qrels)

    if tune_baseline:
        perform_bm25_tuning(brs[0].indexref, topics, qrels)

    if significance:
        perform_experiment_significance(brs, topics, qrels, names)

    if analyze_errors:
        error_analysis(brs[1], topics, qrels, "BM25 (Stemmed Index)")


def pt_br_all(only_baseline=False):
    dataset = pt.datasets.get_dataset("trec-deep-learning-docs")

    topics = dataset.get_topics(variant="test")
    qrels = dataset.get_qrels(variant="test")

    indexref_us = dataset.get_index(variant="terrier_unstemmed_text")
    index_us = pt.IndexFactory.of(indexref_us)

    bm25_unstemmed = pt.BatchRetrieve(
        index_us, wmodel="BM25",
        metadata=['docno', 'text'],
        controls={"c": 0.5, "bm25.k_1": 1.4},
        properties={"termpipelines": ""}
    )

    if only_baseline:
        return [bm25_unstemmed], topics, qrels, ["BM25 Baseline"]

    indexref_st = dataset.get_index(variant="terrier_stemmed")
    index_st = pt.IndexFactory.of(indexref_st)

    indexref_st5 = dataset.get_index(variant="terrier_stemmed_docT5query")
    index_st5 = pt.IndexFactory.of(indexref_st5)

    bm25_stemmed = pt.BatchRetrieve(
        index_st, wmodel="BM25",
        controls={"c": 0.5, "bm25.k_1": 1.4},
        properties={"termpipelines": "Stopwords,PorterStemmer"}
    )

    bm25_stemmed_qe = pt.BatchRetrieve(
        index_st, wmodel="BM25",
        controls={"c": 0.5, "bm25.k_1": 1.4, "qe": "on", "qemodel": "Bo1"},
        properties={"termpipelines": "Stopwords,PorterStemmer"}
    )

    bm25_stemmed_qe_t5 = pt.BatchRetrieve(
        index_st5, wmodel="BM25",
        controls={"c": 0.5, "bm25.k_1": 1.4, "qe": "on", "qemodel": "Bo1"},
        properties={"termpipelines": "Stopwords,PorterStemmer"}
    )

    brs = [bm25_unstemmed, bm25_stemmed, bm25_stemmed_qe, bm25_stemmed_qe_t5]
    names = ["BM25 (Unstemmed)", "BM25 (Stemmed)", "BM25 (QueryExpansion + Stemmed)",
             "BM25 (QueryExpansion + Stemmed + docT5query)"]

    return brs, topics, qrels, names


# Used for significance testing of the different model improvement results
def perform_experiment_significance(batch_retrievals, topics, qrels, names):
    exp = pt.Experiment(
        batch_retrievals,
        topics,
        qrels,
        eval_metrics=["map", "ndcg"],
        names=names,
        round=3,
        baseline=0
    )

    print(exp)


# Used for tuning the bm25 parameters
def perform_bm25_tuning(indexref, topics, qrels):
    bm25_tuned = pt.BatchRetrieve(indexref, wmodel="BM25", controls={"c": 0.75, "bm25.k_1": 0.75})
    gs = pt.GridSearch(
        bm25_tuned,
        {bm25_tuned: {"c": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                      "bm25.k_1": [0.9, 1.2, 1.4, 1.6, 2, 2.5, 3, 3.5, 4]
                      }},
        topics,
        qrels,
        "map",
        verbose=True
    )
    print(gs)
    # Results are c=0.5 and k_1=1.4


# Initial error analysis, need to update to use the experiment results per query
def error_analysis(br, topics, qrels, name):
    # index = pt.IndexFactory.of(indexref)
    exp: pd.DataFrame = pt.Experiment(
        [br],
        topics,
        qrels,
        filter_by_qrels=True,
        eval_metrics=["map"],
        names=[name],
        perquery=True,
    )
    sorted_failing = exp.sort_values(by=["value"]).head(25)
    analyze_query_length(sorted_failing, topics)
    analyze_word_mismatch(br, sorted_failing, qrels, topics)


def analyze_word_mismatch(br, exp_sorted, qrels, topics):
    print("Analyzing word mismatch")
    worst_queries_5 = exp_sorted.head(5)

    def _extract_query_match(row):
        query = row["query"]
        text = row["text"]
        query_words = set(query.lower().split(" "))
        query_words_freq = {q: 0 for q in query_words}
        for word in text.lower().split(" "):
            if word in query_words:
                query_words_freq[word] += 1
        return query_words_freq

    def _total_query_score(row):
        qid = row['qid']
        wq = worst_queries_5.loc[worst_queries_5['qid'] == qid]
        return round(wq['value'].head(1).values[0], 4)

    pipe = br >> pt.apply.freq(_extract_query_match) >> pt.apply.query_score(_total_query_score)
    results = pipe.transform(topics.loc[topics['qid'].isin(worst_queries_5['qid'])])
    min_freq = results.sort_values('score', ascending=False).groupby('qid').head(5)
    min_freq_red = min_freq[['docno', 'qid', 'query_score', 'query', 'score', 'freq']].sort_values(by=['qid', 'score'])
    min_freq_red['score'] = min_freq_red['score'].round(4)
    min_freq_red.to_csv('min_freq.csv', index=False)
    # exp = pt.Experiment(
    #     [br >> pt.apply.freq_for(_extract_query_match)],
    #     topics,
    #     qrels,
    #     filter_by_qrels=True,
    #     eval_metrics=["map"],
    #     names=["Test Experiment"],
    #     perquery=True
    # )
    # print(exp)

    # for idx, qid in exp_sorted[['qid']].iterrows():
    #     qid = qid['qid']
    #     query_match = topics.loc[topics['qid'] == f'{qid}']
    #     qrel_docs = qrels.loc[qrels['qid'] == f'{qid}']
    #     qrel_docs = qrel_docs.loc[qrel_docs['label'] > 1]
    #     # print(qrel_docs)


def analyze_query_length(exp_sorted, topics):
    length_based_analysis = []
    for idx, row in exp_sorted[["qid", "value"]].iterrows():
        qid, val = row
        query_match = topics.loc[topics['qid'] == f'{qid}']
        query_val = query_match['query'].values[0]
        query_length = len(query_val)

        length_based_analysis.append({'qid': qid, 'qlength': query_length, 'score': round(val, 4)})

    df_lba = pd.DataFrame.from_records(length_based_analysis)
    df_lba.sort_values(by=['qlength']).to_csv('length_based_analysis.csv', index=False)


if __name__ == '__main__':
    if not pt.started():
        # Set pyterrier home to the root of the project as my C drive is slow and does not have a lot of space
        pt.init(home_dir="F:\Personal\coreir-pyterrier\.pyterrier")

    # Set pandas to display all columns
    pd.set_option('max_colwidth', None)
    pd.set_option('display.max_columns', None)

    main(analyze_errors=True)
