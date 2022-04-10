import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

from cbdt.features import feature as ft
from cbdt.features.cbmodel import ClickbaitModel
from cbdt.features.dataset import ClickbaitDataset
from cbdt.features.feature_builder import FeatureBuilder


def build_new_features(cbd, name):
    print('defining features')
    char_3grams = ft.NGramFeature(TfidfVectorizer, o=3, analyzer='char', fit_data=cbd.get_x('postText'), cutoff=3)
    word_3grams = ft.NGramFeature(TfidfVectorizer, o=3, fit_data=cbd.get_x('postText'), cutoff=3)

    stop_word_ratio = ft.ContainsWordsFeature("wordlists/TerrierStopWordList.txt", ratio=True)
    easy_words_ratio = ft.ContainsWordsFeature("wordlists/DaleChallEasyWordList.txt", ratio=True)
    mentions_count = ft.ContainsWordsFeature(['@'], only_words=False)
    hashtags_count = ft.ContainsWordsFeature(['#'], only_words=False)
    clickbait_phrases_count = ft.ContainsWordsFeature("wordlists/DownworthyCommonClickbaitPhrases.txt",
                                                      only_words=False)
    flesch_kincait_score = ft.FleschKincaidScore()
    has_abbrev = ft.ContainsWordsFeature("wordlists/OxfortAbbreviationsList.txt", only_words=False, binary=True)
    number_of_dots = ft.ContainsWordsFeature(['.'], only_words=False)
    start_with_number = ft.StartsWithNumber()
    longest_word_length = ft.LongestWordLength()
    mean_word_length = ft.MeanWordLength()
    char_sum = ft.CharacterSum()
    has_media_attached = ft.HasMediaAttached()
    part_of_day = ft.PartOfDay()
    sentiment_polarity = ft.SentimentPolarity()

    f_builder = FeatureBuilder((char_3grams, 'postText'),
                               (word_3grams, 'postText'),
                               (hashtags_count, 'postText'),
                               (mentions_count, 'postText'),
                               (sentiment_polarity, 'postText'),
                               (flesch_kincait_score, 'postText'),
                               (has_abbrev, 'postText'),
                               (number_of_dots, 'postText'),
                               (start_with_number, 'postText'),
                               (longest_word_length, 'postText'),
                               (mean_word_length, 'postText'),
                               (char_sum, 'postText'),
                               (has_media_attached, 'postMedia'),
                               (part_of_day, 'postTimestamp'),
                               (easy_words_ratio, 'postText'),
                               (stop_word_ratio, 'postText'),
                               (clickbait_phrases_count, 'postText'))

    for file_name in os.listdir("wordlists/general-inquirer"):
        f = ft.ContainsWordsFeature("wordlists/general-inquirer/" + file_name)
        f_builder.add_feature(feature=f, data_field_name='postText')

    # char_3grams_mc = ft.NGramFeature(TfidfVectorizer, o=3, analyzer='char', fit_data=cbd.get_x('targetParagraphs'),
    #                                  cutoff=3)
    # word_3grams_mc = ft.NGramFeature(TfidfVectorizer, o=3, fit_data=cbd.get_x('targetParagraphs'), cutoff=3)
    #
    # f_builder.add_feature(feature=char_3grams_mc, data_field_name='targetParagraphs')
    # f_builder.add_feature(feature=word_3grams_mc, data_field_name='targetParagraphs')
    # f_builder.add_feature(feature=flesch_kincait_score, data_field_name='targetParagraphs')
    # f_builder.add_feature(feature=mean_word_length, data_field_name='targetParagraphs')

    # f_builder.add_feature(feature=longest_word_length, data_field_name='targetTitle')
    # f_builder.add_feature(feature=has_abbrev, data_field_name='targetTitle')
    # f_builder.add_feature(feature=easy_words_ratio, data_field_name='targetTitle')


    print('building features')
    f_builder.build(cbd, save=True)
    print(f'storing feature schema as {name}.pkl')
    pickle.dump(obj=f_builder, file=open(f"../persist/{name}.pkl", "wb"))
    return f_builder


if __name__ == "__main__":
    dataset = ClickbaitDataset(
        instances_path="../corpus/clickbait17-train-170331/instances.jsonl",
        truth_path="../corpus/clickbait17-train-170331/truth.jsonl"
    )

    name = "clickbait_features_tweet_paragraphs_title"

    try:
        with open(f"../persist/{name}.pkl", "rb") as f:
            f_builder = pickle.load(f)
    except Exception:
        f_builder = build_new_features(dataset, name)
    features = f_builder.build_features

    print('training model')
    cbm = ClickbaitModel()
    y = dataset.get_y()
    cbm.classify(features, dataset.get_y_class(), LinearSVC(), evaluate=True)
    # cbm.regress(features, y, "Ridge", evaluate=True)

    print(f'stroring trained model as {name}_trained.pkl')
    cbm.save(f"../persist/{name}_trained.pkl")
