from string import digits, punctuation
from copy import deepcopy
from typing import List
import nltk
import spacy
from nltk.corpus import stopwords


nltk.download('punkt')
nltk.download("stopwords")
german_stop_words = stopwords.words('german')
nlp_de = spacy.load('de_core_news_lg')


def convert_to_lowercase(text: str) -> str:
    """Convert a word into its lowercase representation

    Args:
        text: text as string

    Returns:
        str: manipulated text as str
    """

    temp_var = text
    return temp_var.lower()


def replace_umlauts(text: str) -> str:
    """Replace umlauts for a given text

    Args:
        text: text as string

    Returns:
        temp_var: manipulated text as str
    """

    temp_var = text  # local variable

    # Using str.replace()

    temp_var = temp_var.replace('ä', 'ae')
    temp_var = temp_var.replace('ö', 'oe')
    temp_var = temp_var.replace('ü', 'ue')
    temp_var = temp_var.replace('Ä', 'Ae')
    temp_var = temp_var.replace('Ö', 'Oe')
    temp_var = temp_var.replace('Ü', 'Ue')
    temp_var = temp_var.replace('ß', 'ss')

    return temp_var


def remove_punctuation(text: str) -> str:
    """Remove punctuations (!"#%&\'()*+,-/:;<=>?@[\\]^_`{|}~) for a given text

    Args:
        text: text as string

    Returns:
        temp_var: manipulated text as str
    """

    temp_var = text
    temp_var = temp_var.translate(str.maketrans('', '', punctuation))

    return temp_var


def remove_digits(text: str) -> str:
    """Remove digits (0123456789) for a given text

    Args:
        text: text as string

    Returns:
        temp_var: manipulated text as str
    """
    temp_var = text
    temp_var = temp_var.translate(str.maketrans('', '', digits))

    return temp_var


def remove_short_word(list_text: List[str]) -> List[str]:
    """Remove words with length < 2 for a given list of texts

    Args:
        list_text: list of texts

    Returns:
        new_list: manipulated list of texts
    """
    new_list = deepcopy(list_text)

    for val in list_text:

        if len(val) <= 2:
            new_list.remove(val)

    return new_list


def remove_stopwords(list_text: List[str]) -> List[str]:
    """Remove stopwords (auf, der, die, das, osw) for a given list of texts

    Args:
        list_text: list of texts

    Returns:
        list: manipulated list of texts
    """

    return [word for word in list_text if word not in german_stop_words]


def list_restructure(list_text: List[str]) -> List[str]:
    """Transforms two level nested list into one level list for a given list

    Args:
        list_text: list of texts

    Returns:
        list: manipulated list of texts
    """

    return [j for i in list_text for j in i]


def tokenizer(text: str) -> str:
    """Remove digits for a given text

    Args:
        text: text as string

    Returns:
        temp_var: manipulated text as str
    """

    temp_var = text
    temp_var = nltk.tokenize.word_tokenize(temp_var, language='german')

    return temp_var


def lemmatizer(list_text: List[str]) -> List[str]:
    """Lemmatize words for a given list of texts

    Args:
        list_text: list of texts

    Returns:
        text_lemma: manipulated list of texts
    """

    text_lemma = []

    for text in list_text:
        words = nlp_de(text)
        result = ' '.join([x.lemma_ for x in words])
        text_lemma.append(result)

    return text_lemma


def clean_text(text: str):
    """ Clean a text separated by whitespace

    1. Convert to lowercase
    2. remove punctuation
    3. tokenize
    4. remove digits
    5. remove stopwords
    6. remove short words
    7. perform lemmatization
    8. replace 'umlauts'
    9. Join with whitespaces

    Args:
        text:

    Returns:

    """
    json_token = text.split(" ")
    if len(json_token) > 0:
        text_lower = [convert_to_lowercase(word) for word in json_token]
        text_no_punct = [remove_punctuation(word) for word in text_lower]
        text_tokenized = [tokenizer(word) for word in text_no_punct]
        text_tokenized = list_restructure(text_tokenized)
        text_no_digits = [remove_digits(word) for word in text_tokenized]
        text_no_stopwords = remove_stopwords(text_no_digits)
        text_no_short_word = remove_short_word(text_no_stopwords)
        text_lemma = lemmatizer(text_no_short_word)
        word_tokens = [replace_umlauts(word) for word in text_lemma]
        final_text = " ".join(word_tokens)
    else:
        final_text = 'Empty'

    return final_text