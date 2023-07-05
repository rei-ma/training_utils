import numpy as np

def ocr_number_correction(text: str) -> str:
    """ Reformat string representation of number to be able to convert it to float
    1. Remove currency symbols
    2. Strip whitespaces and leading 0
    3. Replace , by .
    4. Replace ambigious characters with digits

    Args:
        text: Text representation of number

    Returns:
        str: reformatted string
    """
    currency_symbols = ["$", "€", "%", "EUR", "EURO"]
    text = text.strip()
    # text = text.strip("0")
    text = text.replace(" ", "")
    # Format to float string
    if text.count(".") == 2:
        li = text.rfind(".")
        if li - text.find(".") == 4:
            text = text[:li] + "," + text[li + 1:]
    if text.count(",") == 2:
        li = text.find(",")
        text = text[:li] + "," + text[li + 1:]

    if "." in text and "," in text:
        if text.find(".") > text.find(","):
            text.replace(",", "")
        else:
            text = text.replace(".", "")

    text = text.replace(",", ".")

    for currency_symbol in currency_symbols:
        text = text.replace(currency_symbol, "")

    text = text.replace("O", "0").replace("I", "1").replace("Q", "0").strip()  # Textract Correction
    return text


def is_number(text: str) -> bool:
    """ Check if string represents a number
    First perform correction and then try to convert to float

    Args:
        text: String with possible number

    Returns:
        bool: Indicator if conversion is possible
    """
    try:
        float(ocr_number_correction(text))
        return True
    except:
        return False


def is_betrag(text: str) -> bool:
    """ Check if string represents an amount
    First check if it is a number and also contains two decimal places

    Args:
        text: String with possible amount

    Returns:
        bool: Indicator if it is an amount

    """
    text = text.lstrip("0")
    if len(text) < 3:
        return False
    if text[-3] not in [".", ","]:
        return False
    return is_number(text)


def to_float(text: str) -> float:
    """ Convert string to float if possible
    Else return np.nan

    Args:
        text: string representation of float

    Returns:
        float: converted float or np.nan

    """
    text = ocr_number_correction(text)
    try:
        return float(text)
    except:
        return np.nan

