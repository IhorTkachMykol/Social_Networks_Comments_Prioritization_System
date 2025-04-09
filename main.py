from src.data_preprocessing.remove_emoji import remove_emoji
from src.keywords_classifier.detect_keywords import detect_keywords
from src.spam_classifier.spam_classifier import detect_spam
from src.data_preprocessing.remove_emoji import convert_emoji_to_text
from src.langchain.my_langchain_history import conversation


def detect_class(query):
    """
    Detects the class of the input query based on keyword, spam, or normal content.

    Args:
        query (str): The input query to classify.

    Returns:
        str: The class of the input query, which can be one of the following:
            - "emoji" if the query contains only emojis.
            - "keyword" if the query is identified as a keyword.
            - "spam" if the query is classified as spam.
            - "normal" if the query is neither a keyword nor spam.
    """
    # ========== Input Preprocessing ==========

    query = convert_emoji_to_text(query) # Remove all emoji from text
    

    # example, remove if doesn't work
    query = query.lower().strip() # make all lowercase

    # ========== Branching Logic ==========


    is_keyword = detect_keywords(query)
    if is_keyword:
        return "keyword"
    else:
        is_spam = detect_spam(query)
        if is_spam:
            return "spam"
    
    conversation(query)

if __name__ == '__main__':
   query = input("You: ")
   print(detect_class(query))