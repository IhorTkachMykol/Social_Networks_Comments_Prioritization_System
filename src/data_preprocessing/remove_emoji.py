import re
import emoji

def remove_emoji(text):
    '''this function removes emoji from text'''
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Example:
# data['text']=data['text'].apply(lambda x: remove_emoji(x))

def convert_emoji_to_text(query):
    '''
    Convert emojis into text representations.
    
    Args:
        query (str): User's comment.
        
    Returns:
        str: Query with emojis converted into text.
    '''
    try:
        text_with_aliases = emoji.demojize(query, delimiters=("", ""))
        return text_with_aliases
    except Exception as e:
        print(f"Error converting emojis to text: {e}")
        return query