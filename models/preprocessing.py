import string
import re
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
slangs = open(os.path.join(base_dir, "slang.txt"),"r",encoding="utf-8", errors='replace')
clear_slangs= []
for newlines in slangs:
  strip_re = newlines.strip("\n")
  split = re.split(r'[:]',strip_re)
  clear_slangs.append(split)
slangs = [[k.strip(), v.strip()] for k,v in clear_slangs]
dict_slangs = {key:values for key,values in slangs}

def replace_use_dic(text):
    wordlist = TextBlob(text).words
    for k, v in enumerate(wordlist):
        if v in dict_slangs:
            wordlist[k] = dict_slangs[v]
    text = ' '.join(wordlist)
    return text

def lowercase(review_text):
  low = review_text.lower()
  return low

def remove_punctuation(review_text, default_text=" "):
  list_punct = string.punctuation
  delete_punct = str.maketrans(list_punct,' '*len(list_punct))
  new_review = ' '.join(review_text.translate(delete_punct).split())

  return new_review

def remove_superscript(review_text):
  number = re.compile("["u"\U00002070"
                      u"\U000000B9"
                      u"\U000000B2-\U000000B3"
                      u"\U00002074-\U00002079"
                      u"\U0000207A-\U0000207E"
                      u"U0000200D"
                      "]+", flags=re.UNICODE)
  return number.sub(r'', review_text)



def remove_non_clear_symbols(word):
    # Ekspresi reguler untuk mencocokkan simbol-simbol yang tidak jelas
    pattern = r'[^\w\s]'  # Mengabaikan karakter alfanumerik
    cleaned_word = re.sub(pattern, '', word)
    return cleaned_word

def word_repetition(review_text):
  review = re.sub(r'(.)\1+', r'\1\1', review_text)
  return review

def repetition(review_text):
  repeat = re.sub(r'\b(\w+)(?:\W\1\b)+', r'\1',review_text, flags=re.IGNORECASE)
  return repeat

def remove_extra_whitespaces(review_text):
  review = re.sub(r'\s+',' ', review_text)
  return review

#Banned words
bannedword = ['wkwk', 'wkwkw','wkwkwk','hihi','hihihii','hihihi','hehehe','hehehehe','hehe',
         'huhu','huhuu','ancok','guak','cokcok','hhmm','annya','huftt', 'nya', 'kiw', 'kiww', ' mmmmuah yummie']
re_banned_words = re.compile(r"\b(" + "|".join(bannedword) + ")\\W", re.I)
def RemoveBannedWords(toPrint):
    global re_banned_words
    return re_banned_words.sub("", toPrint)