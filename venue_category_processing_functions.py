# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 16:14:36 2020

@author: Dillon Morse
"""

# =============================================================================
# Drop all stop-words (given by nltk corpus), punctuation, and any 
#   additionally-provided common words from the provided string.
#
# Return the pruned string
# =============================================================================
def drop_common_words( category, common = [] ):
    
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    import string 
    
    stop = stopwords.words('english') + list(string.punctuation) + list(common)
    token_words = [i for i in word_tokenize( category.lower() ) 
                           if i not in stop]    

    string = ''
    for word in token_words:
        string += ' ' + word
    
    return string


# =============================================================================
# Unpack provided nested lists/dicts as a flat list 
# =============================================================================
def unpack(it):
    if isinstance(it, list):
        for sub_it in it:
            yield from unpack(sub_it)
    elif isinstance(it, dict):
        for value in it.values():
            yield from unpack(value)
    else:
        yield it
        
        
# =============================================================================
# Given a list of words, appends the list with the synonyms of these words as
#   well (with option to specify the number of synonyms to include)        
# =============================================================================
def include_synonyms( token_words, num_syns = -1 ):
    from PyDictionary import PyDictionary
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    import string
    
    dic = PyDictionary() 
    stop = stopwords.words('english') + list(string.punctuation)
    synonyms = []

    for word in token_words:
        try:
            syns = [word_tokenize(i) for i in dic.synonym(word) if i not in stop]
            if num_syns == -1:
                synonyms += list( unpack(syns) )
            else:
                synonyms += list( unpack(syns) )[:num_syns]
                synonyms = [word.lower() for word in synonyms]
        except TypeError :
            print('No synonyms for ', word)
    
    #synonyms = list( set( synonyms ) )
    
    return synonyms


# =============================================================================
# Given a list of words, pass each through the SnowballStemmer and return
# =============================================================================
def stem(bow):
    from nltk.stem import SnowballStemmer
    stemmer = SnowballStemmer("english")
    
    stemmed_words = []
    for word in bow:
        stemmed_words.append( stemmer.stem(word) )
    return stemmed_words


# =============================================================================
# Given a four-square-provided category, tokenize, lower, strip accents, 
#   remove common/stop words, optionally include synonyms, optionally stem,
#   then retrieve definition for each remaining word.
#
# Return list of all words appearing in the definitions of the category word
#   and their synonyms. 
# =============================================================================
def get_definition_bow( category,
                        common = [], 
                        incl_synonyms = False,
                        num_syns = 0,
                        stem_bow = False):
    
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    import string  
    from PyDictionary import PyDictionary
    from unidecode import unidecode
    
    category = drop_common_words( category, common ) # Remove common words
    
    dic = PyDictionary() 
    
    stop = stopwords.words('english') + list(string.punctuation)
    stop += ["'s"] # Include apostrophe-s in words that will always be dropped
    
    token_words = [unidecode(i) for i in word_tokenize( category.lower() ) 
                                if i not in stop]
    
    token_words = [word.replace('bbq', 'barbecue') for word in token_words]
        # bbq is common enough that it needs to be handled manually
    
    
    if incl_synonyms:
        token_words += include_synonyms( token_words, num_syns ) 
    
    token_words = [word.replace('-', '') for word in token_words]
        # Hyphens confuse the dictionary
    
    print('Token words to get defs for are: ', token_words)
    defin_str = ''
    for word in [i for i in token_words if i not in stop]:
        defins = []
        
            # Take noun-definition first if available, then adj. then verb.
        for part_speech in ('Noun', 'Adjective', 'Verb', 'Adverb'):
            try:
                defins = dic.meaning(word)[part_speech]
                #break
            except:
                continue
        #else:
            #print('No parts of speech match the definition of ', word)
                # If there are no nouns, adjective, or verbs matching
          
        for defin in defins:
            defin_str += defin
            defin_str += ' '
        defin_str = defin_str.strip()
    bow = [i for i in word_tokenize( defin_str.lower() ) if i not in stop]
    bow += token_words

    if stem_bow:
        bow = stem(bow)    
           
    return bow


# =============================================================================
# Given a list of words, combine them in to a single long string
# =============================================================================
def list_to_str( bow ):
    string = ''
    for word in bow:
        string += word
        string += ' '
    string = string.strip()
    return string


# =============================================================================
# 
# =============================================================================
if __name__ == '__main__':

    cat = 'golf'
    
    common = ['joint', 'shop', 'spot', ]
    #print( drop_com, common_words(cat, common) )
    a = get_definition_bow(cat, stem_bow = False, incl_synonyms = True, num_syns = 1)
   # print( a )
    
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
    vectorizer.fit(a)
    
    b = vectorizer.transform((a))
    
    #print(b)