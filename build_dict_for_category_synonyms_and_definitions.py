# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 12:48:46 2020

@author: Dillon Morse
"""

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string 
from PyDictionary import PyDictionary


# =============================================================================
# Create a list of all unique venue categories
# =============================================================================
def get_all_cats():
    import pandas as pd
    from unidecode import unidecode
    
    df = pd.concat( [pd.read_csv('denver_venue_data.csv'),
                     pd.read_csv('triangle_venue_data.csv') ]).reset_index(drop = True)
    df.dropna(axis = 0, how = 'any', inplace = True)
    df['Category'] = [unidecode(word) for word in df['Category'] ]
    all_cats = df['Category'].unique()
    
    return list(all_cats)
    

# =============================================================================
# Build dictionary containing all category keywords as well as all of their 
#   synonyms.
# Dict keys: top-level: category, next level: 'keywords', 'synonyms'
# =============================================================================
def build_dict(categories):
    dictionary = {}
    num_categories = len(categories)
    
    l = 0
    for cat in categories:
        dictionary[cat] = {}
        
        
        # Lowercase, remove stop words, keep all 1 and 2-grams 
        #   Under key: 'keywords'
        stop = stopwords.words('english') + list(string.punctuation)
        token_words = [ i for i in word_tokenize( cat.lower() ) 
                                if i not in stop ]
        bigrams = []
        if len(token_words) >= 2:
            for j in range( len(token_words)-1 ):
                bigrams.append( token_words[j] + ' ' + token_words[j+1] )
            
        dictionary[cat]['keywords'] = token_words + bigrams
        
        
        # Get synonyms for each keyword, under key: synonyms
        dictionary[cat]['synonyms'] = {}
        one_grams = [ word for word in dictionary[cat]['keywords'] 
                                    if len(word_tokenize(word)) == 1  ]
        
        for m, word in enumerate(one_grams):
            if word == 'bbq':
                one_grams[m] = 'barbecue' # bbq isn't recognized by dictionary        
        
        for word in one_grams:
            try:
                syns = PyDictionary.synonym(word)
            except:
                continue
            if syns == None:
                syns = []
            dictionary[cat]['synonyms'][word] = [k for k in syns 
                                                if len(word_tokenize(k)) == 1 ] 
        l += 1
        pct = int( l/num_categories*100 )
        if pct%5 == 0:
            print(pct, ' percent complete building dictionary.')
    return dictionary
    
    
# =============================================================================
# Build dictionary containing the definition of all keywords and all synonyms
# Dict keys: top-level: category, next-level: 'Keyword_Defs', 'Synonym_Defs'
# =============================================================================
def get_defs(dictionary):
    
    categories = [word for word in dictionary]
    num_categories = len(categories)
    definitions = {}
    
    l = 0
    for cat in categories:
        definitions[cat] = {}
           
        # Add definitions for base words
        defin = []
        one_grams = [ word for word in dictionary[cat]['keywords'] 
                            if len(word_tokenize(word)) == 1  ]
        
        for m, word in enumerate(one_grams):
            if word == 'bbq':
                one_grams[m] = 'barbecue' # bbq isn't recognized by dictionary
            
        for word in one_grams:
            PyDict = PyDictionary.meaning(word)
            if PyDict == None:
                defin += []
            else:
                for part in PyDict:
                    defin += PyDict[part]
        definitions[cat]['Keyword_Defs'] = defin
        
        
        # Add definitions for synonyms
        definitions[cat]['Synonym_Defs'] = {}
        for word in dictionary[cat]['synonyms']:
            definitions[cat]['Synonym_Defs'][word] = {}
            for syn in dictionary[cat]['synonyms'][word]:
                defin = []
                PyDict = None
                try:
                    PyDict = PyDictionary.meaning(syn)
                except:
                    continue
                if PyDict == None:
                    defin += []
                else:
                    for part in PyDict:
                        defin += PyDict[part]               
                definitions[cat]['Synonym_Defs'][word][syn] = defin
        
        l += 1
        pct = int( l/num_categories*100 )
        if pct%10 == 0:
            print(pct, ' percent complete fetching definitions.')
    return definitions


all_cats = get_all_cats()


# =============================================================================
# Re-build the above dictionaries, a very time-consuming process
# Save results to pickle files for later access
# =============================================================================
def rebuild_all_category_dicts():
    all_cats = get_all_cats()
    dictionary = build_dict( all_cats )
    definitions = get_defs( dictionary )
    
    import pickle
    
    with open('dict_for_cat_keywords.pickle', 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('dict_for_cat_defs.pickle', 'wb') as handle:
        pickle.dump(definitions, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return None
    

# =============================================================================
# Delete dictionary entries for non-useful categories (ensure that they are
#   also removed from cities dataframes)
# =============================================================================
def del_entries( dictionary, definitions, cats):
    
    dict_new = dict(dictionary)
    defs_new = dict(definitions)
        
    for category in cats:
        del dict_new[category]
        del defs_new[category]
    
    return dict_new, defs_new

# =============================================================================
# Builds dictionary for use by clustering algorithm. Each entry contains a
#   list of strings, among them are the category keywords and synonyms as
#   well as the definitions of all the above. Can control the number of
#   synonyms used - note that many of the synonyms are related to other uses
#   of the keywords in various contexts and thus are not beneficial
# Dict keys: category
# =============================================================================
def all_category_defs(num_syns = 0):
    
    import pickle
    with open('dict_for_cat_keywords.pickle', 'rb') as handle:
        dictionary = pickle.load(handle)  
    with open('dict_for_cat_defs.pickle', 'rb') as handle:
        definitions = pickle.load(handle)
    
    
    remove_cats = ['Moving Target', 'Road', 'Intersection']
    dictionary, definition = del_entries(dictionary, definitions, remove_cats)
    
    all_defs = {}
    
    if num_syns == -1:
        for cat in dictionary:
            keys = dictionary[cat]['keywords']
            one_grams = []
            for word in keys:
                if len(word_tokenize(word)) == 1:
                    one_grams.append(word)
            all_defs[cat] = one_grams
    else:
        for cat in dictionary:
            defin = []
            
            defin += dictionary[cat]['keywords']
            defin += definitions[cat]['Keyword_Defs']
            
            if num_syns > 0:
                for word in dictionary[cat]['synonyms']:
                    l = len( dictionary[cat]['synonyms'][word] )
                    k = min( num_syns, l )
                    
                    defin += dictionary[cat]['synonyms'][word][:k]
                    
                    j = 0
                    for syn in definitions[cat]['Synonym_Defs'][word]:
                        defin += definitions[cat]['Synonym_Defs'][word][syn]
                        j += 1
                        if j == num_syns:
                            break
            
            all_defs[cat] = defin 
        
    return all_defs


# =============================================================================
# Remove all stopwords and lemmatize (or stem - lemmatization recommended)
#   the dictionary 'all_cat_defs' which is intended to be the output of the 
#   similarly-named above function
# =============================================================================
def clean_defs( all_cat_defs, stem = False, lem = False ):
   
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    import string 
    
    
    if stem:
        from nltk.stem import SnowballStemmer
        stemmer = SnowballStemmer("english")
    if lem:
        import spacy
        sp = spacy.load('en_core_web_lg')
    
    stop = stopwords.words('english') + list(string.punctuation) + ["'s"]
    
    cleaned_defs = {}
    
    for cat in all_cat_defs:
        defin = []
        for strin in all_cat_defs[cat]:
            
            if stem:
                token_words = [stemmer.stem(i) for i in 
                              word_tokenize( strin.lower() ) if i not in stop] 
            if lem:
                strin = sp( strin.lower() )
                token_words = [ word.lemma_ for word in strin 
                                                     if word.text not in stop]
            else:
                token_words = [i for i in word_tokenize( strin.lower() ) 
                                       if i not in stop]
            
            new_string = ''
            for word in token_words:
                new_string += word + ' ' 
            new_string = new_string.rstrip()
            defin.append(new_string)         
            
        cleaned_defs[cat] = defin
    return cleaned_defs


# =============================================================================
# If called as main, run the above functions for testing purposes
# =============================================================================
if __name__ == '__main__':
    all_defs = all_category_defs(num_syns = 0)
    cleaned_defs = clean_defs( all_defs, stem = False, lem = True )


# =============================================================================
# Use spacy words-to-vec to vectorize the keywords, synonyms, and definitions
#   for each category. The input here is intended to be the output of
#   clean_defs().
# Builds dateframe with 300 columns corresponding the 300 dimensions of the 
#   word vectors, each row corresponding to a category. The final column
#   contains the category names from four-square.
# Writes output to pickle file for later use.
# =============================================================================
def vectorize_defs( defs  ):
    import spacy 
    import numpy as np
    import pandas as pd

    
    vect_dim = 300
    all_feat_vectors = np.zeros( (len(defs), vect_dim) )
    
    nlp = spacy.load('en_core_web_lg')
    for i, category in enumerate( defs ):
        num_defs = len( defs[category] )
        
        feat_vector = np.zeros( (1, vect_dim) )
        for defin in defs[category]:
            feat_vector += nlp(defin).vector
            
        feat_vector = np.divide( feat_vector, num_defs )
    
        all_feat_vectors[i,:] = feat_vector
        
    word_vecs = pd.DataFrame( data = all_feat_vectors )
    word_vecs['Category'] = [cat for cat in defs]
    
    word_vecs.to_pickle('vectorized_defs_dataframe.pickle')
    return None


# =============================================================================
# Some of the keywords from the four-square categories are not present in the
#   PyDictionary and so need to be manually added. Be sure to do this prior
#   to cleaning the definitions.
# =============================================================================
def manual_defs( def_dict ):
    defins = def_dict
    
    defins['Gastropub'].append('a pub that specializes in serving high quality food')
    
    defins['Vape Store'].append('inhale and exhale the vapor produced by an electronic cigarette or similar device')
    defins['Vape Store'].append('an electronic cigarette or similar device')

    defins['Acupuncturist'].append('a person who practices acupuncture')
    defins['Acupuncturist'].append('a system of integrative medicine that involves pricking the skin or tissues with needles used to alleviate pain and to treat various physical mental and emotional conditions')
    
    defins['Coworking Space'].append('the use of an office or other working environment by people who are self employed or working for different employers typically so as to share equipment ideas and knowledge')
    
    defins['Creperie'].append('a small restaurant typically one in france in which a variety of crepes are served')
    
    defins['Ramen Restaurant'].append('in oriental cuisine quick cooking noodles typically served in a broth with meat and vegetables')
    
    defins['Martial Arts Dojo'].append('a room or hall in which judo and other martial arts are practiced')
    
    defins['Motorsports Shop'].append('any of several sports involving the racing or competitive driving of motor vehicles')
    
    defins['Tex-Mex Restaurant'].append('especially of cooking and music having a blend of mexican and southern american features originally characteristic of the border regions of texas and mexico')
    
    defins['Churrascaria'].append('a churrascaria is a place where meat is cooked in churrasco style which translates roughly from the portuguese word for barbecue churrascaria cuisine is typically served rodizio style where roving waiters serve the barbecued meats from large skewers directly onto the seated diners plates')
    defins['Churrascaria'].append('a restaurant that specializes in serving steaks')
    
    return defins


# =============================================================================
# Cluster the category information contained in the
#   'vectorized_defs_dataframe.pickle' file. Output results to the file
#   'clustered_categories.pickle'
# =============================================================================
def cluster_cats(num_clusters, rebuild = True, num_syns = 0, plot = False):
    
    import pandas as pd
    
    
    # =========================================================================
    # Re-build definitions dictionary with different numbers of synonyms used. 
    #   Can be slow - a minute or more depending on how many synonyms are 
    #   included
    # =========================================================================
    
    if rebuild:
        print('Getting all category definitions.')
        all_defs = all_category_defs(num_syns = num_syns)
        all_defs = manual_defs(all_defs)
        print('Cleaning all definitions.')
        cleaned_defs = clean_defs( all_defs, stem = False, lem = True )
        print('Vectorizing all definitions.')
        vectorize_defs( cleaned_defs  )
        
        
    # =========================================================================
    # Read in vectorized definitions from pickle file
    # =========================================================================
    word_vecs = pd.read_pickle('vectorized_defs_dataframe.pickle')
    
    
    # =========================================================================
    # Cluster the definition word vectors
    # =========================================================================
    from sklearn.cluster import KMeans
    
    ap = KMeans(n_clusters = num_clusters)
    ap.fit( word_vecs.iloc[:,:-1] )
    cluster_labels = ap.labels_
    cluster_labels = pd.DataFrame(cluster_labels, columns=['ClusterLabel'])
    cluster_labels['Category'] = word_vecs['Category']
    cluster_labels.to_pickle('clustered_categories.pickle')
    
    if plot:
        # =====================================================================
        # Use PCA to reduce dimensions, plot results color-coded by the 
        #   clusters above
        # =====================================================================
        from sklearn.decomposition import PCA
        
        redu_word_vecs = (pd.DataFrame( data = PCA(n_components = 3)
                           .fit_transform( word_vecs.iloc[:, :-1] ) )
                         )
        redu_word_vecs['Category'] = word_vecs['Category']
        redu_word_vecs['Cluster'] = cluster_labels['ClusterLabel']
        
        import matplotlib
        matplotlib.use('nbagg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        from IPython import get_ipython
        get_ipython().run_line_magic('matplotlib', 'qt')
        
        x = redu_word_vecs.iloc[:,0]
        y = redu_word_vecs.iloc[:,1]
        z = redu_word_vecs.iloc[:,2]
        
        fig = plt.figure()
        ax = Axes3D(fig)
        
        ax.scatter(x, y, z,
                    c = redu_word_vecs.iloc[:,4],
                    cmap = plt.cm.tab10,
                    alpha = 1,
                    s = 150,
                    )
        
        plt.show(cluster_labels)
    
    #cluster_info(cluster_labels)
    
    return None
        
# =============================================================================
# Print info about clustering
# =============================================================================
def cluster_info(cluster_labels):
    
    cluster_labels['Total'] = 1
    sums = cluster_labels.groupby('ClusterLabel').sum()
    
    # Find closest - mape to the rows of the summed rows
    
    print(sums)
    
    
    
    return


# =============================================================================
# Cluster category labels, map on to city region dataframe
# =============================================================================
def category_cluster_and_map(city_data, num_clusters = 10, rebuild = True,
                             num_syns = 2, plot = False ):
    import pandas as pd
    from unidecode import unidecode
    
    all_data = city_data.copy()
    
    cluster_cats(num_clusters = num_clusters,
                 rebuild = rebuild,
                 num_syns = num_syns,
                 plot = plot
                 )
    category_clusters = pd.read_pickle('clustered_categories.pickle')
    
    category_clusters['Total'] = 1
    category_clusters.groupby('ClusterLabel').sum()
    
    cluster_dict = (category_clusters.set_index('Category')
                                     .to_dict('dict')['ClusterLabel']
                                     )
    all_data['Cluster'] = [cluster_dict[unidecode(category)] 
                           for category in all_data['Category']]
    
    return all_data



