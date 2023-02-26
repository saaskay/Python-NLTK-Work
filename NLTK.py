import nltk
from nltk.corpus import brown
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

# Download necessary NLTK data
nltk.download('brown')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('gutenberg')
nltk.download('treebank')
# Explore Brown Corpus
size = len(brown.words())
tokens = brown.words()
categories = brown.categories()
govt_size = len(brown.words(categories='government'))
freq_dist = FreqDist(tokens)
most_common = freq_dist.most_common(10)
sentences = len(brown.sents())

print("Brown Corpus Info:")
print(f"Size: {size}")
print(f"Tokens: {len(tokens)}")
print(f"Word Types: {len(set(tokens))}")
print(f"Government Category Size: {govt_size}")
print(f"Most Common Tokens: {most_common}")
print(f"Number of Sentences: {sentences}")

# Explore Raw Corpus
raw_corpus = nltk.corpus.gutenberg.raw('melville-moby_dick.txt')
print("\nRaw Corpus:")
print(raw_corpus[:100])

# Explore POS Tagged Corpus
tagged_corpus = nltk.corpus.treebank.tagged_sents()[0]
print("\nPOS Tagged Corpus:")
print(tagged_corpus)

# Perform Text Processing on Sample Text
sample_text = """This is a sample text with multiple sentences. It contains punctuations,
                 upper and lowercase letters, and stop words. The purpose of this text is to 
                 demonstrate text processing using Python NLTK library."""

# Word Segmentation
word_tokens = word_tokenize(sample_text)
print("\nWord Segmentation:")
print(word_tokens)

# Sentence Segmentation
sent_tokens = sent_tokenize(sample_text)
print("\nSentence Segmentation:")
print(sent_tokens)

# Convert to Lowercase
lowercase_tokens = [token.lower() for token in word_tokens]
print("\nLowercase Conversion:")
print(lowercase_tokens)

# Stop Words Removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in lowercase_tokens if token not in stop_words]
print("\nStop Words Removal:")
print(filtered_tokens)

# Stemming
porter_stemmer = PorterStemmer()
stemmed_tokens = [porter_stemmer.stem(token) for token in filtered_tokens]
print("\nStemming:")
print(stemmed_tokens)

# Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [wordnet_lemmatizer.lemmatize(token) for token in filtered_tokens]
print("\nLemmatization:")
print(lemmatized_tokens)

# Part of Speech Tagging
pos_tokens = pos_tag(filtered_tokens)
print("\nPart of Speech Tagging:")
print(pos_tokens)

