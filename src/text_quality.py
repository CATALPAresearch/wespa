import textstat
from language_tool_python import LanguageTool
import textdescriptives as td
import spacy
from collections import Counter
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sentence_transformers import SentenceTransformer, util
from collections import Counter
from nltk import word_tokenize


class Text_Quality:

    def __init__(self):
        self.language = 'de'
        self.language_country = 'de-DE' #'en-US'
        self.nlp = spacy.load('de_dep_news_trf')# en_core_web_sm, de_core_news_sm, de_dep_news_trf
        self.nlp.add_pipe('textdescriptives/all')
        #nltk.download('punkt_tab')
        #nltk.download('punkt')
        
        

    def compute_quantitative_measures(self, text):
        doc = self.nlp(text)
        doc._.sentence_length # mean, median, ...
        #print(doc._.quality.n_stop_words)
        
        return {
            "word_count": Counter(word_tokenize(text)),
            "syllable_count": textstat.syllable_count(text),
            "sentence_count": textstat.sentence_count(text),
            "char_count": textstat.char_count(text, ignore_spaces=True),
            "letter_count": textstat.letter_count(text, ignore_spaces=True),
            #"proportions": doc._.pos_proportions, # proportion of adjectives, nouns, verbs, ...
            #"stop_words": doc._.n_stop_words, # The number of stop words in the document.
            #"bullet_points": doc._.proportion_bullet_points # Proportion of lines in a documents which start with a bullet point.
        }

    def compute_readability(self, text):
        """Evaluates how easily a reader can understand the text, often using metrics like the Flesch-Kincaid Grade Level."""
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
        return flesch_kincaid_grade
    
    def compute_gramaticality(self, text):
        """Assesses the correctness of grammar in the text."""
        tool = LanguageTool(self.language_country)
        matches = tool.check(text)
        grammar_errors = len(matches)
        return grammar_errors
    
    def compute_coherence(self, text):
        """Evaluates the logical flow and connectivity between sentences and paragraphs."""
        doc = self.nlp(text)
        coherence = doc._.coherence
        return coherence

    def compute_lexic_diversity(self, text):
        """Assesses the range of vocabulary used in the text."""
        nltk.download('punkt')
        words = nltk.word_tokenize(text)
        unique_words = set(words)
        lexical_diversity = len(unique_words) / len(words)
        return lexical_diversity
    
    def compute_text_complexity(self, text):
        """Analyzes the complexity of the text, including syntactic and semantic aspects."""
        complexity = textstat.text_standard(text)
        return complexity
    
    def compute_consistency(self, text):
        """Evaluates how consistently the text adheres to its main topics."""
        vectorizer = TfidfVectorizer(stop_words=self.language)
        X = vectorizer.fit_transform(text)
        nmf = NMF(n_components=1, random_state=1)
        nmf.fit(X)
        topic_words = nmf.components_
        return topic_words
    
    def compute_semantic_similarity(self, text_student, text_assignment):
        """Evaluate how similar the student text is to assignment text"""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        sentences = [text_student, text_assignment]
        embeddings = model.encode(sentences, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        return similarity
    
    def compute_information_desity(self, text):
        """Evaluates the amount of information conveyed in the text relative to its length."""
        words = nltk.word_tokenize(text)
        information_density = len(set(words)) / len(words)
        return information_density
    
    def compute_duplicate_lines(self, text):
        """Duplicate lines character fraction: Fraction of characters in a document which are contained within duplicate lines."""
        doc = self.nlp(text)
        duplicate_line = doc._.duplicate_lines_chr_fraction()
        return duplicate_line

    def compute_duplicate_paragraphs(self, text):
        """Duplicate paragraphs character fraction: Fraction of characters in a document which are contained within duplicate paragraphs."""
        doc = self.nlp(text)
        duplicate_paragraphs = doc._.duplicate_paragraphs_chr_fraction()
        return duplicate_paragraphs
        
    ##
    """https://hlasse.github.io/TextDescriptives/dependencydistance.html"""


    def run (self, text):
        """..."""
        self.compute_quantitative_measures(text)
        
        #self.compute_gramaticality(text)
        #self.compute_duplicate_lines(text)
        #self.compute_duplicate_paragraphs(text)
    
        #self.compute_readability(text)
        
        #self.compute_lexic_diversity(text)
        #self.compute_text_complexity(text)

        #self.compute_coherence(text)
        #self.compute_consistency(text)
        #self.compute_information_desity(text)
        
        #self.compute_semantic_similarity(text)
        
        
if __name__ == '__main__':
    # install model: python -m spacy download de_dep_news_trf
    tq = Text_Quality()
    text = "Das Leben ist schön"
    #tq.run(text)
    print(tq.compute_quantitative_measures(text))
