from collections import Counter
import json
import re
import textstat
from language_tool_python import LanguageTool
import textdescriptives as td
import spacy
from spacy.lang.de import German
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk import word_tokenize
from nltk.corpus import wordnet
import pandas as pd



class Text_Quality:

    def __init__(self):
        self.language = 'de' # de?
        self.language_long = 'english' # de?
        self.language_country = 'de-DE' #'en-US'
        self.nlp = spacy.load('de_dep_news_trf')# en_core_web_sm, de_core_news_sm, de_dep_news_trf
        self.nlp.add_pipe('textdescriptives/all')
        # FixMe Check whether the following nltk files have been downloaded already
        #nltk.download('punkt_tab')
        #nltk.download('punkt')
        #nltk.download('wordnet')
        
        
    def compute_quantitative_measures(self, corpus):
        """        
        ADJ: adjective, e.g. big, old, green, incomprehensible, first
        ADP: adposition, e.g. in, to, during
        ADV: adverb, e.g. very, tomorrow, down, where, there
        AUX: auxiliary, e.g. is, has (done), will (do), should (do)
        CONJ: conjunction, e.g. and, or, but
        CCONJ: coordinating conjunction, e.g. and, or, but
        DET: determiner, e.g. a, an, the
        INTJ: interjection, e.g. psst, ouch, bravo, hello
        NOUN: noun, e.g. girl, cat, tree, air, beauty
        NUM: numeral, e.g. 1, 2017, one, seventy-seven, IV, MMXIV
        PART: particle, e.g. ’s, not,
        PRON: pronoun, e.g I, you, he, she, myself, themselves, somebody
        PROPN: proper noun, e.g. Mary, John, London, NATO, HBO
        PUNCT: punctuation, e.g. ., (, ), ?
        SCONJ: subordinating conjunction, e.g. if, while, that
        SYM: symbol, e.g. $, %, §, ©, +, −, ×, ÷, =, :), 😝
        VERB: verb, e.g. run, runs, running, eat, ate, eating
        X: other, e.g. sfpksdpsxmsa
        SPACE: space, e.g.
        """
        text = " ".join(corpus)
        nlp = spacy.load("de_core_news_sm")
        nlp.add_pipe("textdescriptives/all")
        doc = nlp(text)
        # doc._.sentence_length # mean, median, ...
        
        number_of_words = 0
        counter = Counter(word_tokenize(text))
        for t in counter:
            number_of_words = number_of_words + counter[t] 
        
        number_of_lines = len(text.split('/n'))

        simple_measures = {
            "char_count": textstat.char_count(text, ignore_spaces=False),
            "letter_count": textstat.letter_count(text, ignore_spaces=True),
            "syllable_count": textstat.syllable_count(text),
            "word_count": number_of_words,
            "sentence_count": textstat.sentence_count(text),
            #"sentence_length_mean": sentence_length_mean" # TODO
            #"sentence_length_median": sentence_length_median" # TODO
            "line_count": number_of_lines,
            #"paragraph_count": paragraph_count" # TODO
            "stop_words": doc._.quality.n_stop_words.value / number_of_words,
            "bullet_points": doc._.quality.proportion_bullet_points.value / number_of_lines
            }
        
        pos_tags = doc._.pos_proportions
        pos_tags['proportion_adjective'] = pos_tags.pop('pos_prop_ADJ')
        pos_tags['proportion_adposition'] = pos_tags.pop('pos_prop_ADP')
        pos_tags['proportion_adverbs'] = pos_tags.pop('pos_prop_ADV')
        pos_tags['proportion_adjective'] = pos_tags.pop('pos_prop_AUX')
        pos_tags['proportion_conjunctions'] = pos_tags.pop('pos_prop_CCONJ')
        pos_tags['proportion_determiner'] = pos_tags.pop('pos_prop_DET')
        pos_tags['proportion_interjection'] = pos_tags.pop('pos_prop_INTJ')
        pos_tags['proportion_noun'] = pos_tags.pop('pos_prop_NOUN')
        pos_tags['proportion_numeral'] = pos_tags.pop('pos_prop_NUM')
        pos_tags['proportion_particle'] = pos_tags.pop('pos_prop_PART')
        pos_tags['proportion_pronoun'] = pos_tags.pop('pos_prop_PRON')
        pos_tags['proportion_proper_noun'] = pos_tags.pop('pos_prop_PROPN')
        pos_tags['proportion_punctuation'] = pos_tags.pop('pos_prop_PUNCT')
        pos_tags['proportion_subordinating_conjunction'] = pos_tags.pop('pos_prop_SCONJ')
        pos_tags['proportion_symbol'] = pos_tags.pop('pos_prop_SYM')
        pos_tags['proportion_verb'] = pos_tags.pop('pos_prop_VERB')
        pos_tags['proportion_others'] = pos_tags.pop('pos_prop_X')

        return simple_measures | pos_tags
        

    def compute_readability(self, text):
        """Evaluates how easily a reader can understand the text, often using metrics like the Flesch-Kincaid Grade Level."""
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
        return {"readbility": flesch_kincaid_grade} #TODO: normalise 0..1
    
    def compute_gramaticality(self, text):
        """Assesses the correctness of grammar in the text."""
        tool = LanguageTool(self.language_country)
        matches = tool.check(text)
        grammar_errors = len(matches)
        return {"grammar_errors": grammar_errors}
    
    def compute_coherence(self, text): #TODO/Remove
        """Evaluates the logical flow and connectivity between sentences and paragraphs."""
        doc = self.nlp(text)
        coherence = doc._.coherence
        return coherence


    def compute_lexic_diversity(self, text):
        """Assesses the range of vocabulary used in the text."""
        #nltk.download('punkt')
        words = nltk.word_tokenize(text)
        unique_words = set(words)
        lexical_diversity = len(unique_words) / len(words)
        return {"lexical_diversity": lexical_diversity}
    

    def compute_text_complexity(self, text):
        """Analyzes the complexity of the text, including syntactic and semantic aspects."""
        school_grade_level = textstat.text_standard(text) # en only
        school_grade_score = textstat.dale_chall_readability_score(text) / 9.9
        return {"text_complexity": school_grade_score}
    

    def compute_consistency(self, corpus): #todo/remove
        """Evaluates how consistently the text adheres to its main topics."""
        vectorizer = TfidfVectorizer(stop_words=self.language_long)
        X = vectorizer.fit_transform(corpus)
        nmf = NMF(n_components=1, random_state=1)
        nmf.fit(X)
        topic_words = nmf.components_
        return topic_words
    

    def compute_semantic_similarity(self, text_student, text_assignment): #todo
        """Evaluate how similar the student text is to assignment text"""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        sentences = [text_student, text_assignment]
        embeddings = model.encode(sentences, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        return similarity
    

    def compute_information_desity(self, text): #TODO duplicate?
        """Evaluates the amount of information conveyed in the text relative to its length."""
        words = nltk.word_tokenize(text)
        information_density = len(set(words)) / len(words)
        return information_density
    

    def compute_duplicate_lines(self, text):
        """Duplicate lines character fraction: Fraction of characters in a document which are contained within duplicate lines."""
        nlp = German()
        nlp.add_pipe('sentencizer')
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        unique_sentences = set(sentences)
        return {"duplicate_lines": len(unique_sentences)/len(sentences)} 


    def compute_duplicate_paragraphs(self, text): #TODO
        """Duplicate paragraphs character fraction: Fraction of characters in a document which are contained within duplicate paragraphs."""
        doc = self.nlp(text)
        duplicate_paragraphs = doc._.duplicate_paragraphs_chr_fraction()
        return duplicate_paragraphs
    

    def compute_transitional_words(self, text):
        """Find all transitional word. These worde indicate a strong cohesion and thus argumentation"""
        transitional_words = [
            "weil", "da", "denn", "deshalb", "deswegen", "darum", "daher", "folglich", "infolgedessen", "aus diesem Grund", "somit", "also", "dadurch", "demnach", "indessen", "indes", "mittlerweile", "unterdessen", "inzwischen", "trotzdem", "dennoch", "allerdings", "immerhin", "nichtsdestotrotz", "jedoch", "wohingegen", "andererseits", "auf der einen Seite", "auf der anderen Seite", "einerseits", "andererseits", "währenddessen", "während", "zudem", "außerdem", "ferner", "obendrein", "überdies", "überdies hinaus", "nicht nur", "sondern auch", "sowohl als auch", "ebenso", "gleichermaßen", "vergleichsweise", "analog dazu", "ähnlich", "entsprechend", "im Vergleich dazu", "genauso", "insbesondere", "vor allem", "namentlich", "nämlich", "explizit", "das heißt", "sprich", "mit anderen Worten", "beziehungsweise", "genauer gesagt", "insbesondere", "speziell", "beispielweise", "etwa", "zum Beispiel", "unter anderem", "so etwa", "so zum Beispiel"," wie etwa", "vornehmlich", "in erster Linie", "vorrangig", "erstens", "zweitens", "drittens", "abschließend", "letztendlich", "schließlich", "zuletzt", "alles in allem", "zusammenfassend", "kurz gesagt", "resümierend", "folgerichtig", "abschließend betrachtet", "unter dem Strich", "zu guter Letzt", "abschließend", "schließlich", "letztendlich", "um es zusammenzufassen", "kurzum", "mit einem Wort", "beispielsweise", "z.B.", "z. B.", "bspw."
        ]
        pattern = re.compile(r'\b(?:' + '|'.join(transitional_words) + r')\b', re.IGNORECASE)
        matches = pattern.findall(text)
        return {"transitional_words": len(matches)} #TODO normalise 0..1


    def compute_x():#TODO
        """https://hlasse.github.io/TextDescriptives/dependencydistance.html"""
        pass
   

    def get_noun_roles(self, doc):
        """Extract nouns and classify them as subject, object, or other."""
        nouns = []
        for token in doc:
            if token.pos_ == "NOUN":
                role = "other"
                if "subj" in token.dep_:
                    role = "subject"
                elif "obj" in token.dep_:
                    role = "object"
                nouns.append((token.text.lower(), role))
        return nouns

    def get_synonyms(self, word):
        """Retrieve synonyms for a given word from WordNet."""
        synonyms = set()
        for syn in wordnet.synsets(word, pos=wordnet.NOUN):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace("_", " "))
        return synonyms

    def build_lexical_chains(self, nouns):
        """Create lexical chains using explicit and implicit relations (synonyms)."""
        chains = []
        word_to_chain = {}

        for noun, role in nouns:
            added = False
            # Check if word belongs to an existing chain
            for chain in chains:
                chain_words = set(word for word, _ in chain)
                if noun in chain_words:
                    chain.append((noun, role))
                    word_to_chain[noun] = chain
                    added = True
                    break

                # Check for implicit synonym match
                for word in chain_words:
                    if noun in self.get_synonyms(word) or word in self.get_synonyms(noun):
                        chain.append((noun, role))
                        word_to_chain[noun] = chain
                        added = True
                        break

            # If no matching chain is found, create a new one
            if not added:
                new_chain = [(noun, role)]
                chains.append(new_chain)
                word_to_chain[noun] = new_chain

        return chains

    def compute_lexical_chains(self, text):
        """Pipeline to extract lexical chains from a text."""
        nlp = spacy.load("de_core_news_sm")
        doc = nlp(text)
        nouns = self.get_noun_roles(doc)
        chains = self.build_lexical_chains(nouns)

        # Print results
        chain_length = list()
        for i, chain in enumerate(chains, 1):
            words = [word for word, role in chain]
            roles = [role for _, role in chain]
            if len(chain)>1:
                print(f"Chain {i}: {words} (Length: {len(chain)})")
                print(f"Roles: {roles}")
                chain_length.append(len(chain))
        return {
            "lexical_chains": len(chain_length), 
            "avg_chain_lenght": sum(chain_length)/len(chain_length) 
            }
    
    

    def run (self, corpus):
        """Compute and combine all indicators for a given text"""
        text = '\n'.join(corpus)
        
        # descriptive indicators
        qm = self.compute_quantitative_measures(corpus) 

        # indicators about the correctness
        gc = self.compute_gramaticality(text) 
        dl = self.compute_duplicate_lines(text)
        #self.compute_duplicate_paragraphs(text)
    
        # indicators about the style
        rb = self.compute_readability(text)
        
        # indicators for language use
        ld = self.compute_lexic_diversity(text)
        tc = self.compute_text_complexity(text)

        # indicators for argumentation
        tw = self.compute_transitional_words(text)
        lc = self.compute_lexical_chains(text)

        #self.compute_coherence(text)
        #self.compute_consistency(text) 
        #self.compute_information_desity(text)
        
        #self.compute_semantic_similarity(text)

        res = tc | ld | rb| dl  | gc | qm | tw | lc
        return res

        

if __name__ == '__main__':
    # install model: python -m spacy download de_dep_news_trf
    tq = Text_Quality()
    text = "Das Leben ist schön, wunderschön"
    corpus = [
        "In Hamburg lebten zwei Ameisen,",
        "Die wollten nach Australien reisen.",
        "Und leistet dann recht gern Verzicht.",
        "Bei Altona auf der Chaussee",
        "Da taten ihnen die Beine weh,",
        "Und da verzichteten sie weise",
        "Denn auf den letzten Teil der Reise.",
        "So will man oft und kann doch nicht.",
        "Und leistet dann recht gern Verzicht.",
    ]
    test_data = (
        "Playing games has always been thought to be important to "
        "the development of well-balanced and creative children; "
        "however, what part, if any, they should play in the lives "
        "of adults has never been researched that deeply. I believe "
        "that playing games is every bit as important for adults "
        "as for children. Not only is taking time out to play games "
        "with our children and other adults valuable to building "
        "interpersonal relationships but is also a wonderful way "
        "to release built up tension."
        "to release built up tension."
        )
    text2 = "Welt ist schlecht"
    
    result = tq.run(corpus)
    print(json.dumps(result, indent=4, sort_keys=True))

    text = '\n'.join(corpus)

    #print(tq.compute_lexical_chains(text))  # DONE
    #print(tq.compute_transitional_words(text)) # DONE

    #print(tq.compute_quantitative_measures(corpus)) # DONE
    #print(tq.compute_gramaticality(text)) # DONE
    #print(tq.compute_duplicate_lines(text)) # DONE
    #print(tq.compute_duplicate_paragraphs(text))
    
    #print(tq.compute_readability(text)) # DONE
        
    #print(tq.compute_lexic_diversity(text)) # DONE
    #print(tq.compute_text_complexity(text)) # DONE

    ##print(tq.compute_coherence(text)) # nan
    ##print(tq.compute_consistency(corpus)) # # https://medium.com/towards-data-science/let-us-extract-some-topics-from-text-data-part-iii-non-negative-matrix-factorization-nmf-8eba8c8edada
    ###print(tq.compute_information_desity(text))
        
    ##print(tq.compute_semantic_similarity(text, text2))
