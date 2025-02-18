from collections import Counter
import json
import re
from pathlib import Path
import textstat
from language_tool_python import LanguageTool
import textdescriptives as td
import spacy
from spacy.lang.de import German
import de_core_news_sm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk import word_tokenize
from nltk.corpus import wordnet
import pandas as pd

from .extract_easy_sync import Extract_Easy_Sync
from .settings import *

class Preprocess_Text_Quality:

    def __init__(self, semester):
        self.semester = semester
        self.tq = Text_Quality()
        self.period_split_interval = 0

    def split_text_progression_by_threshold(self, all_text, semester, start_date, end_date, period_split_interval = 'days', group_column="group_id"):
        """..."""
        es = Extract_Easy_Sync(semester, None)
        date_thresholds = es.generate_observation_times(start_date, end_date, period_split_interval)
        #print(date_thresholds)
        
        df1 = pd.DataFrame(all_text)

        df1 = df1.dropna(axis=0)
        df1['timestamp'] = df1['timestamp'].astype(float)
        #df1[group_column] = df1[group_column].astype(int)
        df1 = df1.sort_values(by=[group_column, 'timestamp'])

        # Initialize the new column with False
        df1['is_first_above_threshold'] = False
        df1['period_split_interval'] = period_split_interval
        
        # Iterate through groups and check for the first occurrence above any threshold
        for index, group_df in df1.groupby(group_column):
            for threshold in sorted(date_thresholds):
                #print(index, threshold)
                first_above = group_df[group_df['timestamp'] < threshold.astype(float)] #.sort_values(by=['timestamp'])
                if first_above.shape[0] > 0:
                    #for item in first_above.itertuples(index=True):
                    #print(first_above.index.max())
                    df1.loc[first_above.index.max(), 'is_first_above_threshold'] = True
        
        time_breaks = df1[df1['is_first_above_threshold']==True]
        time_breaks = time_breaks[[
            'id', 
            'moodle_group_id', 
            #'moodle_author_id', 
            'moodle_pad_id', 
            'timestamp']
            ]
        return time_breaks


    def determine_text_quality_from_csv(self, df_text_raw):
        """Deprecated"""
        test = self.tq.run('test test test')
        text_quality_results = pd.DataFrame(columns=['group_id', 'pad_id', 'timestamp', 'period_split_interval'] + [(k) for k, v in test.items()] + ['text'])
        i=0
        for row in df_text_raw.itertuples():
            if(row.text != '' or row.text== None):
                qs_result = self.tq.run(row.text) # FixMe .replace('\n','')
                text_quality_results.loc[i] = [row.group_id] + [row.pad_id] + [row.timestamp] + [row.period_split_interval] + [(v) for k, v in qs_result.items()]  + [row.text.replace('\n',' ')]
                i=i+1

        text_quality_results.to_csv(
                    f'{output_path}/{project_name}-{self.semester}-02.2xxx-text-quality', 
                    index=False,
                    quotechar='"'
                    )
        return text_quality_results
    

    def determine_text_quality_from_files(self, period_split_interval='days'):
        """Optimized function for processing text files"""
        
        self.period_split_interval = period_split_interval
        print('Processing text quality from files...')

        test_keys = list(self.tq.run('test test test').keys())
        columns = ['group_id', 'pad_id', 'timestamp', 'period_split_interval'] + test_keys + ['text']

        results = []  # Store results before DataFrame conversion

        folder = Path(f'{output_path}text/')  
        for txt_file in folder.glob("*.txt"):
            with txt_file.open("r", encoding="utf-8") as file:
                content = file.read().replace('\n', ' ')  # Read and clean text
                
                split_file_name = txt_file.stem.split('-')  # No need for `.name.split()`
                if len(split_file_name) > 4:
                    semester, group_id, pad_id, timestamp = split_file_name[1:5]

                    if self.semester == semester:
                        text_quality = self.tq.run(content)
                        results.append([group_id, pad_id, timestamp, period_split_interval] + list(text_quality.values()) + [content])

        # Convert to DataFrame in one step
        text_quality_results = pd.DataFrame(results, columns=columns)

        # Save to CSV
        self.save_data(text_quality_results, 'text-features.csv')

        return text_quality_results



    def determine_text_quality_from_files_old(self, period_split_interval='days'):
        """..."""
        self.period_split_interval = period_split_interval
        print('determine_text_quality_from_files')
        test = self.tq.run('test test test')
        text_quality_results = pd.DataFrame(columns=['group_id', 'pad_id', 'timestamp', 'period_split_interval'] + [(k) for k, v in test.items()] + ['text'])
        
        folder = Path(f'{output_path}text/')  # Convert to Path object
        i=0
        for txt_file in folder.glob("*.txt"):
            with txt_file.open("r", encoding="utf-8") as file:
                content = file.read()  # Read the file content
                
                split_file_name = txt_file.name.split('-')
                if len(split_file_name) > 3:
                    semester = split_file_name[1]
                    group_id = split_file_name[2]
                    pad_id = split_file_name[3]
                    timestamp = split_file_name[4]
                    period_split_interval = self.period_split_interval

                    qs_result = self.tq.run(content)
                    
                    if self.semester == semester:
                        #text = content.replace('\n',' ')
                        text = ''
                        text_quality_results.loc[i] = [group_id] + [pad_id] + [timestamp] + [self.period_split_interval] + [(v) for k, v in qs_result.items()] + [text]
                        i=i+1

        self.save_data(text_quality_results, 'text-features.csv')
        return text_quality_results
    

    def save_data(self, df, filename):
        file_path = f'{output_path}/{project_name}-{self.semester}-{self.period_split_interval}-08-{filename}'
        df.to_csv(
            file_path,
            index=False,
            mode='a',
            quotechar='"',
            header=not os.path.exists(file_path)
        )


class Text_Quality:
    """
    ...
    """
    def __init__(self):
        self.language = 'de' # de?
        self.language_long = 'english' # de?
        self.language_country = 'de-DE' #'en-US'
        
        # FixMe Check whether the following nltk files have been downloaded already
        #nltk.download('punkt_tab')
        #nltk.download('punkt')
        #nltk.download('wordnet')
    
    def init_spacy(self, text, model='de_core_news_md'):
        #nlp = spacy.load(model)#de_dep_news_trf de_core_news_md ,en_core_web_md, de_core_news_sm, de_dep_news_trf
        nlp = de_core_news_sm.load()
        nlp.max_length = 3000000
        nlp.add_pipe('textdescriptives/all')
        self.doc = nlp(text)

        
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
        #nlp = spacy.load("de_core_news_sm")
        #self.nlp.add_pipe("textdescriptives/all")
        #doc = self.nlp(text)#, disable = ['ner', 'parser'])
        # doc._.sentence_length # mean, median, ...
        
        number_of_words = 0
        counter = Counter(word_tokenize(text))
        for t in counter:
            number_of_words = number_of_words + counter[t] 
        number_of_words = 1 if number_of_words == 0 else number_of_words

        number_of_lines = len(text.split('/n'))
        umber_of_lines = 1 if number_of_lines == 0 else number_of_lines

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
            "stop_words": self.doc._.quality.n_stop_words.value / number_of_words,
            "bullet_points": self.doc._.quality.proportion_bullet_points.value / number_of_lines
            }
        
        pos_tags = self.doc._.pos_proportions
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
        #print(pos_tags)
        return simple_measures | pos_tags
        

    def compute_readability(self, text):
        """Evaluates how easily a reader can understand the text, often using metrics like the Flesch-Kincaid Grade Level."""
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
        return {"readbility": flesch_kincaid_grade / 100} 
    
    def compute_gramaticality(self, text):
        """Assesses the correctness of grammar in the text."""
        tool = LanguageTool(self.language_country)
        matches = tool.check(text)
        grammar_errors = len(matches)
        return {"grammar_errors": grammar_errors}
    
    def compute_coherence(self, text): #TODO/Remove
        """Evaluates the logical flow and connectivity between sentences and paragraphs."""
        #doc = self.nlp(text)#, disable = ['ner', 'parser'])
        coherence = self.doc._.coherence
        return coherence


    def compute_lexic_diversity(self, text):
        """Assesses the range of vocabulary used in the text."""
        #nltk.download('punkt')
        words = nltk.word_tokenize(text)
        if len(words)==0:
            return {"lexical_diversity": 0}    
        unique_words = set(words)
        lexical_diversity = len(unique_words) / len(words)
        return {"lexical_diversity": lexical_diversity}
    

    def compute_text_complexity(self, text):
        """Analyzes the complexity of the text, including syntactic and semantic aspects."""
        school_grade_level = textstat.text_standard(text) # en only
        school_grade_score = textstat.dale_chall_readability_score(text) / 10 # ? correct
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
        if len(words)==0:
            return 0
        information_density = len(set(words)) / len(words)
        return information_density
    

    def compute_duplicate_lines(self, text):
        """Duplicate lines character fraction: Fraction of characters in a document which are contained within duplicate lines."""
        nlp = German()
        nlp.max_length = 3000000
        nlp.add_pipe('sentencizer')
        doc = nlp(text)#, disable = ['ner', 'parser'])
        sentences = [sent.text.strip() for sent in doc.sents]
        if len(sentences)==0:
            return {"duplicate_lines": 0}
        if len(sentences)==1:
            return {"duplicate_lines": 1}     
        unique_sentences = set(sentences)
        return {"duplicate_lines": len(unique_sentences)/len(sentences)} 


    def compute_duplicate_paragraphs(self, text): #TODO
        """Duplicate paragraphs character fraction: Fraction of characters in a document which are contained within duplicate paragraphs."""
        #self.doc = self.nlp(text)#, disable = ['ner', 'parser'])
        duplicate_paragraphs = self.doc._.duplicate_paragraphs_chr_fraction()
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
   

    def get_noun_roles(self, document):
        """Extract nouns and classify them as subject, object, or other."""
        nouns = []
        for token in document:
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
        #nlp = spacy.load("de_core_news_sm")
        #doc = self.nlp(text)#, disable = ['ner', 'parser'])
        nouns = self.get_noun_roles(self.doc)
        chains = self.build_lexical_chains(nouns)

        chain_length = list()
        for i, chain in enumerate(chains, 1):
            words = [word for word, role in chain]
            roles = [role for _, role in chain]
            if len(chain)>1:
                #print(f"Chain {i}: {words} (Length: {len(chain)})")
                #print(f"Roles: {roles}")
                chain_length.append(len(chain))
        chain_length = 1 if chain_length == 0 else chain_length
        if len(chain_length)==0:
            return {
            "lexical_chains": 0, 
            "avg_chain_lenght": 0 
            } 
        return {
            "lexical_chains": len(chain_length), 
            "avg_chain_lenght": sum(chain_length)/len(chain_length) 
            }
    
    

    def run (self, corpus, model='de_core_news_md'):
        """Compute and combine all indicators for a given text"""
        text = ' '.join(corpus) # FoxMe corpus should be joined with \n but spacy breaks if linebreaks are passed. Spacy pipeline should be implemented

        self.init_spacy(text, model)
        
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

    #self.init_spacy(text)
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
