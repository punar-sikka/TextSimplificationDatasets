import sys
import os
import numpy as np
import codecs
import logging
#from SARI import SARIsent
from nltk.translate.bleu_score import *
smooth = SmoothingFunction()
from nltk import word_tokenize
from textstat.textstat import textstat
import Levenshtein
import nltk
from nltk.tokenize import RegexpTokenizer
import syllables_en

nltk.download('punkt')

TOKENIZER = RegexpTokenizer('(?u)\W+|\$[\d\.]+|\S+')
SPECIAL_CHARS = ['.', ',', '!', '?']
logging.basicConfig(format = u'[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', level = logging.NOTSET)


def ReadInFile (filename):
    
    with open(filename) as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    return lines


def SARIngram(sgrams, cgrams, rgramslist, numref):


    rgramsall = [rgram for rgrams in rgramslist for rgram in rgrams]
    rgramcounter = Counter(rgramsall)
    
    sgramcounter = Counter(sgrams)
    sgramcounter_rep = Counter()
    for sgram, scount in sgramcounter.items():
        sgramcounter_rep[sgram] = scount * numref
        
    cgramcounter = Counter(cgrams)
    cgramcounter_rep = Counter()
    for cgram, ccount in cgramcounter.items():
        cgramcounter_rep[cgram] = ccount * numref
    
    
    # KEEP
    keepgramcounter_rep = sgramcounter_rep & cgramcounter_rep
    keepgramcountergood_rep = keepgramcounter_rep & rgramcounter
    keepgramcounterall_rep = sgramcounter_rep & rgramcounter

    keeptmpscore1 = 0
    keeptmpscore2 = 0
    for keepgram in keepgramcountergood_rep:
        keeptmpscore1 += keepgramcountergood_rep[keepgram] / keepgramcounter_rep[keepgram]
        keeptmpscore2 += keepgramcountergood_rep[keepgram] / keepgramcounterall_rep[keepgram]
        #print "KEEP", keepgram, keepscore, cgramcounter[keepgram], sgramcounter[keepgram], rgramcounter[keepgram]
    keepscore_precision = 0
    if len(keepgramcounter_rep) > 0:
        keepscore_precision = keeptmpscore1 / len(keepgramcounter_rep)
    keepscore_recall = 0
    if len(keepgramcounterall_rep) > 0:
        keepscore_recall = keeptmpscore2 / len(keepgramcounterall_rep)
    keepscore = 0
    if keepscore_precision > 0 or keepscore_recall > 0:
        keepscore = 2 * keepscore_precision * keepscore_recall / (keepscore_precision + keepscore_recall)


    # DELETION
    delgramcounter_rep = sgramcounter_rep - cgramcounter_rep
    delgramcountergood_rep = delgramcounter_rep - rgramcounter
    delgramcounterall_rep = sgramcounter_rep - rgramcounter

    deltmpscore1 = 0
    deltmpscore2 = 0
    for delgram in delgramcountergood_rep:
        deltmpscore1 += delgramcountergood_rep[delgram] / delgramcounter_rep[delgram]
        deltmpscore2 += delgramcountergood_rep[delgram] / delgramcounterall_rep[delgram]
    delscore_precision = 0
    if len(delgramcounter_rep) > 0:
        delscore_precision = deltmpscore1 / len(delgramcounter_rep)
    delscore_recall = 0
    if len(delgramcounterall_rep) > 0:
        delscore_recall = deltmpscore1 / len(delgramcounterall_rep)
    delscore = 0
    if delscore_precision > 0 or delscore_recall > 0:
        delscore = 2 * delscore_precision * delscore_recall / (delscore_precision + delscore_recall)


    # ADDITION
    addgramcounter = set(cgramcounter) - set(sgramcounter)
    addgramcountergood = set(addgramcounter) & set(rgramcounter)
    addgramcounterall = set(rgramcounter) - set(sgramcounter)

    addtmpscore = 0
    for addgram in addgramcountergood:
        addtmpscore += 1

    addscore_precision = 0
    addscore_recall = 0
    if len(addgramcounter) > 0:
        addscore_precision = addtmpscore / len(addgramcounter)
    if len(addgramcounterall) > 0:
        addscore_recall = addtmpscore / len(addgramcounterall)
    addscore = 0
    if addscore_precision > 0 or addscore_recall > 0:
        addscore = 2 * addscore_precision * addscore_recall / (addscore_precision + addscore_recall)
    

    return (keepscore, delscore_precision, addscore)
    

def SARIsent (ssent, csent, rsents) :
    numref = len(rsents)    

    s1grams = ssent.lower().split(" ")
    c1grams = csent.lower().split(" ")
    s2grams = []
    c2grams = []
    s3grams = []
    c3grams = []
    s4grams = []
    c4grams = []
 
    r1gramslist = []
    r2gramslist = []
    r3gramslist = []
    r4gramslist = []
    for rsent in rsents:
        r1grams = rsent.lower().split(" ")    
        r2grams = []
        r3grams = []
        r4grams = []
        r1gramslist.append(r1grams)
        for i in range(0, len(r1grams)-1) :
            if i < len(r1grams) - 1:
                r2gram = r1grams[i] + " " + r1grams[i+1]
                r2grams.append(r2gram)
            if i < len(r1grams)-2:
                r3gram = r1grams[i] + " " + r1grams[i+1] + " " + r1grams[i+2]
                r3grams.append(r3gram)
            if i < len(r1grams)-3:
                r4gram = r1grams[i] + " " + r1grams[i+1] + " " + r1grams[i+2] + " " + r1grams[i+3]
                r4grams.append(r4gram)        
        r2gramslist.append(r2grams)
        r3gramslist.append(r3grams)
        r4gramslist.append(r4grams)
       
    for i in range(0, len(s1grams)-1) :
        if i < len(s1grams) - 1:
            s2gram = s1grams[i] + " " + s1grams[i+1]
            s2grams.append(s2gram)
        if i < len(s1grams)-2:
            s3gram = s1grams[i] + " " + s1grams[i+1] + " " + s1grams[i+2]
            s3grams.append(s3gram)
        if i < len(s1grams)-3:
            s4gram = s1grams[i] + " " + s1grams[i+1] + " " + s1grams[i+2] + " " + s1grams[i+3]
            s4grams.append(s4gram)
            
    for i in range(0, len(c1grams)-1) :
        if i < len(c1grams) - 1:
            c2gram = c1grams[i] + " " + c1grams[i+1]
            c2grams.append(c2gram)
        if i < len(c1grams)-2:
            c3gram = c1grams[i] + " " + c1grams[i+1] + " " + c1grams[i+2]
            c3grams.append(c3gram)
        if i < len(c1grams)-3:
            c4gram = c1grams[i] + " " + c1grams[i+1] + " " + c1grams[i+2] + " " + c1grams[i+3]
            c4grams.append(c4gram)

    
    (keep1score, del1score, add1score) = SARIngram(s1grams, c1grams, r1gramslist, numref)
    (keep2score, del2score, add2score) = SARIngram(s2grams, c2grams, r2gramslist, numref)
    (keep3score, del3score, add3score) = SARIngram(s3grams, c3grams, r3gramslist, numref)
    (keep4score, del4score, add4score) = SARIngram(s4grams, c4grams, r4gramslist, numref)

    avgkeepscore = sum([keep1score,keep2score,keep3score,keep4score])/4
    avgdelscore = sum([del1score,del2score,del3score,del4score])/4
    avgaddscore = sum([add1score,add2score,add3score,add4score])/4
    finalscore = ( avgkeepscore + avgdelscore + avgaddscore ) / 3

    return finalscore
def get_words(text=''):
    words = TOKENIZER.tokenize(text)
    filtered_words = []
    for word in words:
        if word in SPECIAL_CHARS or word == " ":
            pass
        else:
            new_word = word.replace(",","").replace(".","")
            new_word = new_word.replace("!","").replace("?","")
            filtered_words.append(new_word)
    return filtered_words

def get_sentences(text=''):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(text)
    return sentences

def count_syllables(words):
    syllableCount = 0
    for word in words:
        syllableCount += syllables_en.count(word)
    return syllableCount

def files_in_folder(mypath):
    return [ os.path.join(mypath,f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath,f)) ]

def folders_in_folder(mypath):
    return [ os.path.join(mypath,f) for f in os.listdir(mypath) if os.path.isdir(os.path.join(mypath,f)) ]

def files_in_folder_only(mypath):
    return [ f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath,f)) ]

def remove_features(sent):
    tokens = sent.split(" ")
    return " ".join([token.split("|")[0] for token in tokens])

def remove_underscores(sent):
    return sent.replace("_", " ")

def replace_parant(sent):
    sent = sent.replace("-lrb-", "(").replace("-rrb-", ")")
    return sent.replace("(", "-lrb-").replace(")", "-rrb-")

def lowstrip(sent):
    return sent.lower().strip()

def normalize(sent):
    return replace_parant(lowstrip(sent))

def as_is(sent):
    return sent

def get_hypothesis(filename):
    hypothesis = '-'
    if "_h1" in filename:
        hypothesis = '1'
    elif "_h2" in filename:
        hypothesis = '2'
    elif "_h3" in filename:
        hypothesis = '3'
    elif "_h4" in filename:
        hypothesis = '4'
    return hypothesis

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

def print_scores(pairs, whichone = ''):
    # replace filenames by hypothesis name for csv pretty print
    for k,v in pairs:
        hypothesis = get_hypothesis(k)
        print("\t".join( [whichone, "{:10.2f}".format(v), k, hypothesis] ))

def SARI_file(source, preds, refs, preprocess):
    files = [codecs.open(fis, "r", 'utf-8') for fis in [source, preds, refs]]
    scores = []
    for src, pred, ref in zip(*files):
        references = [preprocess(r) for r in ref.split('\t')]
        scores.append(SARIsent(preprocess(src), preprocess(pred), references))
    for fis in files:
        fis.close()
    return mean(scores)


# BLEU doesn't need the source
def BLEU_file(source, preds, refs, preprocess=as_is):
    files = [codecs.open(fis, "r", 'utf-8') for fis in [preds, refs]]
    scores = []
    references = []
    hypothese = []
    for pred, ref in zip(*files):
        references.append([word_tokenize(preprocess(r)) for r in ref.split('\t')])
        hypothese.append(word_tokenize(preprocess(pred)))
    for fis in files:
        fis.close()
    # Smoothing method 3: NIST geometric sequence smoothing
    return corpus_bleu(references, hypothese, smoothing_function=smooth.method3)


def worddiff_file(source, preds, refs, preprocess):
    files = [codecs.open(fis, "r", 'utf-8') for fis in [source, preds]]
    worddiff = 0
    n = 0
    for src, pred in zip(*files):
        source = word_tokenize(preprocess(src))
        hypothese = word_tokenize(preprocess(pred))
        n += 1
        worddiff += len(source) - len(hypothese)

    worddiff /= float(n)
    for fis in files:
        fis.close()

    return worddiff / 100.0


def IsSame_file(source, preds, refs, preprocess):
    files = [codecs.open(fis, "r", 'utf-8') for fis in [source, preds]]
    issame = 0
    n = 0.
    for src, pred in zip(*files):
        source = preprocess(src)
        hypothese = preprocess(pred)
        n += 1
        issame += source == hypothese

    issame /= n
    for fis in files:
        fis.close()

    return issame / 100.0


def FKGL_file(source, preds, refs, preprocess):
    files = [codecs.open(fis, "r", 'utf-8') for fis in [source, preds]]
    score = 0
    n = 0.
    for src, pred in zip(*files):
        hypothese = preprocess(pred)
        words = get_words(hypothese)
        word_count = float(len(words))
        sentence_count = float(len(get_sentences(hypothese)))
        syllable_count = float(count_syllables(words))
        score += 0.39 * (word_count / sentence_count) + 11.8 * (syllable_count / word_count) - 15.59
        n += 1

    score /= n
    for fis in files:
        fis.close()

    return round(score, 2) / 100


def FKdiff_file(source, preds, refs, preprocess):
    files = [codecs.open(fis, "r", 'utf-8') for fis in [source, preds]]
    fkdiff = 0
    n = 0.
    for src, pred in zip(*files):
        # hypothese = preprocess(pred)
        # source = preprocess(src)
        hypothese = (pred)
        source = (src)
        # print(source)
        # print(hypothese)

        fkdiff += (textstat.flesch_reading_ease(hypothese) - textstat.flesch_reading_ease(source))
        n += 1
        # fkdiff= 1/(1+np.exp(-fkdiff))

    fkdiff /= n
    for fis in files:
        fis.close()

    return fkdiff / 100.0


def LD_file(source, preds, refs, preprocess):
    files = [codecs.open(fis, "r", 'utf-8') for fis in [source, preds]]
    LD = 0
    n = 0.
    for src, pred in zip(*files):
        hypothese = preprocess(pred)
        source = preprocess(src)
        LD += Levenshtein.distance(hypothese, source)
        n += 1

    LD /= n
    for fis in files:
        fis.close()

    return LD / 100.0


def score(source, refs, fold, METRIC_file, preprocess=as_is):
    # new_files = files_in_folder(fold)
    data = []
    for fis in fold:
        # ignore log files
        if ".log" in os.path.basename(fis):
            continue
        logging.info("Processing "+os.path.basename(fis))
        val = 100*METRIC_file(source, fis, refs, preprocess)
        logging.info("Done "+str(val))
        data.append((os.path.basename(fis), val))
    data.sort(key=lambda tup: tup[1])
    data.reverse()
    return data, None


def map_to_array(score_dict):
    def get_beam_order_from_filename(filename):
        filename = filename.split('_')
        beam = int(filename[2][1:])
        hyp_order = int(filename[3][1])
        return beam, hyp_order, filename[1]

    score_arr_dict = {}
    for filename, val in score_dict:
        try:
            beam, hyp_order, subset = get_beam_order_from_filename(filename)
        except:
            beam, hyp_order, subset = 5, 1, 'test'
        if subset in score_arr_dict:
            score_arr_dict[subset][beam-5, hyp_order-1] = round(val, 2)
        else:
            score_arr_dict[subset] = np.zeros((8, 5))
            score_arr_dict[subset][beam - 5, hyp_order - 1] = round(val, 2)
    return score_arr_dict


if __name__ == '__main__':
    try:
        source = sys.argv[1]
        logging.info("Source: " + source)
        refs = sys.argv[2]
        logging.info("References in tsv format: " + refs)
        pred_path = sys.argv[3]
        logging.info("Path of predictions: " + pred_path)
        n_best = int(sys.argv[4])
    except:
        logging.error("Input parameters must be: " + sys.argv[0]
            + "    SOURCE_FILE    REFS_TSV (paste -d \"\t\" * > reference.tsv)    DIRECTORY_OF_PREDICTIONS")
        sys.exit(1)

    '''
        SARI can become very unstable to small changes in the data.
        The newsela turk references have all the parantheses replaced
        with -lrb- and -rrb-. Our output, however, contains the actual
        parantheses '(', ')', thus we prefer to apply a preprocessing
        step to normalize the text.
    '''
    preds = open(pred_path, 'r').readlines()
    fold = []
    for idx in range(n_best):
        preds_tmp = preds[idx::n_best]
        filename_tmp = pred_path+'_h{}'.format(idx+1)
        fold.append(filename_tmp)
        open(filename_tmp, 'w').write(''.join(preds_tmp))

    sari_test, sari_arr = score(source, refs, fold, SARI_file, normalize)
    bleu_test, bleu_arr = score(source, refs, fold, BLEU_file, lowstrip)
    print(f"SARI : {sari_test}")
    print(f"BLEU : {bleu_test}")
