import re
import os
from nltk import PunktSentenceTokenizer

from spacy.tokens import Doc, Span, Token
import multiprocessing
import itertools
import numpy as np
from spacy.language import Language

from recap_am.controller.extract_features import set_features
from recap_am.controller.nlp import parse
from recap_am.model.config import Config


config = Config.get_instance()
lang = config["nlp"]["language"]


QUOTE_CHARS = r"[\"“”«»]"
PARENS_OPEN  = r"\("
PARENS_CLOSE = r"\)"
POINTS       = r"[.!?]"

# ↳ groupement unique pour tes abréviations
ABBR_REGEX = r"|".join([
    r"Art\.", r"Abs\.", r"u\.a\.", r"U\.a\.", r"u\.E\.", r"U\.E\.",
    r"vgl\.", r"Vgl\.", r"bzw\.", r"i\.V\.m\.", r"Buchst\.", r"d\.h\.",
    r"paras\.", r"para\.", r"No\.", r"pp\.", r"D\.R\.", r"Eur\.", r"i\.e\.",
    r"Comm\.", r"eg\.", r"cf\.", r"loc\.", r"no\.", r"cit\.", r"seq\.",
    r"R\.\s*v\.", r"v\.", r"[A-Z]\.", r"III\.", r"IV\.", r"V\.", r"VI\.",
    r"VII\.", r"VIII\.", r"IX\.", r"X\."
])
#  , r" p\.", r"J\.A\.", r"C\.O\.", r"S\.", r"Ms\.",
#     r"no\.\s*\d+", r"a\.m\.", r"p\.m\.",
#     r"Nos\.", r"Mr\.", r"Prof\.",
#     r"i\.", r"ii\.", r"PARA\.", r"R\.R\.", r"H\.P\."
ABBR_RE = re.compile(ABBR_REGEX)

def clean_text(text: str) -> str:
    """
    1) Supprime &nbsp; et espaces multiples.
    2) Remplace les points des abréviations par un token spécial ‹§dot§›
       → ils ne déclencheront pas le PunktSentenceTokenizer.
    3) Remplace les points *à l’intérieur* des paires () ou "" par ‹§dot§›.
    4) Retire ponctuations parasites, garde la casse.
    """
    text = re.sub(r"&nbsp;[a-zA-Z0-9]?", " ", text)

    # protéger abréviations
    text = ABBR_RE.sub(lambda m: m.group(0).replace(".", "§dot§"), text)

    # protéger dans guillemets
    def protect_inner(match):
        inner = match.group(0)
        return inner.replace(".", "§dot§")
    text = re.sub(f"{QUOTE_CHARS}.*?{QUOTE_CHARS}", protect_inner, text)
    text = re.sub(f"{PARENS_OPEN}.*?{PARENS_CLOSE}", protect_inner, text)

    # nettoyer divers symboles / espaces
    text = text.replace("...", ".")
    text = re.sub(r"[;]", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text

@Language.factory("pre_segment")               # ← factory = picklable
def make_pre_segment(nlp, name):
    tokenizer = PunktSentenceTokenizer()       # instancié une fois

    def pre_segment(doc: Doc) -> Doc:
        txt_clean = doc.text
        if len(txt_clean.split()) <= 3:        # trop court → rien à faire
            return doc

        sentences = tokenizer.tokenize(txt_clean)

        for sent in sentences:
            words = re.findall(r"\w+|[^\s\w]", sent)
            for i in range(len(doc) - len(words) + 1):
                if [t.text for t in doc[i:i+len(words)]] == words:
                    doc[i].is_sent_start = True
                    break                      # trouvé → passe à la phrase suivante
        return doc

    return pre_segment
parse.add_pipe("pre_segment", before="parser")    # NLTK sentence splitter

@Language.component("restore_dots")
def restore_dots(doc: Doc) -> Doc:
    with doc.retokenize() as retok:
        for tok in doc:
            if "§dot§" in tok.text:
                # spaCy ≥3.7
                if hasattr(retok, "assign"):
                    retok.assign(tok, orth=tok.text.replace("§dot§", "."))
                # spaCy ≤3.6
                else:
                    retok.merge(doc[tok.i: tok.i+1],
                                 attrs={"ORTH": tok.text.replace("§dot§", ".")})
    return doc

parse.add_pipe("restore_dots", after="pre_segment", name="restore_dots")

# def clean_text(text):
#     text = re.sub(r"&nbsp;[a-zA-Z0-9]?", "", text)
#     text = (
#         text.replace("Art.", "Artikel")
#         .replace("Abs.", "Absatz")
#         .replace("u.a.", "unter anderem")
#         .replace("U.a.", "Unter anderem")
#         .replace("u.E.", "unseres Erachtens")
#         .replace("U.E.", "Unseres Erachtens")
#         .replace("vgl.", "vergleiche")
#         .replace("Vgl.", "Vergleiche")
#         .replace("bzw.", "beziehungsweise")
#         .replace("i.V.m.", "im Vergleich mit")
#         .replace("Buchst.", "Buchstabe")
#         .replace("d.h.", "das heißt")
#         .replace("paras.", "paragraphs")  
#         .replace("No.", "No") 
#         .replace(" p.","page")
#         .replace("pp.","pp")
#         .replace("D.R.","D R")
#         .replace("Eur.","European")
#         .replace("i.e.","i e")
#         .replace("Comm.","Comm")
#         .replace("eg.", "eg")
#         .replace("cf.", "cf")
#         .replace("Court.","Court")
#         .replace("seq.","seq")
#         .replace("'", "")
#         .replace("-", " ")
#         .replace(";", "")
#     )
#     text = re.sub(r"[^a-zA-Z0-9.,?!äÄöÖüÜ:;&ß%$'\"()[\]{} -]\n", "", text)
#     text = text.replace("...", "")
#     text = re.sub(r" +", " ", text)
#     text = text.strip(" ")
#     return text

# @Language.component("pre_segment")
# def pre_segment(doc):
#     """Définir les limites de phrases avec NLTK au lieu de spaCy."""
#     if len(str(doc.text).split()) > 3:
#         tokenizer = PunktSentenceTokenizer(doc.text)
#         sentences = tokenizer.tokenize(doc.text)
#         for nltk_sentence in sentences:
#             words = re.findall(r"[\w]+|[^\s\w]", nltk_sentence)
#             for i in range(len(doc) - len(words) + 1):
#                 token_list = [str(token) for token in doc[i : i + len(words)]]
#                 if token_list == words:
#                     doc[i].is_sent_start = True
#                     for token in doc[i + 1 : i + len(words)]:
#                         token.is_sent_start = False
#     return doc

# On ajoute le composant au pipeline en passant son nom
# parse.add_pipe("pre_segment", before="parser")


# Exceptions (abréviations, etc.)
# EXCEPTIONS = [
#     r'Art\.$', r'Abs\.$', r'u\.a\.$', r'U\.a\.$', r'u\.E\.$', r'U\.E\.$',
#     r'vgl\.$', r'Vgl\.$', r'bzw\.$', r'i\.V\.m\.$', r'Buchst\.$', r'd\.h\.$',
#     r'paras\.$', r'para\.$', r'No\.$', r' p\.$', r'pp\.$', r'D\.R\.$',
#     r'Eur\.$', r'i\.e\.$', r'Comm\.$', r'eg\.$', r'cf\.$', r'loc\.$', r'no\.$',
#     r'cit\.$', r'seq\.$', r'R\.\s*v\.$', r' v\.$', r'C\.$',
#     r'J\.A\.$', r'C\.O\.$', r'S\.$', r'Ms\.$', r'no\.\s*\d+', r'a\.m\.$', r'p\.m\.$',
#     r'D\.$', r'W\.$', r'E\.$', r'B\.$', r"Nos\.$", r'Mr\.$', r'Prof\.$', r'v\.$',
#     r'i\.$', r'ii\.$', r'PARA\.$', r'II\.$', r'[A-Z]\.$', r'III\.$', r'IV\.$',
#     r'R\.R\.', r'H\.P\.$', r'V\.$', r'VI\.$', r'VII\.$', r'VIII\.$', r'IX\.$', r'X\.'
# ]
# EXCEPTIONS = [re.compile(e) for e in EXCEPTIONS]

# # Détection de fragment (à merger avec la phrase précédente)
# def is_fragment(s):
#     if len(s) < 20 or re.fullmatch(r"[A-Z]\.", s.strip()):
#         return True
#     tokens = s.split()
#     return len(tokens) < 4 and all(t.endswith('.') for t in tokens)

# def clean_segmented_sentences(sentences):
#     result = []
#     for s in sentences:
#         if result and is_fragment(s):
#             result[-1] += ' ' + s
#         else:
#             result.append(s)
#     return result

# @Language.component("pre_segment")
# def pre_segment(doc: Doc) -> Doc:
#     text = doc.text
#     sentences = []
#     buf = []
#     in_quotes = 0
#     in_parens = 0
#     for i, ch in enumerate(text):
#         print("ch",ch)
#         buf.append(ch)

#         if ch in {'"', '“', '”', '«', '»'}:
#             in_quotes ^= 1
#         elif ch == '(':
#             in_parens += 1
#         elif ch == ')':
#             in_parens = max(0, in_parens - 1)

#         if ch in {'.', '!', '?'} and in_quotes == 0 and in_parens == 0:
#             end = ''.join(buf).strip()
#             is_exception = any(regex.search(end) for regex in EXCEPTIONS)
#             lookahead = text[i+1:i+5]
#             if i + 1 == len(text) or (not is_exception and re.match(r'\s+[A-Z]', lookahead)):
#                 sentences.append(end)
#                 buf = []

#     if buf:
#         sentences.append(''.join(buf).strip())

#     sentences = clean_segmented_sentences(sentences)
#     offset = 0
#     starts_found = 0

#     for sent in sentences:
#         sent = sent.strip()
#         if not sent:
#             continue
#         start_char = text.find(sent, offset)
#         if start_char == -1:
#             continue  
#         offset = start_char + len(sent)
#         while start_char < len(text) and text[start_char].isspace():
#             start_char += 1
#         end_char = min(start_char + 1, len(text))
#         span = doc.char_span(start_char, end_char, alignment_mode="expand")
#         if span is None:
#             for tok in doc:
#                 if tok.idx >= start_char:
#                     tok.is_sent_start = True
#                     starts_found += 1
#                     break
#             continue
#         doc[span.start].is_sent_start = True
#         starts_found += 1
#     if len(doc):
#         doc[0].is_sent_start = True
#     print(f"pre_segment: {starts_found} phrases marquées / doc len={len(doc)}")

#     return doc
# parse.add_pipe("pre_segment", before="parser")


def get_sentences(doc):
    """Retourne la liste des phrases."""
    return list(doc.sents)

# def add_labels(doc, labels):
#     """Ajoute les étiquettes à partir d'une liste."""
#     adu_labels_list = []
#     clpr_label_list = []
#     for idx, label in enumerate(labels):
#         label = label.strip("\n").strip(" ")
#         if label == "Claim":
#             adu_labels_list.append(1)
#             clpr_label_list.append(1)
#         elif label == "Premise":
#             adu_labels_list.append(1)
#             clpr_label_list.append(0)
#         elif label == "MajorClaim":
#             adu_labels_list.append(1)
#             clpr_label_list.append(1)
#         elif label == "None":
#             adu_labels_list.append(0)
#         elif label == "ADU":
#             adu_labels_list.append(1)
#         elif label == "1":
#             adu_labels_list.append(1)
#         elif label == "0":
#             adu_labels_list.append(0)
#     if len(adu_labels_list) > len(doc._.Features):
#         adu_labels_list = adu_labels_list[: len(doc._.Features)]
#     elif len(adu_labels_list) < len(doc._.Features):
#         add_on = np.random.randint(low=0, high=1, size=len(doc._.Features) - len(adu_labels_list)).tolist()
#         adu_labels_list.extend(add_on)
#     nr_adus = sum([1 for l in adu_labels_list if l == 1])
#     if len(clpr_label_list) > nr_adus:
#         clpr_label_list = clpr_label_list[:nr_adus]
#     elif len(clpr_label_list) < nr_adus:
#         add_on = np.random.randint(low=0, high=1, size=nr_adus - len(clpr_label_list)).tolist()
#         clpr_label_list.extend(add_on)
#     doc._.Labels = adu_labels_list
#     doc._.CLPR_Labels = clpr_label_list
#     return doc

# add_label pour prédiction multi_class des mc
def add_labels(doc, labels):
    adu_labels_list   = []
    clpr_label_list   = []
    mc_label_list     = []

    for idx, label in enumerate(labels):
        label = label.strip("\n").strip(" ")

        if label == "Claim":
            adu_labels_list.append(1)
            clpr_label_list.append(1)
            mc_label_list.append(0)

        elif label == "Premise":
            adu_labels_list.append(1)
            clpr_label_list.append(0)
            mc_label_list.append(0)

        elif label == "MajorClaim":
            adu_labels_list.append(1)
            clpr_label_list.append(1)
            mc_label_list.append(1)

        elif label == "None":
            adu_labels_list.append(0)

        elif label in {"ADU", "1"}:
            adu_labels_list.append(1)

        elif label == "0":
            adu_labels_list.append(0)
    n = len(doc._.Features)
    # 1) ADU labels
    if len(adu_labels_list) > n:
        adu_labels_list = adu_labels_list[:n]
    elif len(adu_labels_list) < n:
        adu_labels_list.extend([0] * (n - len(adu_labels_list)))
    # 2) CLPR labels — on s’appuie sur le nombre d’ADU = sum(adu_labels_list)
    nr_adus = sum([1 for l in adu_labels_list if l == 1])
    if len(clpr_label_list) > nr_adus:
        clpr_label_list = clpr_label_list[:nr_adus]
    elif len(clpr_label_list) < nr_adus:
         add_on = np.random.randint(low=0, high=2, size=nr_adus - len(clpr_label_list)).tolist()
         clpr_label_list.extend(add_on)
    # 3) MC labels
    if len(mc_label_list) > nr_adus:
        mc_label_list = mc_label_list[:nr_adus]
    elif len(mc_label_list) < nr_adus:
        mc_label_list.extend([0] * (nr_adus - len(mc_label_list)))

    doc._.Labels     = adu_labels_list
    doc._.CLPR_Labels = clpr_label_list
    doc._.MC_Labels  = mc_label_list

    return doc


def get_token_label(token):
    """Retourne l'étiquette d'un token."""
    label_list = token.doc._.Labels
    for idx, sent in enumerate(token.doc.sents):
        if idx + 1 < len(list(token.doc.sents)):
            if token.i >= sent.start and token.i < list(token.doc.sents)[idx + 1].start:
                return label_list[idx]
        else:
            return label_list[idx]

def get_sentence_label(span):
    """Retourne l'étiquette d'une phrase."""
    return span.doc._.Labels[span._.index]

def get_index(span):
    """Retourne l'indice de la phrase dans le document."""
    for idx, s in enumerate(span.doc.sents):
        if span == s:
            return idx

def set_empty_labels(doc):
    """Initialise toutes les étiquettes à zéro pour chaque phrase."""
    labels = [0] * len(list(doc.sents))
    doc._.Labels = labels
    doc._.CLPR_Labels = labels
    doc._.MC_Labels = labels 

    return doc

def get_ADU(doc, mc=False):
    """Retourne toutes les phrases étiquetées comme ADU."""
    adu = doc._.Labels
    result = []
    for idx, s in enumerate(doc._.sentences):
        if adu[idx] == 1:
            result.append(s)
    return result

def get_CL(doc, mc=False):
    """Retourne toutes les phrases étiquetées comme ADU mais non comme MajorClaim."""
    adu = doc._.CLPR_Labels
    result = []
    for idx, s in enumerate(doc._.ADU_Sents):
        if adu[idx] == 1:
            result.append(s)
    return result

def get_PR(doc, mc=False):
    """Retourne toutes les phrases étiquetées comme ADU mais non comme MajorClaim."""
    adu = doc._.CLPR_Labels
    result = []
    for idx, s in enumerate(doc._.ADU_Sents):
        if adu[idx] == 0:
            result.append(s)
    return result

def get_features(span):
    return span.doc._.Features[span._.index]

def get_mc(doc):
    for idx, val in enumerate(list(doc.sents)):
        if doc._.MC_List[idx] == 1:
            return val

# nouveau code pour récupérer phrase associé à clpr
def get_clpr_label(span: Span) -> int:
    """
    Retourne 1 si l’ADU courante est une Claim, 0 si Premise.
    Les indices de CLPR_Labels sont comptés uniquement parmi les phrases
    où ADU == 1, d’où la petite conversion ci-dessous.
    """
    # indice de la phrase dans le document
    sent_idx = span._.index

    # Construire la liste (ordonnée) des indices des phrases ADU
    adu_indices = [i for i, lab in enumerate(span.doc._.Labels) if lab == 1]

    if sent_idx not in adu_indices:        # la phrase n’est pas une ADU
        return 0

    # position de l’ADU dans la sous-liste des ADU
    adu_pos = adu_indices.index(sent_idx)
    return span.doc._.CLPR_Labels[adu_pos]




# Pour prédiction multiclass des mc
def get_mc_label(span: Span) -> int:
    sent_idx    = span._.index
    adu_indices = [i for i, lab in enumerate(span.doc._.Labels) if lab == 1]

    if sent_idx not in adu_indices:      # pas une ADU → 0
        return 0

    adu_pos = adu_indices.index(sent_idx)
    return span.doc._.MC_Labels[adu_pos]

def get_major_claims(doc):
    # retourne la liste de Span (phrases) labellées MJ=1
    return [sent for sent in doc.sents if sent._.MC_Label == 1]

# rajout des span, doc pour entrainment mc
Span.set_extension("MC_Label",getter=get_mc_label)
Span.set_extension("Label", getter=get_sentence_label)
# Span.set_extension("CLPR_Label", getter=get_sentence_label)
Span.set_extension("CLPR_Label", getter=get_clpr_label, force=True) #plus de sureté 
Span.set_extension("index", getter=get_index)
Span.set_extension("Feature", getter=get_features)
Span.set_extension("mc", default=0)

Token.set_extension("Label", getter=get_token_label)

Doc.set_extension("ADU_Sents", getter=get_ADU)
Doc.set_extension("Claim_Sents", getter=get_CL)
Doc.set_extension("Premise_Sents", getter=get_PR)
Doc.set_extension("MC_List", default=[])
# Doc.set_extension("MajorClaim", getter=get_mc)
Doc.set_extension("sentences", getter=get_sentences)
Doc.set_extension("Labels", default=[0])
Doc.set_extension("CLPR_Labels", default=[0])
Doc.set_extension("MC_Labels", default=[0])
Doc.set_extension("MajorClaim_Spans", getter=get_major_claims)
Doc.set_extension("MC_Features",default=[])




def prep_production(filename, input_text):
    input_text = clean_text(input_text)
    doc = parse(input_text)
    doc._.key = filename
    set_features(doc)
    set_empty_labels(doc)
    return doc

def merge_docs(doc_list):
    comb_feat = list(itertools.chain.from_iterable(list(map(lambda x: x._.Features, doc_list))))
    comb_label = list(itertools.chain.from_iterable(list(map(lambda x: x._.Labels, doc_list))))
    comb_clpr_label = list(itertools.chain.from_iterable(list(map(lambda x: x._.CLPR_Labels, doc_list))))
    comb_embedding = list(itertools.chain.from_iterable(list(map(lambda x: x._.embeddings, doc_list))))
    # Pour prédiction multiclass des mc
    comb_mc_label= list(itertools.chain.from_iterable(list(map(lambda x: x._.MC_Labels, doc_list))))

    final_text = "FinalDocument"
    final_doc = parse(final_text)
    final_doc._.Features = comb_feat
    final_doc._.Labels = comb_label
    final_doc._.CLPR_Labels = comb_clpr_label
    #Pour prédiction multiclass des mc
    final_doc._.MC_Labels = comb_mc_label
    final_doc._.MC_List = comb_mc_label[:]
    final_doc._.embeddings = comb_embedding
    print("Merged Lists")
    return final_doc

def prep_training(filename, input_text, labels_list):
    doc = parse(input_text)
    doc._.key = filename
    doc = set_features(doc)
    doc = add_labels(doc, labels_list)
    return doc

def read_in(file_name1, file_name2, texts, label_list):
    if os.path.isfile(file_name1):
        with open(file_name1, "r+", encoding="utf8", errors="ignore") as f:
            text = f.read()
        with open(file_name2, "r+", encoding="utf8", errors="ignore") as f:
            labels = f.read().split("\n")
    else:
        with open(config["adu"]["path"]["input"] + "/" + file_name1, "r+", encoding="utf8", errors="ignore") as f:
            text = f.read()
        with open(config["adu"]["path"]["label"] + "/" + file_name2, "r+", encoding="utf8", errors="ignore") as f:
            labels = f.read().split("\n")
    text = clean_text(text)
    texts.append(text)
    label_list.append(labels)

def read_files(input_list, label):
    # Initialiser localement les listes partagées
    if config["debug"]:
        texts = []
        label_list = []
    else:
        manager = multiprocessing.Manager()
        texts = manager.list()
        label_list = manager.list()
    if isinstance(input_list, list) or isinstance(input_list, tuple):
        jobs = []
        for idx, infile in enumerate(input_list):
            print("Reading Document\t%s" % infile)
            p = multiprocessing.Process(target=read_in, args=(infile, label[idx], texts, label_list))
            #dans p on a texts (clean et tous) et label_list
            jobs.append(p)
            p.start()
        for proc in jobs:
            proc.join()#permet d'attendre que le processus parent se termine avant de continuer 
        doc_list = []

        for idx, doc in enumerate(
            parse.pipe(
                texts,
                disable=["ner"],
                batch_size=80,
                # n_process=multiprocessing.cpu_count(),
                n_process=1
            )
        ):
            doc._.key = input_list[idx]
            doc = set_features(doc)#prépare le texte pour une analyse plus poussée en collectant des indices à exploiter dans un pipeline NLP

            doc = add_labels(doc, label_list[idx])#attribution des étiquettes aux phrases en identifiant les ADUet en les classant comme Claim, MajorClaim ou Premise,et en ajustant les listes de labels à la taille des caractéristiques du document.
            doc_list.append(doc)
            # for i, sent in enumerate(doc.sents):
            #     adu  = doc._.Labels[i]
            #     clpr = sent._.CLPR_Label     
            #     # mc   = sent._.MC_Label       
            #     print(f"[{i:02d}] \"{sent.text.strip()}\" → ADU={adu}, CLPR={clpr}")
        final_doc = merge_docs(doc_list)#fusion de plusieurs documents en un seul en combinant leurs caractéristiques (Features), leurs étiquettes (Labels, CLPR_Labels) et leurs embeddings, puis crée un nouveau document contenant ces informations consolidées.
        print("Final Document Type:", type(final_doc))
        print("Nombre de Features:", len(final_doc._.Features))
        print("Les Labels:", final_doc._.Labels)
        print("Les CLPR Labels:", final_doc._.CLPR_Labels)
        print("Nombre d'Embeddings:", len(final_doc._.embeddings))

    else:
        with open(input_list, "r+", encoding="utf8") as f:
            text = f.read()
        text = clean_text(text)
        with open(label, "r+", encoding="utf8") as f:
            labels = f.read().split("\n")
        final_doc = prep_training(input_list, text, labels)
        print("Final Document Type:", type(final_doc))
        print("Nombre de Features:", len(final_doc._.Features))
        print("Nombre de Labels:", len(final_doc._.Labels))
        print("Nombre de CLPR Labels:", len(final_doc._.CLPR_Labels))
        print("Nombre d'Embeddings:", len(final_doc._.embeddings))
    return final_doc
