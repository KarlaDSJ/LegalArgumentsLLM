from recap_am.adu import utilities
from recap_am.adu.classify import (
    fit_model,
    test_model,
    predict,
    predict_mc,
    predict_clpr,
    fit_clpr_model,
    fit_mc_model,
    test_clpr_model,
    test_model_mc
)
from recap_am.model.config import Config
from recap_am.adu.feature_select import filter_feats, add_embeddings

config = Config.get_instance()


def run_train(input_doc):
    """Train ADU and CLPR models."""
    input_doc = filter_feats(input_doc, load=False)
    print("Finished Feature selection")
    input_doc = add_embeddings(input_doc)
    print("Added Embeddings")
    adu_model = fit_model(input_doc)
    print("Fit ADU Model")
    return adu_model


def run_clpr_train(input_doc):
    """Train ADU and CLPR models."""
    input_doc = filter_feats(input_doc, load=False)
    print("Finished Feature selection")
    input_doc = add_embeddings(input_doc)
    clpr_feats = []
    for idx, l in enumerate(input_doc._.Labels):
        if l == 1:
            clpr_feats.append(input_doc._.Features[idx])
    input_doc._.CLPR_Features = clpr_feats#je comprends pas pourquoi il ajoute le cpl_feat qui correspond à si oui ou non c'est une adu alors que dans clpr_features on a si oui ou non c'est une claim/majorclaim
    #c'est à regarder dans read_files, bizarre car dans clpr on est censé faire classification de premisse/claim
    print("Added Embeddings")
    adu_model = fit_clpr_model(input_doc)
    print("Fit CLPR Model")
    return adu_model

def run_mc_train(input_doc):
    """Train ADU and CLPR models."""
    input_doc = filter_feats(input_doc, load=False)
    print("Finished Feature selection")
    input_doc = add_embeddings(input_doc)
    mc_feats = []
    for idx, l in enumerate(input_doc._.Labels):
        if l == 1:
            mc_feats.append(input_doc._.Features[idx])
    input_doc._.MC_Features = mc_feats
    print("mc_feats",mc_feats)
    print("Added Embeddings")
    mc_model = fit_mc_model(input_doc)
    print("Fit CLPR Model")
    return mc_model


def run_test(input_doc, model):
    """Test ADU and CLPR models."""
    input_doc = filter_feats(input_doc, load=True)
    print("Filtered feats")
    input_doc = add_embeddings(input_doc)
    print("Added embeds")
    acc, prec, rec, f1 = test_model(model, input_doc)
    return acc, prec, rec, f1


def run_clpr_test(input_doc, model):
    """Test ADU and CLPR models."""
    input_doc = filter_feats(input_doc, load=True)
    print("Filtered feats")
    input_doc = add_embeddings(input_doc)
    clpr_feats = []
    for idx, l in enumerate(input_doc._.Labels):
        if l == 1:
            clpr_feats.append(input_doc._.Features[idx])
    input_doc._.CLPR_Features = clpr_feats
    # print("features test",clpr_feats)
    print("Added embeds")
    acc, prec, rec, f1 = test_clpr_model(model, input_doc)
    return acc, prec, rec, f1

def run_test_mc(input_doc, model):
    """Test ADU and CLPR models."""
    input_doc = filter_feats(input_doc, load=True)
    print("Filtered feats")
    input_doc = add_embeddings(input_doc)
    mc_feats = []
    for idx, l in enumerate(input_doc._.Labels):
        if l == 1:
            mc_feats.append(input_doc._.Features[idx])
    input_doc._.MC_Features = mc_feats
    print("Added embeds")
    acc, prec, rec, f1 = test_model_mc(model, input_doc)
    return acc, prec, rec, f1


def run_production(input_doc):
    """Apply classification on doc."""
    input_doc = filter_feats(input_doc, load=True)
    input_doc = add_embeddings(input_doc)
    doc = predict(input_doc)
    clpr_feats = []
    for idx, l in enumerate(input_doc._.Labels):
        if l == 1:
            clpr_feats.append(input_doc._.Features[idx])
    if len(clpr_feats) < 2:
        input_doc._.Labels = [1 for s in input_doc._.Labels]
        clpr_feats = input_doc._.Features
    input_doc._.CLPR_Features = clpr_feats
    doc = predict_clpr(input_doc)
    doc = predict_mc(input_doc)
    return doc
