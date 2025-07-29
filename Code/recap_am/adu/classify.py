import numpy as np
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedShuffleSplit,
    RandomizedSearchCV,
)
from sklearn.metrics.pairwise import cosine_similarity
from recap_am.adu import utilities
from recap_am.model.config import Config
import torch
import torch.nn as nn
from recap_am.adu import cnn

config = Config.get_instance()


def fit_model(input_doc):
    """Load and train model."""
    model = utilities.get_model()
    model = train_model(model, input_doc)
    return model


def fit_clpr_model(input_doc):
    """Load and train model."""
    model = utilities.get_model()
    print(model)
    model = train_clpr_model(model, input_doc)
    return model

def fit_mc_model(input_doc):
    """Load and train model."""
    model = utilities.get_model()
    print(model)
    model = train_mc_model(model, input_doc)
    return model


def train_model(model, input_doc):
    """Apply GridSearch or normal fitting to model and save it."""
    feature = input_doc._.Features
    label = input_doc._.Labels
    feature = np.asarray(feature)
    label = np.asarray(label)
    print("features_clpr",feature)
    print("len",len(feature))
    print("label_clpr",label)
    print("len label",len(label))
    train_method = config["adu"]["train_method"]
    model_type = config["adu"]["model"]
    if (
        train_method in ["GridSearch", "RandomSearch"]
        and not model_type == "AutoML"
        and not model_type == "Stacking"
    ):
        param_grid = utilities.get_param_grid()
        cv_split = StratifiedShuffleSplit(
            n_splits=config["adu"]["n_splits"], test_size=0.33
        )
        if train_method == "GridSearch":
            model = GridSearchCV(model, param_grid=param_grid, cv=cv_split, refit=True)
        elif train_method == "RandomSearch":
            model = RandomizedSearchCV(
                model, cv=cv_split, refit=True, param_distributions=param_grid
            )
        model.fit(feature, label)
        utilities.save_model(model, model.best_params_)
    elif train_method=="cnn":
        num_epochs = config["adu"].get("cnn_epochs", 20) #10
        batch_size = config["adu"].get("cnn_batch_size", 64)#32
        lr = config["adu"].get("cnn_lr", 1e-4)

        X = torch.tensor(feature, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(X, y)
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cnn_model = cnn.LegalClassifier(
            input_dim=X.shape[1],
            hidden_dim=config["adu"].get("cnn_hidden_dim", 512),#256
            num_classes=len(np.unique(label)),
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(cnn_model.parameters(), lr=lr,weight_decay=1e-5)

        cnn_model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for embeddings_batch, labels_batch in train_loader:
                embeddings_batch = embeddings_batch.to(device)
                labels_batch = labels_batch.to(device)

                outputs = cnn_model(embeddings_batch)
                loss = criterion(outputs, labels_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * embeddings_batch.size(0)

            avg_loss = epoch_loss / len(dataset)
            print(f"Epoch {epoch+1}/{num_epochs} — Loss: {avg_loss:.4f}")

        torch.save(cnn_model.state_dict(), config["adu"].get("cnn_save_path", "cnn_model.pt"))
        print("CNN model saved to disk.")
        return cnn_model

    
    else:
        model.fit(feature, label)
        utilities.save_model(model, params=None)
    return model


def train_clpr_model(model, input_doc):
    """Apply GridSearch or normal fitting to model and save it."""
    feature = input_doc._.CLPR_Features
    label = input_doc._.CLPR_Labels
    print("features_clpr",feature)
    print("len",len(feature))
    print("label_clpr",label)
    print("len label",len(label))
    feature = np.asarray(feature)
    label = np.asarray(label)
    train_method = config["adu"]["train_method"]
    model_type = config["adu"]["model"]
    if (
        train_method in ["GridSearch", "RandomSearch"]
        and not model_type == "AutoML"
        and not model_type == "Stacking"
    ):
        param_grid = utilities.get_param_grid()
        cv_split = StratifiedShuffleSplit(
            n_splits=config["adu"]["n_splits"], test_size=0.33
        )
        if train_method == "GridSearch":
            model = GridSearchCV(model, param_grid=param_grid, cv=cv_split, refit=True)
        elif train_method == "RandomSearch":
            model = RandomizedSearchCV(
                model, cv=cv_split, refit=True, param_distributions=param_grid
            )
        model.fit(feature, label)
        utilities.save_clpr_model(model, model.best_params_)
    elif train_method=="cnn" :
        num_epochs  = config["adu"].get("cnn_epochs", 20)
        batch_size  = config["adu"].get("cnn_batch_size", 64)
        lr          = config["adu"].get("cnn_lr", 1e-4)
        hidden_dim  = config["adu"].get("cnn_hidden_dim", 512)
        dropout_p   = config["adu"].get("cnn_dropout", 0.5)
        weight_decay = config["adu"].get("cnn_weight_decay", 1e-5)

        # Conversion des features/labels en tenseurs
        X = torch.tensor(feature, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(X, y)
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Instanciation du même LegalClassifier, mais on spécifie hidden_dim
        cnn_model = cnn.LegalClassifier(
            input_dim=X.shape[1],
            hidden_dim=hidden_dim,
            num_classes=len(np.unique(label))
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            cnn_model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        best_state = None
        best_loss = float('inf')
        patience_counter = 0
        patience_max = config["adu"].get("cnn_patience", 5)

        for epoch in range(num_epochs):
            cnn_model.train()
            epoch_loss = 0.0
            for embeddings_batch, labels_batch in train_loader:
                embeddings_batch = embeddings_batch.to(device)
                labels_batch = labels_batch.to(device)

                outputs = cnn_model(embeddings_batch)
                loss = criterion(outputs, labels_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * embeddings_batch.size(0)

            avg_loss = epoch_loss / len(dataset)
            print(f"[CLPR CNN] Epoch {epoch+1}/{num_epochs} — Loss: {avg_loss:.4f}")

            # Early stopping simplifié sur la loss d'entraînement (pas de val ici)
            # Si besoin, on peut ajouter un sous-ensemble validation comme dans train_model.
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = cnn_model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience_max:
                    print("[CLPR CNN] Early stopping (pas d'amélioration).")
                    break

        # Sauvegarde du meilleur état
        save_path = config["adu"].get("cnn_clpr_save_path", "cnn_clpr_model.pt")
        torch.save(best_state, save_path)
        print(f"[CLPR CNN] Modèle enregistré sous '{save_path}'.")
        return cnn_model


    else:
        model.fit(feature, label)
        utilities.save_clpr_model(model, params=None)
    return model


def train_mc_model(model, input_doc):
    """Apply GridSearch/RandomSearch or CNN training on MC features, then save."""
    feature = input_doc._.MC_Features
    label = input_doc._.MC_Labels
    feature = np.asarray(feature)
    label = np.asarray(label)

    train_method = config["adu"]["train_method"]
    model_type = config["adu"]["model"]

    # 1. GridSearchCV / RandomizedSearchCV pour modèles scikit-learn
    if (
        train_method in ["GridSearch", "RandomSearch"]
        and model_type not in ["AutoML", "Stacking"]
    ):
        param_grid = utilities.get_param_grid()
        cv_split = StratifiedShuffleSplit(
            n_splits=config["adu"]["n_splits"], test_size=0.33
        )
        if train_method == "GridSearch":
            model = GridSearchCV(model, param_grid=param_grid, cv=cv_split, refit=True)
        else:  # RandomSearch
            model = RandomizedSearchCV(
                model, cv=cv_split, refit=True, param_distributions=param_grid
            )
        model.fit(feature, label)
        utilities.save_mc_model(model, model.best_params_)
        return model

    # 2. CNN PyTorch
    elif train_method == "cnn":
        num_epochs   = config["adu"].get("cnn_epochs", 50)
        batch_size   = config["adu"].get("cnn_batch_size", 64)
        lr           = config["adu"].get("cnn_lr", 1e-4)
        hidden_dim   = config["adu"].get("cnn_hidden_dim", 512)
        dropout_p    = config["adu"].get("cnn_dropout", 0.5)
        weight_decay = config["adu"].get("cnn_weight_decay", 1e-5)

        X = torch.tensor(feature, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(X, y)
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cnn_model = cnn.LegalClassifier(
            input_dim=X.shape[1],
            hidden_dim=hidden_dim,
            num_classes=len(np.unique(label))
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            cnn_model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        best_state = None
        best_loss = float('inf')
        patience_counter = 0
        patience_max = config["adu"].get("cnn_patience", 5)

        for epoch in range(num_epochs):
            cnn_model.train()
            epoch_loss = 0.0
            for embeddings_batch, labels_batch in train_loader:
                embeddings_batch = embeddings_batch.to(device)
                labels_batch = labels_batch.to(device)

                outputs = cnn_model(embeddings_batch)
                loss = criterion(outputs, labels_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * embeddings_batch.size(0)

            avg_loss = epoch_loss / len(dataset)
            print(f"[MC CNN] Epoch {epoch+1}/{num_epochs} — Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = cnn_model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience_max:
                    print("[MC CNN] Early stopping (pas d'amélioration).")
                    break

        save_path = config["adu"].get("cnn_save_mc_path", "cnn_mc_model.pt")
        torch.save(best_state, save_path)
        print(f"[MC CNN] Modèle enregistré sous '{save_path}'.")
        return cnn_model

    else:
        model.fit(feature, label)
        utilities.save_mc_model(model, params=None)
        return model


# def test_model(model, input_doc):
#     """Test model and return metrics."""
#     feature = input_doc._.Features
#     label = input_doc._.Labels
#     feature = np.asarray(feature)
#     predictions = model.predict(feature)
#     label = np.asarray(label)
#     acc, prec, rec, f1 = utilities.print_metrics(label, predictions)
#     return acc, prec, rec, f1


# pour gérer le cnn et le reste
def test_model(model, input_doc):
    feature = input_doc._.Features
    label   = input_doc._.Labels
    X_np    = np.asarray(feature)   
    y_np    = np.asarray(label)    
    print("feature test",feature)
    print("label test",label)

    

    if isinstance(model, torch.nn.Module):
        print("cnn")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_tensor = torch.tensor(X_np, dtype=torch.float32).to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(X_tensor)                  
            preds_tensor = torch.argmax(outputs, dim=1)  
            predictions = preds_tensor.cpu().numpy()      
    else:
        predictions = model.predict(X_np)
        print("predictions",predictions)

    acc, prec, rec, f1 = utilities.print_metrics(y_np, predictions)
    return acc, prec, rec, f1


# def test_clpr_model(model, input_doc):
#     """Test model and return metrics."""
#     feature = input_doc._.CLPR_Features
#     label = input_doc._.CLPR_Labels
#     # print("label test",label)
#     feature = np.asarray(feature)
#     predictions = model.predict(feature)
#     # print("predictions",predictions)
#     acc, prec, rec, f1 = utilities.print_metrics(label, predictions)
#     return acc, prec, rec, f1

def test_clpr_model(model, input_doc):
    """Test model and return metrics."""
    feature = input_doc._.CLPR_Features
    print("feature test",feature)
    label = input_doc._.CLPR_Labels
    print("label test",label)
    X_np = np.array(feature)
    Y_np = np.array(label) 
    if isinstance(model, torch.nn.Module):
        print("cnn")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_tensor = torch.tensor(X_np, dtype=torch.float32).to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(X_tensor)                  
            preds_tensor = torch.argmax(outputs, dim=1)  
            predictions = preds_tensor.cpu().numpy() 
    else :
        predictions = model.predict(X_np)
        print("predictions", predictions)
    acc, prec, rec, f1 = utilities.print_metrics(Y_np, predictions)
    return acc, prec, rec, f1

def test_model_mc(model, input_doc):
    """Test model and return metrics."""
    feature = input_doc._.MC_Features
    label = input_doc._.MC_Labels
    feature = np.asarray(feature)
    predictions = model.predict(feature)
    label = np.asarray(label)
    acc, prec, rec, f1 = utilities.print_metrics(label, predictions)
    return acc, prec, rec, f1

def predict(input_doc):
    """Apply model on doc."""
    feature = input_doc._.Features
    feature = np.asarray(feature)
    model = utilities.load_model()
    predictions = model.predict(feature)
    input_doc._.Labels = predictions
    return input_doc


def predict_clpr(input_doc):
    """Apply model on doc."""
    feature = input_doc._.CLPR_Features
    feature = np.asarray(feature)
    model = utilities.load_clpr_model()
    predictions = model.predict(feature)
    input_doc._.CLPR_Labels = predictions.tolist()
    return input_doc


#faire tourner les prédictions de mc que sur les claims 
# def predict_mc(input_doc):
#     labels = list(input_doc._.CLPR_Labels)
#     claim_embeddings = [
#         np.asarray(embed)
#         for label, embed in zip(labels, input_doc._.embeddings)
#         if label == 1
#     ]

#     if not claim_embeddings:
#         input_doc._.MC_List = [0] * len(labels)
#         return input_doc

#     method = config["adu"]["MC"]["method"]
#     if method == "centroid":
#         sim_results = compute_centroid_sim(claim_embeddings)
#     elif method == "pairwise":
#         sim_results = compute_pairwise_sim(claim_embeddings)
#     else:
#         input_doc._.MC_List = get_first_claim(input_doc)
#         return input_doc

#     mc_list = [0] * len(labels)
#     claim_indices = [i for i, lab in enumerate(labels) if lab == 1]

#     for orig_idx, sim_val in zip(claim_indices, sim_results):
#         mc_list[orig_idx] = sim_val

#     input_doc._.MC_List = mc_list
#     return input_doc



def predict_mc(input_doc):
    """Run majorclaim classification."""
    method = config["adu"]["MC"]["method"]
    embeddings = [
        np.asarray(e)
        for idx, e in enumerate(input_doc._.embeddings)
        if input_doc._.Labels[idx] == 1
    ]

    if method == "centroid":
        result = compute_centroid_sim(embeddings)
        return_list = [0] * len(input_doc._.sentences)
        mc_iter = 0
        for idx, l in enumerate(input_doc._.Labels):
            if l == 1:
                return_list[idx] = result[mc_iter]
                mc_iter += 1
            else:
                return_list[idx] = 0
        input_doc._.MC_List = return_list
    elif method == "pairwise":
        result = compute_pairwise_sim(embeddings)
        return_list = [0] * len(input_doc._.sentences)
        mc_iter = 0
        for idx, l in enumerate(input_doc._.Labels):
            if l == 1:
                return_list[idx] = result[mc_iter]
                mc_iter += 1
            else:
                return_list[idx] = 0
        input_doc._.MC_List = return_list
    elif method == "first":
        input_doc._.MC_List = get_first_claim(input_doc)
    else:
        input_doc._.MC_List = get_first_claim(input_doc)  # TODO: Maybe wrong fallback

    return input_doc


def get_first_claim(input_doc):
    sentences = input_doc._.sentences
    mc_list = [0] * len(sentences)

    for i, sent in enumerate(sentences):
        if sent._.Label == 1:
            mc_list[i] = 1
            break

    if 1 not in mc_list:
        mc_list[0] = 1

    return mc_list


def get_centroid(embeddings):
    """Compute embedding centroid."""
    centroid = sum_vectors(embeddings) / len(embeddings)
    return centroid


def sum_vectors(vec_list):
    """Sum up vectors element wise."""
    np_list = []
    for vec in vec_list:
        np_list.append(np.array(vec))
    s = sum(np_list)
    return s

#     """Retourne un one-hot sur l'embedding dont la sim au centroïde est la plus proche de la moyenne."""
# def compute_centroid_sim(embeddings):
#     center = get_centroid(embeddings).reshape(1, -1)
    
#     sims = []
#     for emb in embeddings:
#         sims.append(cosine_similarity(emb.reshape(1, -1), center)[0,0])
    
#     mean_sim = np.mean(sims)
    
#     idx_typical = int(np.argmin([abs(s - mean_sim) for s in sims]))
    
#     mc_list = [0] * len(embeddings)
#     mc_list[idx_typical] = 1
#     return mc_list

def compute_centroid_sim(embeddings):
    """Compute similarity to centroid and return list with nearest vector marked as 1."""
    center = get_centroid(embeddings).reshape(1, -1)

    def sim(x):
        return cosine_similarity(x, center)

    mc_list = [0] * len(embeddings)
    center_sim = []
    for embed in embeddings:
        center_sim.append(sim(embed.reshape(1, -1)))
    max_id = max(range(len(center_sim)), key=center_sim.__getitem__)
    mc_list[max_id] = 1
    return mc_list


def compute_pairwise_sim(embeddings):
    """Compute pairwise similarity and mark vector with highest average to all with 1."""
    mc_list = [0] * len(embeddings)
    pair_sim_sum = []
    for embed1 in embeddings:
        pair_sim = 0
        for embed2 in embeddings:
            pair_sim += cosine_similarity(embed1.reshape(1, -1), embed2.reshape(1, -1))
        pair_sim_sum.append(pair_sim / len(embeddings))
    index_max = max(range(len(pair_sim_sum)), key=pair_sim_sum.__getitem__)
    mc_list[index_max] = 1
    return mc_list
