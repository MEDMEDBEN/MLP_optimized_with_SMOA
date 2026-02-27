# app.py - Streamlit : affichage + comparaison MLP seul vs MLP + SMOA
import streamlit as st
import pandas as pd
import json
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# ======================== CONFIGURATION PAGE ========================
st.set_page_config(
    page_title="Classification - MLP vs MLP + SMOA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================== CSS MODERNE ========================
st.markdown("""
    <style>
    /* Page principale */
    .main {background-color: #f8f9fa;}
    .stApp {background-image: linear-gradient(135deg, #667eea 0%, #764ba2 100%);}
    
    /* Sidebar - Design moderne et élégant */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
        padding: 30px 20px;
    }
    
    /* Titre de la sidebar */
    .sidebar .sidebar-content h1 {
        color: #ecf0f1;
        text-align: center;
        font-size: 24px;
        margin-bottom: 30px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        border-bottom: 3px solid #667eea;
        padding-bottom: 15px;
    }
    
    /* Boutons radio (menu) */
    .sidebar .sidebar-content label {
        color: #ecf0f1;
        font-size: 16px;
        font-weight: 500;
        padding: 12px 15px;
        margin: 8px 0;
        border-radius: 8px;
        transition: all 0.3s ease;
        cursor: pointer;
        display: block;
        border-left: 4px solid transparent;
    }
    
    .sidebar .sidebar-content label:hover {
        background-color: rgba(102, 126, 234, 0.2);
        border-left: 4px solid #667eea;
        padding-left: 20px;
    }
    
    /* Boutons sélectionnés */
    .sidebar input[type="radio"]:checked + span {
        color: #667eea;
        font-weight: bold;
    }
    
    /* Titres et sous-titres */
    h1, h2, h3 {color: #2c3e50;}
    
    /* Boutons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 16px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.5);
    }
    
    /* Boîtes métriques */
    .metric-box {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #f0f2f6 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    
    /* Divider */
    hr {
        border-color: #667eea;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ======================== CHEMINS (à adapter si besoin) ========================
# MLP + SMOA
SMOA_MODEL_PATH     = "results_smoa/best_model_smoa.pkl"
SMOA_METRICS_PATH   = "results_smoa/best_hyperparameters.json"
SMOA_REPORT_PATH    = "results_smoa/classification_report_smoa.txt"
SMOA_GRAPH_PATH     = "results_smoa/graphs_summary_smoa.png"

# MLP seul (baseline)
MLP_MODEL_PATH      = "results_mlp/mlp_baseline.pkl"
MLP_REPORT_PATH     = "results_mlp/classification_report_mlp.txt"
MLP_GRAPH_PATH      = "results_mlp/accuracy_mlp.png"

# Helpers
def render_results(title, base_dir):
    st.subheader(title)
    if not os.path.exists(base_dir):
        st.info(f"Dossier non trouvé : {base_dir}")
        return

    found = False
    for root, _, files in os.walk(base_dir):
        files = sorted(files)
        for fname in files:
            found = True
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, base_dir)
            st.markdown(f"**Fichier : {rel}**")
            lower = fname.lower()
            try:
                if lower.endswith((".png", ".jpg", ".jpeg", ".gif")):
                    st.image(fpath, use_container_width=True)
                elif lower.endswith(".txt"):
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    st.text_area("Contenu", content, height=260, key=f"txt-{title}-{rel}")
                elif lower.endswith(".json"):
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        data = json.load(f)
                    st.json(data)
                elif lower.endswith(".pkl"):
                    st.info(f"Fichier modèle trouvé : {rel}")
                else:
                    st.write(f"Fichier disponible : {rel}")
            except Exception as e:
                st.error(f"Erreur lors de l'affichage de {rel} : {e}")

    if not found:
        st.info("Aucun fichier trouvé dans ce dossier.")


def extract_accuracy(report_path):
    try:
        if not os.path.exists(report_path):
            return None
        with open(report_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "accuracy" in line.lower():
                    parts = line.strip().split()
                    for p in parts:
                        try:
                            val = float(p)
                            if 0 <= val <= 1:
                                return val
                        except ValueError:
                            continue
        return None
    except Exception:
        return None


def parse_classification_report(report_path):
    """Extrait les informations clés du rapport de classification"""
    try:
        with open(report_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return content
    except Exception:
        return None

# Ton tableau de données
DATA_PATH = "data/covtype.csv"   # ← CHANGE ÇA POUR TON FICHIER

# Nom de la dataset (tu peux changer)
DATASET_NAME = "Mon Tableau de Données (lignes × colonnes)"  # ← CHANGE ÇA

# ======================== SIDEBAR ========================
st.sidebar.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <h1 style="color: #667eea; font-size: 32px; margin: 0;">🌲 STATS Dashboard</h1>
    <p style="color: #95a5a6; font-size: 12px; margin: 10px 0 0 0;">Forest Cover Classification</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

st.sidebar.markdown("""
<style>
.sidebar-menu {
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.menu-item {
    padding: 15px;
    border-radius: 8px;
    text-align: center;
    font-weight: 600;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "📍 Navigation",
    [
        "🌲 Données (EDA)",
        "📊 Comparaison",
        "🤖 Prédiction",
        "🧬 Algo"
    ],
    label_visibility="collapsed"
)

# Mapper les options affichées aux clés utilisées
page_map = {
    "🌲 Données (EDA)": "Données (EDA)",
    "📊 Comparaison": "Comparaison des modèles",
    "🤖 Prédiction": "Prédiction (SMOA)",
    "🧬 Algo": "Algo (explications)"
}
page = page_map.get(page, page)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; font-size: 12px; color: #95a5a6; margin-top: 30px;">
    <p>🚀 <b>v1.0</b> | ML Classification App</p>
    <p>Powered by Streamlit & scikit-learn</p>
</div>
""", unsafe_allow_html=True)

# ======================== ACCUEIL ========================
if page == "Données (EDA)":
    st.title("🌲 Données et EDA")
    st.markdown("""
    ### 🌳 Forest Cover Type (Covertype)
    
    Le jeu de données **Forest Cover Type (Covertype)** décrit des parcelles du **Roosevelt National Forest** 
    (Colorado, USA) et vise à prédire le type de couvert forestier dominant (7 classes) à partir de variables 
    topographiques, de distances à certaines structures et de caractéristiques de sol et de zones sauvages. 
    
    Chaque observation correspond à une cellule de grille de **30 m × 30 m** échantillonnée sur quatre zones 
    sauvages distinctes.
    """)

    if os.path.exists(DATA_PATH):
        try:
            if DATA_PATH.lower().endswith('.csv'):
                df = pd.read_csv(DATA_PATH)
            elif DATA_PATH.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(DATA_PATH)
            else:
                st.error("Format non supporté (CSV ou Excel uniquement).")
                df = None
        except Exception as e:
            st.error(f"Erreur lors du chargement des données: {e}")
            df = None

        if df is not None:
            st.success(f"📊 Dimensions : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

            # Aperçu des données EN PREMIER
            st.subheader("👀 Aperçu des données")
            st.dataframe(df.head(50), use_container_width=True)
            
            # Téléchargement
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Télécharger le tableau complet (CSV)", csv, "dataset_complet.csv", "text/csv")

            # Colonnes & Types + Valeurs manquantes (côte à côte)
            colA, colB = st.columns(2)
            with colA:
                st.subheader("📋 Colonnes & Types")
                types_df = pd.DataFrame({
                    "colonne": df.columns,
                    "type": [str(t) for t in df.dtypes]
                })
                st.dataframe(types_df, use_container_width=True)

            with colB:
                st.subheader("⚠️ Valeurs manquantes")
                na_df = df.isna().sum().reset_index()
                na_df.columns = ["colonne", "manquants"]
                st.dataframe(na_df, use_container_width=True)

            # Statistiques descriptives
            st.subheader("📈 Statistiques descriptives (numériques)")
            desc = df.select_dtypes(include=['number']).describe().T
            st.dataframe(desc, use_container_width=True)

            # Corrélation
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) >= 2:
                st.subheader("🔗 Matrice de corrélation (10 premières colonnes numériques)")
                # Prendre seulement les 10 premières colonnes numériques
                top_cols = numeric_cols[:10]
                corr = df[top_cols].corr()
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", 
                            cbar_kws={"label": "Corrélation"}, ax=ax, 
                            square=True, linewidths=0.5, vmin=-1, vmax=1)
                ax.set_title("Matrice de Corrélation", fontsize=14, fontweight='bold', pad=20)
                plt.xticks(rotation=45, ha='right', fontsize=10)
                plt.yticks(rotation=0, fontsize=10)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            # Distribution simple d'une colonne numérique
            st.subheader("📊 Distribution d'une colonne")
            col_for_dist = st.selectbox("📍 Choisir une colonne pour la distribution", df.columns.tolist())
            if col_for_dist:
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                if pd.api.types.is_numeric_dtype(df[col_for_dist]):
                    ax2.hist(df[col_for_dist].dropna(), bins=30, color="#667eea", edgecolor='black')
                    ax2.set_title(f"Histogramme - {col_for_dist}", fontsize=14, fontweight='bold')
                    ax2.set_xlabel("Valeurs", fontsize=12)
                    ax2.set_ylabel("Fréquence", fontsize=12)
                else:
                    df[col_for_dist].value_counts().plot(kind='bar', ax=ax2, color="#764ba2", edgecolor='black')
                    ax2.set_title(f"Comptes par catégorie - {col_for_dist}", fontsize=14, fontweight='bold')
                    ax2.set_xlabel("Catégories", fontsize=12)
                    ax2.set_ylabel("Fréquence", fontsize=12)
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()
    else:
        st.warning(f"Fichier non trouvé : {DATA_PATH}")
        st.info("Place ton fichier dans le dossier 'data/' et modifie le chemin dans le code.")

# ======================== COMPARAISON DES MODÈLES ========================
elif page == "Comparaison des modèles":
    st.title("Comparaison : MLP seul vs MLP + SMOA")
    
    # Afficher les chemins pour le debug
    with st.expander("🔍 Chemins des fichiers (debug)"):
        st.info(f"""
        **Chemins recherchés :**
        - MLP Rapport : `{MLP_REPORT_PATH}` → {'✅ Trouvé' if os.path.exists(MLP_REPORT_PATH) else '❌ Manquant'}
        - MLP Graph : `{MLP_GRAPH_PATH}` → {'✅ Trouvé' if os.path.exists(MLP_GRAPH_PATH) else '❌ Manquant'}
        - SMOA Rapport : `{SMOA_REPORT_PATH}` → {'✅ Trouvé' if os.path.exists(SMOA_REPORT_PATH) else '❌ Manquant'}
        - SMOA Graph : `{SMOA_GRAPH_PATH}` → {'✅ Trouvé' if os.path.exists(SMOA_GRAPH_PATH) else '❌ Manquant'}
        - SMOA Metrics : `{SMOA_METRICS_PATH}` → {'✅ Trouvé' if os.path.exists(SMOA_METRICS_PATH) else '❌ Manquant'}
        
        **Répertoire courant :** `{os.getcwd()}`
        """)
    
    acc_mlp = extract_accuracy(MLP_REPORT_PATH)
    acc_smoa = extract_accuracy(SMOA_REPORT_PATH)

    # Métriques principales côte à côte
    colA, colB = st.columns(2)
    with colA:
        st.subheader("🧠 MLP seul (baseline)")
        st.metric("Accuracy", f"{acc_mlp*100:.2f} %" if acc_mlp is not None else "N/A")
    with colB:
        st.subheader("🔬 MLP + SMOA (optimisé)")
        st.metric("Accuracy", f"{acc_smoa*100:.2f} %" if acc_smoa is not None else "N/A")

    # Rapports détaillés - ZONE COMPLÈTE
    st.markdown("---")
    st.subheader("📊 Rapports de Classification Détaillés")
    
    col_rep1, col_rep2 = st.columns(2)
    
    with col_rep1:
        st.markdown("### **MLP (Baseline)**")
        if os.path.exists(MLP_REPORT_PATH):
            report_mlp = parse_classification_report(MLP_REPORT_PATH)
            if report_mlp:
                with st.container():
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 20px; border-radius: 10px; color: white;">
                    <h4 style="margin-top: 0;">Résultats du modèle MLP</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.text(report_mlp)
        else:
            st.info("Rapport MLP non trouvé")
    
    with col_rep2:
        st.markdown("### **MLP + SMOA (Optimisé)**")
        if os.path.exists(SMOA_REPORT_PATH):
            report_smoa = parse_classification_report(SMOA_REPORT_PATH)
            if report_smoa:
                with st.container():
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                padding: 20px; border-radius: 10px; color: white;">
                    <h4 style="margin-top: 0;">Résultats du modèle MLP + SMOA</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.text(report_smoa)
        else:
            st.info("Rapport SMOA non trouvé")

    # Matrices de confusion côte à côte
    st.markdown("---")
    st.subheader("🔥 Matrices de Confusion")
    col3, col4 = st.columns(2)
    with col3:
        mlp_confusion_path = "results_mlp/confusion_matrix_mlp.png"
        if os.path.exists(mlp_confusion_path):
            st.image(mlp_confusion_path, caption="Matrice de confusion MLP", use_container_width=True)
        else:
            st.warning(f"❌ Fichier manquant : {mlp_confusion_path}")
    with col4:
        smoa_confusion_path = "results_smoa/confusion_matrix_smoa.png"
        if os.path.exists(smoa_confusion_path):
            st.image(smoa_confusion_path, caption="Matrice de confusion MLP + SMOA", use_container_width=True)
        else:
            st.warning(f"❌ Fichier manquant : {smoa_confusion_path}")

    # Courbes d'accuracy / training côte à côte
    st.markdown("---")
    st.subheader("📈 Courbes d'Entraînement et Accuracy")
    col5, col6 = st.columns(2)
    with col5:
        if os.path.exists(MLP_GRAPH_PATH):
            st.image(MLP_GRAPH_PATH, caption="Courbe accuracy MLP", use_container_width=True)
        else:
            st.warning(f"❌ Fichier manquant : {MLP_GRAPH_PATH}")
    with col6:
        if os.path.exists(SMOA_GRAPH_PATH):
            st.image(SMOA_GRAPH_PATH, caption="Courbes (loss/accuracy) MLP + SMOA", use_container_width=True)
        else:
            st.warning(f"❌ Fichier manquant : {SMOA_GRAPH_PATH}")

    # Fichiers complets en bas
    st.markdown("---")
    st.subheader("📁 Fichiers Complets")
    col7, col8 = st.columns(2)
    with col7:
        with st.expander("📂 Fichiers MLP (complets)"):
            render_results("", "results_mlp")
    with col8:
        with st.expander("📂 Fichiers SMOA (complets)"):
            render_results("", "results_smoa")

    # Hyperparamètres SMOA à la fin
    st.markdown("---")
    if os.path.exists(SMOA_METRICS_PATH):
        st.subheader("⚙️ Hyperparamètres Optimaux (SMOA)")
        try:
            with open(SMOA_METRICS_PATH, "r", encoding="utf-8", errors="ignore") as f:
                hp = json.load(f)
            st.json(hp)
        except Exception as e:
            st.error(f"Impossible de lire best_hyperparameters.json : {e}")

# ======================== PRÉDICTION (SMOA) ========================
elif page == "Prédiction (SMOA)":
    st.title("🤖 Prédire avec le modèle SMOA")
    model_path = "results_smoa/best_model_smoa.pkl"
    scaler_path = "results_smoa/scaler_smoa.pkl"

    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        st.warning("Fichiers modèle/scaler SMOA introuvables dans results_smoa/. Ajoute best_model_smoa.pkl et scaler_smoa.pkl.")
    else:
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
        except Exception as e:
            st.error(f"Erreur lors du chargement du modèle ou du scaler : {e}")
            model = None
            scaler = None

        if model is not None and scaler is not None:
            st.markdown("📝 **Formulaire rapide** (12 premières features) pour une prédiction unique, puis prédictions par CSV.")

            # Charger un échantillon de référence pour proposer des valeurs
            df_ref = None
            if os.path.exists(DATA_PATH):
                try:
                    df_ref = pd.read_csv(DATA_PATH).drop(columns=["Cover_Type"], errors="ignore")
                except Exception:
                    df_ref = None

            feature_list_full = list(getattr(scaler, "feature_names_in_", []))
            feature_list = feature_list_full[:12] if feature_list_full else []

            form_values = {}
            with st.form("form_pred_single"):
                cols = st.columns(2)
                for idx, col in enumerate(feature_list):
                    target_col = cols[idx % 2]
                    series = None
                    if df_ref is not None and col in df_ref.columns:
                        series = df_ref[col].dropna()

                    options = None
                    default_val = 0.0
                    if series is not None and not series.empty:
                        uniq = series.unique()
                        if len(uniq) <= 20:
                            options = sorted(pd.unique(uniq).tolist())
                        default_val = float(series.median()) if pd.api.types.is_numeric_dtype(series) else 0.0

                    with target_col:
                        if options is not None:
                            selected = st.selectbox(f"📌 {col}", options)
                            form_values[col] = selected
                        else:
                            form_values[col] = st.number_input(f"🔢 {col}", value=default_val)

                submitted = st.form_submit_button("🎯 Prédire (ligne unique)")

            if submitted:
                try:
                    all_values = form_values.copy()
                    missing_features = [f for f in feature_list_full if f not in all_values]
                    if missing_features and df_ref is not None:
                        random_row = df_ref.sample(1).iloc[0]
                        for feat in missing_features:
                            if feat in df_ref.columns:
                                all_values[feat] = random_row[feat]
                            else:
                                all_values[feat] = 0
                    row_df = pd.DataFrame([all_values])[feature_list_full]
                    row_scaled = scaler.transform(row_df)
                    pred = model.predict(row_scaled)[0]
                    st.success(f"Classe prédite : **{pred}**")
                except Exception as e:
                    st.error(f"Erreur lors de la prédiction via formulaire : {e}")

            st.markdown("---")
            st.markdown("📤 **Prédictions par CSV** : Charge un CSV avec les mêmes colonnes d'entraînement (sans la cible). Si 'Cover_Type' est présent, il sera ignoré.")
            uploaded = st.file_uploader("📁 Choisir un fichier CSV", type=["csv"])

            df_pred = None
            if uploaded is not None:
                try:
                    df_pred = pd.read_csv(uploaded)
                except Exception as e:
                    st.error(f"Impossible de lire le CSV : {e}")

            if df_pred is None and os.path.exists(DATA_PATH):
                st.info("✅ Aucun fichier chargé, utilisation des 50 premières lignes du dataset local (sans la cible).")
                try:
                    df_pred = pd.read_csv(DATA_PATH).head(50)
                except Exception as e:
                    st.error(f"Impossible de lire {DATA_PATH} : {e}")

            if df_pred is not None:
                if "Cover_Type" in df_pred.columns:
                    df_pred = df_pred.drop(columns=["Cover_Type"])

                st.write("👁️ **Prévisualisation des données d'entrée :**")
                st.dataframe(df_pred.head(10))

                try:
                    if hasattr(scaler, "feature_names_in_"):
                        missing = [c for c in scaler.feature_names_in_ if c not in df_pred.columns]
                        if missing:
                            st.error(f"Colonnes manquantes pour la prédiction : {missing}")
                        else:
                            X = df_pred[scaler.feature_names_in_]
                    else:
                        X = df_pred

                    X_scaled = scaler.transform(X)
                    preds = model.predict(X_scaled)
                    st.success("✅ Prédictions générées avec succès!")
                    out = df_pred.copy()
                    out["prediction"] = preds
                    st.dataframe(out.head(20))

                    # Répartition des classes prédites
                    st.subheader("📊 Répartition des classes prédites")
                    counts = pd.Series(preds).value_counts().sort_index()
                    st.bar_chart(counts)

                    # Téléchargement
                    csv_bytes = out.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Télécharger les prédictions (CSV)", csv_bytes, "predictions_smoa.csv", "text/csv")
                except Exception as e:
                    st.error(f"Erreur lors de la prédiction : {e}")

# ======================== PAGE ALGO (explications) ========================
elif page == "Algo (explications)":
    st.title("🧬 Algorithmes : MLP et SMOA")
    st.markdown("""
    Cette page explique le fonctionnement des deux approches utilisées :
    - 🧠 **MLP** (Multilayer Perceptron) pour la classification.
    - 🔬 **SMOA** (optimisation méta-heuristique) pour chercher de bons hyperparamètres du MLP.

    Étapes clés (hors prétraitement et sauvegardes) :
    - 1️⃣ Définition d'une fonction objectif qui, à partir d'un vecteur de paramètres, construit un MLP et évalue sa performance (accuracy de validation).
    - 2️⃣ SMOA explore l'espace des hyperparamètres du MLP, en combinant mouvements vers le meilleur, recherche locale guidée par des paramètres (magnétisme, rayon de senteur), et arrêt anticipé.
    - 3️⃣ Une fois les meilleurs hyperparamètres trouvés, on entraîne le MLP final sur plusieurs époques en enregistrant l'évolution de l'accuracy (courbe d'amélioration).
    """)

    st.subheader("💻 Code de référence (SMOA + MLP)")
    code_str = """# mlp_smoa_forest_cover_colab.py
"""
    # Insérer le code fourni (corrigé minimalement pour l'affichage)
    algo_code = '''# mlp_smoa_forest_cover_colab.py
"""
MLP + SMOA optimisation sur le dataset Forest Cover Type
Colonne cible : Cover_Type (7 classes)
Split : 80% train / 20% test
Sauvegarde dans results_smoa/
Compatible Colab avec téléchargement depuis Kaggle
"""

import os
import json
import joblib
import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

plt.style.use("default")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = [12, 8]

RND = 42
DATA_PATH = "data/covtype.csv"

# ---------------------------
# SMOA (coeur de l'algo)
# ---------------------------
class SMOA:
    def __init__(self, obj_fn, dim, pop_size=8, lb=-3.0, ub=3.0, max_iter=15,
                 mag_strength=0.8, scent_radius=0.5, decay_rate=0.05,
                 sinus_amp=0.1, sinus_freq=0.05, diversity_threshold=1e-4,
                 early_stop=6, seed=None):
        self.obj = obj_fn
        self.dim = dim
        self.pop_size = pop_size
        self.lb = np.ones(dim) * lb if np.isscalar(lb) else np.array(lb)
        self.ub = np.ones(dim) * ub if np.isscalar(ub) else np.array(ub)
        self.max_iter = max_iter
        self.rng = np.random.default_rng(seed)
        self.mag_strength0 = mag_strength
        self.scent_radius0 = scent_radius
        self.decay_rate = decay_rate
        self.sinus_amp = sinus_amp
        self.sinus_freq = sinus_freq
        self.diversity_threshold = diversity_threshold
        self.early_stop = early_stop

    def clamp(self, pop):
        return np.clip(pop, self.lb, self.ub)

    def population_diversity(self, pop):
        return np.mean(np.std(pop, axis=0))

    def adapt_parameters(self, t):
        exp_term = np.exp(-self.decay_rate * t)
        mag = self.mag_strength0 * exp_term + self.sinus_amp * np.sin(2 * np.pi * self.sinus_freq * t)
        scent = self.scent_radius0 * exp_term * (0.7 + 0.3 * np.sin(2 * np.pi * self.sinus_freq * t))
        return max(1e-6, mag), max(1e-6, scent)

    def magnet_move(self, x, best, mag_strength):
        direction = best - x
        dist = np.linalg.norm(direction)
        if dist > 1e-12:
            unit = direction / dist
        else:
            unit = self.rng.normal(size=self.dim)
            unit /= np.linalg.norm(unit)
        rand = self.rng.normal(size=self.dim)
        step = mag_strength * (unit * (0.5 + self.rng.random()) + 0.1 * rand)
        return x + step

    def scent_local_search(self, x, scent_radius, n_samples=3):
        best_x = x.copy()
        best_val, _ = self.obj(best_x)
        for _ in range(n_samples):
            direction = self.rng.normal(size=self.dim)
            nrm = np.linalg.norm(direction)
            if nrm > 0:
                direction = direction / nrm
            r = (self.rng.random() ** (1 / self.dim))
            candidate = x + direction * r * scent_radius * (self.ub - self.lb)
            candidate = self.clamp(candidate)
            val, _ = self.obj(candidate)
            if val < best_val:
                best_x = candidate.copy()
                best_val = val
        return best_x, best_val

    def run(self, verbose=True):
        pop = self.rng.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        fitness = np.array([self.obj(x)[0] for x in pop])
        best_idx = np.argmin(fitness)
        gbest = pop[best_idx].copy()
        gbest_val, gbest_metrics = self.obj(gbest)
        no_improve = 0
        history = {"best": [gbest_val], "mean": [np.mean(fitness)], "best_params": []}

        for t in range(self.max_iter):
            mag_strength, scent_radius = self.adapt_parameters(t)
            div = self.population_diversity(pop)
            if div < self.diversity_threshold:
                mag_strength *= 0.5
                scent_radius *= 1.5

            new_pop = pop.copy()
            new_fit = fitness.copy()
            for i in range(self.pop_size):
                x = pop[i]
                moved = self.magnet_move(x, gbest, mag_strength)
                local_x, local_f = self.scent_local_search(moved, scent_radius)
                new_pop[i] = local_x
                new_fit[i] = local_f
                if local_f < gbest_val:
                    gbest = local_x.copy()
                    gbest_val = local_f
                    gbest_metrics = self.obj(gbest)[1]
                    no_improve = 0

            pop = new_pop
            fitness = new_fit
            if fitness.min() < gbest_val:
                best_idx = np.argmin(fitness)
                gbest = pop[best_idx].copy()
                gbest_val = fitness[best_idx]
                gbest_metrics = self.obj(gbest)[1]
                no_improve = 0
            else:
                no_improve += 1

            history["best"].append(gbest_val)
            history["mean"].append(np.mean(fitness))
            history["best_params"].append(gbest_metrics.get("params", {}))

            if no_improve >= self.early_stop:
                break

        return gbest, gbest_val, gbest_metrics, history


# ---------------------------
# Objectif pour MLP
# ---------------------------
def make_objective(X_train, X_val, y_train, y_val, n_epochs_eval=10):
    activations = ["relu", "tanh", "logistic"]
    solvers = ["adam", "sgd"]

    def obj_fn(vec):
        vec = np.asarray(vec)
        h1 = max(16, int(round(abs(vec[0]) * 240)) + 16)
        h2 = max(8,  int(round(abs(vec[1]) * 120)) + 8)
        hidden_layers = (h1, h2)
        act_idx = int(abs(vec[2]) * len(activations)) % len(activations)
        activation = activations[act_idx]
        solver_idx = int(abs(vec[3]) * len(solvers)) % len(solvers)
        solver = solvers[solver_idx]
        alpha = 10 ** np.clip(vec[4], -6, 0)
        max_iter = int(50 + np.clip(vec[5], 0, 1) * 350)
        lr_init = 10 ** np.clip(vec[6], -5, -1)
        beta_1 = np.clip(vec[7], 0.8, 0.999)
        beta_2 = np.clip(vec[8], 0.9, 0.9999)
        tol = 10 ** np.clip(vec[9], -7, -3)

        params = {
            "hidden_layer_sizes": hidden_layers,
            "activation": activation,
            "solver": solver,
            "alpha": alpha,
            "max_iter": max_iter,
            "learning_rate_init": lr_init,
            "beta_1": beta_1,
            "beta_2": beta_2,
            "tol": tol,
            "random_state": 42,
            "verbose": False,
        }

        sig = inspect.signature(MLPClassifier.__init__)
        accepted = {k for k in sig.parameters.keys() if k != "self"}
        filtered = {k: v for k, v in params.items() if k in accepted}

        try:
            model = MLPClassifier(**filtered)
            classes = np.unique(y_train)
            for _ in range(n_epochs_eval):
                idx = np.random.permutation(len(X_train))
                model.partial_fit(X_train[idx], y_train[idx], classes=classes)
            val_acc = model.score(X_val, y_val)
            loss = 1.0 - val_acc
            metrics = {"params": params, "val_acc": val_acc}
        except Exception as e:
            loss = 1.0
            metrics = {"error": str(e)}

        return loss, metrics

    return obj_fn

'''
    st.code(algo_code, language="python")

    # ======================== EXPLICATIONS MATHÉMATIQUES ========================
    st.subheader("📐 Explications mathématiques et fonctionnement détaillé")
    
    st.markdown("""
    ### 1. **Initialisation de la population**
    
    La population initiale est générée aléatoirement dans l'espace de recherche :
    
    $$\\mathbf{x}_i^{(0)} \\sim \\text{Uniform}(\\mathbf{lb}, \\mathbf{ub}), \\quad i = 1, 2, \\ldots, N_{pop}$$
    
    où $\\mathbf{lb}$ et $\\mathbf{ub}$ sont les bornes inférieure et supérieure de l'espace de recherche, 
    et $N_{pop}$ est la taille de la population.
    
    ---
    
    ### 2. **Paramètres adaptatifs (magnétisme et rayon de senteur)**
    
    Les paramètres de contrôle évoluent au cours des itérations pour équilibrer exploration et exploitation :
    
    $$\\text{mag}(t) = \\text{mag}_0 \\cdot e^{-\\text{decay\\_rate} \\cdot t} + \\text{sinus\\_amp} \\cdot \\sin(2\\pi \\cdot \\text{sinus\\_freq} \\cdot t)$$
    
    $$\\text{scent}(t) = \\text{scent}_0 \\cdot e^{-\\text{decay\\_rate} \\cdot t} \\cdot \\left(0.7 + 0.3 \\cdot \\sin(2\\pi \\cdot \\text{sinus\\_freq} \\cdot t)\\right)$$
    
    **Interprétation** :
    - Le terme exponentiel $e^{-\\text{decay\\_rate} \\cdot t}$ réduit progressivement les paramètres (exploitation croissante)
    - Le terme sinusoïdal ajoute des oscillations pour échapper aux minima locaux (exploration)
    
    ---
    
    ### 3. **Mouvement magnétique (vers le meilleur)**
    
    Chaque particule se déplace vers le meilleur point trouvé ($\\mathbf{g}_{best}$) avec une atténuation aléatoire :
    
    $$\\mathbf{d} = \\mathbf{g}_{best} - \\mathbf{x}_i$$
    
    $$\\mathbf{u} = \\begin{cases} \\frac{\\mathbf{d}}{\\|\\mathbf{d}\\|_2} & \\text{si } \\|\\mathbf{d}\\| > 10^{-12} \\\\ \\text{Normal}(0, 1) & \\text{sinon} \\end{cases}$$
    
    $$\\mathbf{x}_i^{\\text{moved}} = \\mathbf{x}_i + \\text{mag}(t) \\cdot \\left(\\mathbf{u} \\cdot (0.5 + r_1) + 0.1 \\cdot \\boldsymbol{\\epsilon}\\right)$$
    
    où $r_1 \\sim \\text{Uniform}(0, 1)$ et $\\boldsymbol{\\epsilon} \\sim \\text{Normal}(0, 1)$.
    
    **Interprétation** : La particule se rapproche du meilleur avec du bruit stochastique pour explorer localement.
    
    ---
    
    ### 4. **Recherche locale guidée (rayon de senteur)**
    
    Autour de la particule déplacée, on explore le voisinage pour trouver une meilleure solution :
    
    $$\\mathbf{c}_j = \\mathbf{x}_i^{\\text{moved}} + \\mathbf{v} \\cdot r \\cdot \\text{scent}(t) \\cdot (\\mathbf{ub} - \\mathbf{lb})$$
    
    où :
    - $\\mathbf{v} \\sim \\text{Normal}(0, 1)$ (direction aléatoire normalisée)
    - $r \\sim \\text{Uniform}(0, 1)^{1/d}$ (rayon adaptatif à la dimension)
    - $d$ est la dimension de l'espace
    
    Pour chaque candidat $\\mathbf{c}_j$, on évalue la fonction objectif et on garde le meilleur localement.
    
    **Interprétation** : On explore un rayon sphérique autour de la particule pour affiner la solution.
    
    ---
    
    ### 5. **Mise à jour de la meilleure solution globale**
    
    À chaque itération, on met à jour le meilleur point trouvé :
    
    $$f(\\mathbf{g}_{best}^{\\text{new}}) \\leq f(\\mathbf{g}_{best}^{\\text{old}})$$
    
    Si la diversité de la population devient faible, on augmente l'exploration :
    
    $$\\text{Diversité} = \\frac{1}{d} \\sum_{k=1}^{d} \\sigma_k < \\text{diversity\\_threshold}$$
    
    Alors :
    $$\\text{mag}(t) \\leftarrow \\text{mag}(t) \\times 0.5, \\quad \\text{scent}(t) \\leftarrow \\text{scent}(t) \\times 1.5$$
    
    ---
    
    ### 6. **Critères d'arrêt**
    
    L'algorithme s'arrête si :
    - **Nombre d'itérations atteint** : $t \\geq t_{\\max}$
    - **Pas d'amélioration** : Aucune meilleure solution pendant $\\text{early\\_stop}$ itérations
    
    ---
    
    ### 7. **Fonction objectif pour l'optimisation du MLP**
    
    Étant donné un vecteur de paramètres $\\mathbf{vec} \\in \\mathbb{R}^{10}$, on construit un MLP et évalue sa performance :
    
    $$\\text{Loss}(\\mathbf{vec}) = 1 - \\text{Accuracy}_{validation}(\\text{MLP}(\\mathbf{vec}))$$
    
    Les paramètres du MLP sont décodés à partir de $\\mathbf{vec}$ :
    
    | Indice | Paramètre | Formule |
    |--------|-----------|---------|
    | 0 | $h_1$ (couche 1) | $\\max(16, \\lfloor \\|\\text{vec}[0]\\| \\times 240 \\rfloor + 16)$ |
    | 1 | $h_2$ (couche 2) | $\\max(8, \\lfloor \\|\\text{vec}[1]\\| \\times 120 \\rfloor + 8)$ |
    | 2 | activation | $\\{\\text{relu}, \\text{tanh}, \\text{logistic}\\}[\\lfloor \\|\\text{vec}[2]\\| \\cdot 3 \\rfloor]$ |
    | 3 | solver | $\\{\\text{adam}, \\text{sgd}\\}[\\lfloor \\|\\text{vec}[3]\\| \\cdot 2 \\rfloor]$ |
    | 4 | alpha (L2) | $10^{\\text{clip}(\\text{vec}[4], -6, 0)}$ |
    | 5 | max_iter | $\\lfloor 50 + \\text{clip}(\\text{vec}[5], 0, 1) \\times 350 \\rfloor$ |
    | 6 | learning_rate | $10^{\\text{clip}(\\text{vec}[6], -5, -1)}$ |
    | 7 | $\\beta_1$ (momentum) | $\\text{clip}(\\text{vec}[7], 0.8, 0.999)$ |
    | 8 | $\\beta_2$ (RMSprop) | $\\text{clip}(\\text{vec}[8], 0.9, 0.9999)$ |
    | 9 | tolerance | $10^{\\text{clip}(\\text{vec}[9], -7, -3)}$ |
    
    ---
    
    ### **Résumé du processus (step-by-step)**
    
    1. **Initialisation** : Générer une population aléatoire dans l'espace des hyperparamètres
    2. **Évaluation initiale** : Calculer la qualité (accuracy) pour chaque particule
    3. **Boucle principale** (jusqu'à convergence) :
        - **Adaptation** : Mettre à jour les paramètres $\\text{mag}(t)$ et $\\text{scent}(t)$
        - **Mouvement magnétique** : Déplacer chaque particule vers $\\mathbf{g}_{best}$
        - **Recherche locale** : Explorer le rayon de senteur autour de la nouvelle position
        - **Mise à jour globale** : Mettre à jour $\\mathbf{g}_{best}$ si amélioration
        - **Vérification de la diversité** : Si trop faible, augmenter exploration
    4. **Retour** : Les meilleurs hyperparamètres trouvés
    5. **Entraînement final** : Entraîner le MLP avec ces hyperparamètres sur tout l'ensemble d'entraînement
    
    ---
    
    **Avantages de SMOA** :
    - ✅ Équilibre exploration/exploitation dynamique
    - ✅ Pas de gradient requis (méta-heuristique)
    - ✅ Adapté à la recherche d'hyperparamètres discrets et continus
    - ✅ Arrêt anticipé pour économiser les ressources
    """)

    st.markdown("""
    En pratique dans cette application, nous affichons uniquement les résultats sauvegardés. 
    Ajoute tes fichiers dans `results_smoa/` pour voir les sorties correspondantes dans l'onglet Comparaison.
    """)

# ======================== FOOTER ========================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 14px;">
🌲 <b>Application Streamlit</b> – Affichage et Analyse | 🧠 Comparaison MLP vs MLP + SMOA | 📊 2025
</div>
""", unsafe_allow_html=True)