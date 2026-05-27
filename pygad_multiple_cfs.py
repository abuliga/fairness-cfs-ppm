#!/usr/bin/env python
# coding: utf-8
# # Counterfactual Trace Generation with PyGAD (Complex Encoding)
#
# Changes vs original:
#   - top_k_cfs  (CONF key, default 5): how many distinct CFs to retain per trace
#   - pairwise_diversity / pairwise_diversity_matrix helpers
#   - results CSV gains columns: cf_rank, top_k_diversity
#   - trace CSV gains columns:   cf_rank, top_k_diversity
#
from pm4py.objects.log.obj import Trace, Event
import regex as re
from Declare4Py.Utils.Declare.Checkers import ConstraintChecker
from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareAnalyzer import MPDeclareAnalyzer
from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareResultsBrowser import MPDeclareResultsBrowser
import pygad
import random
import os
from sklearn.inspection import permutation_importance
from tqdm.auto import tqdm
from itertools import product as iproduct
from itertools import combinations
from src.encoding.common import get_encoded_df
from src.predictive_model.predictive_model import PredictiveModel, drop_columns
from src.predictive_model.common import ClassificationMethods, get_tensor
from src.evaluation.common import evaluate_classifier
from src.hyperparameter_optimisation.common import retrieve_best_model, HyperoptTarget
from src.encoding.constants import TaskGenerationType, PrefixLengthStrategy, EncodingType, EncodingTypeAttribute
from src.labeling.common import LabelTypes
from src.encoding.time_encoding import TimeEncodingType
from src.log.common import get_log
from time import time
from Declare4Py.ProcessMiningTasks.Discovery.DeclareMiner import DeclareMiner
from Declare4Py.ProcessModels.DeclareModel import DeclareModel
from Declare4Py.D4PyEventLog import D4PyEventLog
import warnings

from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=DeprecationWarning)
from pycaret.classification import *
import numpy as np
import pm4py


# ---------------------------------------------------------------------------
# Diversity helpers
# ---------------------------------------------------------------------------

def pairwise_diversity(cf_solutions: list, n_features: int) -> float:
    """
    Mean normalised pairwise distance across all pairs in a top-k CF set.

    For categorical features: contribution = 1 if values differ, else 0.
    For float features (MinMax-scaled to [0,1]): contribution = |a - b|.

    Returns a scalar in [0, 1]:
        0  -> all CFs are identical
        1  -> every pair differs on every feature
    """
    if len(cf_solutions) < 2:
        return 0.0

    total, count = 0.0, 0
    for i in range(len(cf_solutions)):
        for j in range(i + 1, len(cf_solutions)):
            a, b = cf_solutions[i], cf_solutions[j]
            diff = 0.0
            for ai, bi in zip(a, b):
                if isinstance(ai, (float, np.floating)):
                    diff += abs(float(ai) - float(bi))
                else:
                    diff += 0.0 if ai == bi else 1.0
            total += diff / n_features
            count += 1

    return total / count


def pairwise_diversity_matrix(cf_solutions: list, n_features: int) -> np.ndarray:
    """
    Full symmetric (n_cfs x n_cfs) pairwise-distance matrix.
    Useful for downstream analysis / visualisation.
    """
    n   = len(cf_solutions)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = cf_solutions[i], cf_solutions[j]
            diff = 0.0
            for ai, bi in zip(a, b):
                if isinstance(ai, (float, np.floating)):
                    diff += abs(float(ai) - float(bi))
                else:
                    diff += 0.0 if ai == bi else 1.0
            d = diff / n_features
            mat[i, j] = d
            mat[j, i] = d
    return mat


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(CONF):
    import pandas as pd
    random.seed(CONF['seed'])
    np.random.seed(CONF['seed'])

    # ------------------------------------------------------------------ #
    # top_k_cfs: number of distinct counterfactuals to keep per trace     #
    # ------------------------------------------------------------------ #
    TOP_K = CONF.get('top_k_cfs', 5)

    dataset_name = CONF['data'].split('/')[-1].split('.')[0]
    log          = get_log(filepath=CONF['data'])
    encoding     = CONF['feature_selection']

    # ----- protected attribute assignment -----
    for trace in log:
        if 'bpi2012' in dataset_name:
            trace.attributes['protected'] = trace.attributes['gender'] == 'male'
        elif 'cs' in dataset_name:
            trace.attributes['protected'] = ((trace.attributes['gender'] == 'male'))
        elif 'hb_+age_+gender' in dataset_name:
            trace.attributes['protected'] = (
                (trace.attributes.get('gender') == 'female') and
                (trace.attributes.get('age', 0) > 85.0)
            )
        elif 'hb_-age_+gender' in dataset_name:
            trace.attributes['protected'] = trace.attributes['gender'] == 'female'
        elif 'hb_+age_-gender' in dataset_name:
            trace.attributes['protected'] = trace.attributes['age'] > 85.0
        elif 'hb_-age_-gender' in dataset_name:
            trace.attributes['protected'] = (
                (trace.attributes['gender'] == 'female') and
                (trace.attributes['age'] > 85.0)
            )
    protected_cases   = [t.attributes['concept:name'] for t in log if t.attributes['protected'] is True]
    unprotected_cases = [t.attributes['concept:name'] for t in log if t.attributes['protected'] is False]

    log_df = pm4py.convert_to_dataframe(log)
    prefix_length = CONF['prefix_length']

    # ----- dataset-specific preprocessing -----
    if 'bpi2012' in dataset_name:
        log_df.drop(columns=['case:protected'], inplace=True)
    elif 'cs' in dataset_name:
        log_df.drop(columns=['time', '@@index', '@@case_index'], inplace=True)
        cases_to_remove = (
            log_df[
                (log_df['case:gender'] == 'male') &
                (log_df['concept:name'] == 'mammary screening')
            ]['case:concept:name'].unique()
        )
        log_df = log_df[~log_df['case:concept:name'].isin(cases_to_remove)]
    elif any(tag in dataset_name for tag in ['hb_+age_+gender', 'hb_-age_+gender',
                                              'hb_+age_-gender', 'hb_-age_-gender']):
        log_df = log_df.dropna(axis=1)
        log_df.drop(columns=['@@index', '@@case_index'], inplace=True)

    # ── Numerical attribute categorization helpers ────────────────────────────────
    # Applied on unique case values first, then mapped back to all event rows,
    # to avoid pd.cut distortion from repeated rows in log_df.
    def categorize_age(log_df, case_id_col):
        """
        Age → life-stage category:
          childhood (0-12), adolescence (13-17), youth (18-24),
          young_adult (25-34), middle_aged_adult (35-49),
          mature_adult (50-64), older_adult (65+)
        """
        age_per_case = (
            log_df.drop_duplicates(case_id_col)
            .set_index(case_id_col)['case:age']
        )
        cut = pd.cut(
            age_per_case,
            bins=[0, 12, 17, 24, 34, 49, 64, float('inf')],
            labels=['childhood', 'adolescence', 'youth',
                    'young_adult', 'middle_aged_adult', 'mature_adult', 'older_adult'],
            right=True,
        )
        return log_df[case_id_col].map(cut)

    def categorize_education(log_df, case_id_col):
        """
        yearsOfEducation → formal education level:
          no_formal (0), elementary (1-8), secondary (9-12),
          higher (13-16), graduate (17+)
        """
        edu_per_case = (
            log_df.drop_duplicates(case_id_col)
            .set_index(case_id_col)['case:yearsOfEducation']
        )
        cut = pd.cut(
            edu_per_case,
            bins=[-1, 0, 8, 12, 16, float('inf')],
            labels=['no_formal', 'elementary', 'secondary', 'higher', 'graduate'],
            right=True,
        )
        return log_df[case_id_col].map(cut)

    # ── Dataset-specific preprocessing ───────────────────────────────────────────

    if 'lending' in dataset_name:
        activity_col, case_id_col, label_col = 'concept:name', 'case:concept:name', 'case:label'
        neg_label, pos_label = 'false', 'true'

        log_df['case:gender'] = log_df['case:gender'].map({True: 'male', False: 'female'})
        log_df['case:citizen'] = log_df['case:citizen'].map({True: 'citizen', False: 'non-citizen'})
        log_df['case:german speaking'] = log_df['case:german speaking'].map(
            {True: 'german speaking', False: 'non german speaking'})

        if 'case:age' in log_df.columns:
            log_df['case:age'] = categorize_age(log_df, case_id_col)
        if 'case:yearsOfEducation' in log_df.columns:
            log_df['case:yearsOfEducation'] = categorize_education(log_df, case_id_col)

        drop_extra = ['case:case', 'activity', '@@index', 'case:@@case_index', 'case:protected']
        if CONF['feature_selection'] == EncodingType.COMPLEX.value:
            drop_extra.append('time')
        log_df.drop(columns=drop_extra, inplace=True)

        true_event = log_df[log_df[activity_col] == 'Sign Loan Agreement'].groupby(case_id_col).any()[activity_col]
        log_df[label_col] = log_df[case_id_col].map(true_event).fillna(neg_label)
        log_df[label_col].replace(True, pos_label, inplace=True)


    elif 'hospital' in dataset_name:
        activity_col, case_id_col, label_col = 'concept:name', 'case:concept:name', 'case:label'
        neg_label, pos_label = 'false', 'true'

        log_df['case:protected'] = log_df['protected']
        log_df['case:gender'] = log_df['case:gender'].map({True: 'male', False: 'female'})
        log_df['case:citizen'] = log_df['case:citizen'].map({True: 'citizen', False: 'non-citizen'})
        log_df['case:german speaking'] = log_df['case:german speaking'].map(
            {True: 'german speaking', False: 'non german speaking'})
        log_df['case:private_insurance'] = log_df['case:private_insurance'].map({True: 'yes', False: 'no'})
        log_df['case:underlying_condition'] = log_df['case:underlying_condition'].map({True: 'true', False: 'false'})

        if 'case:age' in log_df.columns:
            log_df['case:age'] = categorize_age(log_df, case_id_col)

        log_df[label_col] = log_df['case:protected']
        log_df[label_col].replace(True, pos_label, inplace=True)
        log_df[label_col].replace(False, neg_label, inplace=True)
        log_df.drop(columns='protected', inplace=True)

        log = pm4py.convert_to_event_log(log_df)
        protected_cases = [t.attributes['concept:name'] for t in log if t.attributes['protected'] is True]
        unprotected_cases = [t.attributes['concept:name'] for t in log if t.attributes['protected'] is False]

        drop_extra = ['case:case', 'activity', '@@index', 'case:@@case_index', 'case:protected']
        if CONF['feature_selection'] == EncodingType.COMPLEX.value:
            drop_extra.append('time')
        log_df.drop(columns=drop_extra, inplace=True)

        label_df = pd.read_csv('hospital_log_labelled.csv')
        label_df['case_id'] = label_df['case_id'].astype(str)
        log_df[case_id_col] = log_df[case_id_col].astype(str)

        label_map = label_df.set_index('case_id')['mishap_no_expert_examination']
        log_df['case:label'] = (
            log_df[case_id_col]
            .map(label_map)  # int 0 or 1
            .map({1: 'true', 0: 'false'})  # string pm4py expects
        )


    elif 'renting' in dataset_name:
        activity_col, case_id_col, label_col = 'concept:name', 'case:concept:name', 'case:label'
        neg_label, pos_label = 'false', 'true'

        log_df['case:gender'] = log_df['case:gender'].map({True: 'male', False: 'female'})
        log_df['case:citizen'] = log_df['case:citizen'].map({True: 'citizen', False: 'non-citizen'})
        log_df['case:german speaking'] = log_df['case:german speaking'].map(
            {True: 'german speaking', False: 'non german speaking'})
        log_df['case:married'] = log_df['case:married'].map({True: 'married', False: 'not married'})

        if 'case:age' in log_df.columns:
            log_df['case:age'] = categorize_age(log_df, case_id_col)
        if 'case:yearsOfEducation' in log_df.columns:
            log_df['case:yearsOfEducation'] = categorize_education(log_df, case_id_col)

        protected_cases = [t.attributes['concept:name'] for t in log if t.attributes['protected'] is True]
        unprotected_cases = [t.attributes['concept:name'] for t in log if t.attributes['protected'] is False]

        drop_extra = ['case:case', 'activity', '@@index', 'case:@@case_index', 'case:protected']
        if CONF['feature_selection'] == EncodingType.COMPLEX.value:
            drop_extra.append('time')
        log_df.drop(columns=drop_extra, inplace=True)

        true_event = log_df[log_df[activity_col] == 'Reject Prospective Tenant'].groupby(case_id_col).any()[
            activity_col]
        log_df[label_col] = log_df[case_id_col].map(true_event).fillna(pos_label)
        log_df[label_col].replace(True, neg_label, inplace=True)


    elif 'hiring' in dataset_name:
        activity_col, case_id_col, label_col = 'concept:name', 'case:concept:name', 'case:label'
        neg_label, pos_label = 'false', 'true'

        log_df['case:gender'] = log_df['case:gender'].map({True: 'male', False: 'female'})
        log_df['case:citizen'] = log_df['case:citizen'].map({True: 'citizen', False: 'non-citizen'})
        log_df['case:religious'] = log_df['case:religious'].map({True: 'religious', False: 'non-religious'})
        log_df['case:german speaking'] = log_df['case:german speaking'].map(
            {True: 'german speaking', False: 'non german speaking'})

        if 'case:age' in log_df.columns:
            log_df['case:age'] = categorize_age(log_df, case_id_col)
        if 'case:yearsOfEducation' in log_df.columns:
            log_df['case:yearsOfEducation'] = categorize_education(log_df, case_id_col)

        drop_extra = ['case:case', 'activity', '@@index', 'case:@@case_index', 'case:protected']
        log_df.drop(columns=drop_extra, inplace=True)

        true_event = (
                log_df[log_df[activity_col] == 'Make Job Offer']
                .groupby(case_id_col).last()[activity_col] == 'Make Job Offer'
        )
        log_df[label_col] = log_df[case_id_col].map(true_event).fillna(neg_label)
        log_df[label_col].replace(True, pos_label, inplace=True)
        log_df[label_col].replace(False, neg_label, inplace=True)
    #log_df.drop(columns=['case:protected'], inplace=True)
    #log_df.drop(columns=['case:age'], inplace=True)
    try:
        log_df['case:label'].replace(True,  'true',  inplace=True)
        log_df['case:label'].replace(False, 'false', inplace=True)
    except Exception:
        print("unable to convert labels")

    log = pm4py.convert_to_event_log(log_df)

    encoder, full_df = get_encoded_df(log=log, CONF=CONF)
    encoder.decode(full_df)

    log_filepath = CONF['data'].replace('.csv', '.xes')
    if not os.path.exists(CONF['data'].replace('.csv', 'xes')):
        pm4py.write_xes(log, log_filepath)

    #full_df.drop(columns=[c for c in full_df.columns if 'timestamp' in c], inplace=True)

    train_size, val_size, test_size = CONF['train_val_test_split']
    train_df, val_df, test_df = np.split(
        full_df,
        [int(train_size * len(full_df)),
         int((train_size + val_size) * len(full_df))]
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier

    import pandas as pd
    if 'bpi2012' in dataset_name:
        # Define categorical and numeric columns
        prefix = [col for col in full_df.columns if 'prefix' in col]
        cat_cols = ["gender"] + prefix
        num_cols = ["AMOUNT_REQ"]

        # Preprocessing
        preprocessor = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols)
        ])

        # Pipeline with MLP
        model = Pipeline([
            ("preprocess", preprocessor),
            ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500))
        ])

        # Train/test split
        X_train = drop_columns(train_df)
        y_train = train_df['label']
        X_val = drop_columns(val_df)
        y_val = val_df['label']
        X_test = drop_columns(test_df)
        y_test = test_df['label']
        #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        # Train
        model.fit(X_train, y_train)

        print("Accuracy:", model.score(X_test, y_test))
        y_pred = model.predict(X_test)
    elif 'hiring' in dataset_name:
        prefix = [col for col in full_df.columns if 'prefix' in col]
        cat_cols =['age','german speaking', 'gender', 'citizen', 'yearsOfEducation', 'religious']+ prefix
        num_cols = []
        label_col = 'label'
        # Preprocessing
        preprocessor = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            #("num", StandardScaler(), num_cols)
        ])

        # Pipeline with MLP
        if CONF['predictive_model'] == 'mlp':
            model = Pipeline([
                ("preprocess", preprocessor),
                ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500))
            ])
        elif CONF['predictive_model'] == 'xgboost':
            model = Pipeline([(
                "preprocess", preprocessor
            ),
                ('xgboost', XGBClassifier())])
        elif CONF['predictive_model'] == 'catboost':
            model = Pipeline([(
                "preprocess", preprocessor
            ),
                ('catboost', CatBoostClassifier(iterations = 250, learning_rate=0.1))])        # Train/test split
        X_train = drop_columns(train_df)
        y_train = train_df['label'].replace({'false': 0, 'true': 1})
        X_val = drop_columns(val_df)
        y_val = val_df['label']
        X_test = drop_columns(test_df)
        y_test = test_df['label']
        y_test = test_df['label'].replace({'false': 0, 'true': 1})
        #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        # Train
        model.fit(X_train, y_train)

        print("Accuracy:", model.score(X_test, y_test))
        y_pred = model.predict(X_test)

    if   'lending'  in dataset_name: sensitive_attributes = ['age','citizen', 'gender', 'german speaking', 'yearsOfEducation']
    #elif 'hiring'   in dataset_name: sensitive_attributes = ['age','german speaking', 'gender', 'citizen', 'yearsOfEducation', 'religious']
    elif 'hiring'   in dataset_name: sensitive_attributes = ['gender']
    elif 'hospital' in dataset_name: sensitive_attributes = ['citizen','private_insurance','underlying_condition','german speaking','gender']
    elif 'renting'  in dataset_name: sensitive_attributes = ['german speaking', 'gender', 'citizen', 'married', 'age']
    elif 'bpi2012'  in dataset_name: sensitive_attributes = ['gender']
    elif 'cs'       in dataset_name: sensitive_attributes = ['gender', 'age']
    elif 'hb'       in dataset_name: sensitive_attributes = ['age', 'gender']

    event_log = D4PyEventLog(case_name="case:concept:name")

    if dataset_name == 'bpi2012':
        log_filepath= '/fairness_logs/bpi2012.xes'
        event_log.parse_xes_log(log_filepath)
    else:
        event_log.parse_xes_log(log_filepath)
    for i,trace in enumerate(event_log.log):
        trace._list = trace._list[:CONF['prefix_length']-1]
        event_log.log[i] = trace
    support    = 0.9
    model_path = f"process_models/{dataset_name}_discovered_model_{support}_prefix_{CONF['prefix_length']}.decl"

    if not os.path.exists(model_path):
        discovery = DeclareMiner(
            log=event_log, consider_vacuity=False,
            min_support=support, itemsets_support=support, max_declare_cardinality=2
        )
        discovered_model: DeclareModel = discovery.run()
        discovered_model.to_file(model_path)
    else:
        discovered_model = DeclareModel().parse_from_file(model_path)

    protected_mask = (
            (test_df['gender'] == 'female') &
            (test_df['trace_id'].isin(protected_cases))
    ).values
    unprotected_mask = (
            (test_df['gender'] != 'female') &
            (test_df['trace_id'].isin(unprotected_cases))
    ).values
    DESIRED_CLASS    = encoder._label_dict['label']['true']

    # ---- Feature importance ----
    def save_feature_importance(model, val_df, dataset_name, fairness_tag, CONF):
        X_val, y_val = drop_columns(val_df), val_df["label"]
        y_val = y_val.replace({'false':0, 'true':1})
        model_type   = CONF['predictive_model']
        r = permutation_importance(model, X_val, y_val,
                                   n_repeats=20, random_state=CONF["seed"])
        imp_df = pd.DataFrame({"feature": X_val.columns,
                               "importance_mean": r.importances_mean,
                               "importance_std":  r.importances_std})
        method = "PFI"

        fname = f"edoc_paper/synthetic_logs/t_s/models/{dataset_name}_{model_type}_{method}_{fairness_tag}_feature_importance_prefix_{CONF['prefix_length']}.csv"
        imp_df.sort_values("importance_mean", ascending=False).to_csv(fname, index=False)
        print(f"[INFO] Saved feature importance → {fname}")

    save_feature_importance(model, val_df, dataset_name, CONF['fairness_opt'], CONF)

    protected_positive_rate   = np.mean(y_pred[protected_mask]   == encoder._label_dict['label']['true'])
    unprotected_positive_rate = np.mean(y_pred[unprotected_mask] == encoder._label_dict['label']['true'])
    stat_par = protected_positive_rate - unprotected_positive_rate
    disp_imp = (protected_positive_rate / unprotected_positive_rate
                if unprotected_positive_rate > 0 else np.nan)

    print("\n=== Fairness Evaluation on Test Set ===")
    print(f"Protected positive rate:   {protected_positive_rate:.4f}")
    print(f"Unprotected positive rate: {unprotected_positive_rate:.4f}")
    print(f"Statistical Parity Diff:   {stat_par:.4f}")
    print(f"Disparate Impact:          {disp_imp:.4f}")

    from sklearn.neighbors import NearestNeighbors

    # Get only positive class
    train_cf_df = train_df[train_df["label"] == 'true']

    # Drop non-feature columns
    X_train_cf_raw = drop_columns(train_cf_df)

    # 🔥 Transform using SAME pipeline preprocessing
    X_train_cf_transformed = model.named_steps["preprocess"].transform(X_train_cf_raw)

    # Fit NN in transformed space
    nn_cf = NearestNeighbors(n_neighbors=5, metric="euclidean")
    nn_cf.fit(X_train_cf_transformed)
    encoded_columns = list(full_df.drop(columns=["trace_id", "label"]).columns)
    cat_maps = {}
    inv_cat_maps = {}

    X_ref = drop_columns(train_df)

    for col in X_ref.columns:
        if X_ref[col].dtype == "object":
            values = sorted(X_ref[col].dropna().unique())
            cat_maps[col] = {v: i for i, v in enumerate(values)}
            inv_cat_maps[col] = {i: v for v, i in cat_maps[col].items()}
    gene_space = []
    gene_types = []

    for col in X_ref.columns:

        # CATEGORICAL → integers
        if col in cat_maps:
            values = list(cat_maps[col].values())
            gene_space.append(values)
            gene_types.append(int)

        # NUMERIC
        else:
            col_min = X_ref[col].min()
            col_max = X_ref[col].max()

            gene_space.append({"low": float(col_min), "high": float(col_max)})
            gene_types.append([float, 8])
    def manifold_distance(solution):
        cf_df = solution_to_df(solution, X_train.columns, inv_cat_maps)
        cf_transformed = model.named_steps["preprocess"].transform(cf_df)
        dist, _ = nn_cf.kneighbors(cf_transformed.reshape(1, -1), return_distance=True)
        return dist.mean()

    def vector_to_pm4py_trace(solution):
        df = pd.DataFrame([solution], columns=encoded_columns)
        encoder.decode(df)
        row   = df.iloc[0]
        trace = Trace()
        for attr in sensitive_attributes:
            trace.attributes[attr] = df[attr]
        prefix_cols = sorted(
            [c for c in df.columns if c.startswith("prefix_")],
            key=lambda x: int(re.findall(r"\d+", x)[0])
        )
        for col in prefix_cols:
            act = row[col]
            if pd.isna(act) or act == 0:
                continue
            ev = Event({"concept:name": act})
            for attr in sensitive_attributes:
                ev[attr] = df[attr]
            trace.append(ev)
        return trace

    def solution_to_df(solution, columns, inv_cat_maps):
        row = {}

        for i, col in enumerate(columns):
            val = solution[i]

            if col in inv_cat_maps:
                row[col] = inv_cat_maps[col][int(val)]
            else:
                row[col] = val

        return pd.DataFrame([row])
    def solutions_to_df(solutions, encoded_columns, encoder):
        df = pd.DataFrame(solutions, columns=encoded_columns)
        encoder.decode(df)
        return df

    def mixed_distance(sol, orig, columns, model, inv_cat_maps):
        # Convert both to DataFrames
        sol_df = solution_to_df(sol, columns, inv_cat_maps)
        orig_df = solution_to_df(orig, columns, inv_cat_maps)

        # Apply SAME preprocessing as MLP
        sol_trans = model.named_steps["preprocess"].transform(sol_df)
        orig_trans = model.named_steps["preprocess"].transform(orig_df)

        # Euclidean distance in transformed space
        dist = np.linalg.norm(sol_trans - orig_trans)

        # Normalize by number of features
        return dist / sol_trans.shape[1]
    def declare_conformance(solution):
        cf_trace = vector_to_pm4py_trace(solution)
        if len(cf_trace) == 0:
            return 0.0
        res       = ConstraintChecker().check_trace_conformance(
            trace=cf_trace, decl_model=discovered_model, consider_vacuity=False,
        )

        satisfied = [r for r in res if r.state == "Satisfied"]
        return len(satisfied) / len(res)

    def strict_sparsity(original, counterfactual, columns, cat_maps, X_ref, epsilon_scale=0.3):
        changes = 0

        for i, col in enumerate(columns):
            ov = original[i]
            cv = counterfactual[i]

            # CATEGORICAL
            if col in cat_maps:
                if int(ov) != int(cv):
                    changes += 1

        return changes / len(original)

    def sensitive_change_count(original, counterfactual, sensitive_attributes):
        changes = 0
        changed_attributes = []
        for attr in sensitive_attributes:
            idx  = encoded_columns.index(attr)
            ov   = original[idx]
            cv   = counterfactual[idx]
            if isinstance(ov, (float, np.floating)):
                epsilon = 0.8 / encoder._numeric_encoder[attr].data_range_
                if abs(ov - cv) > epsilon:
                    changes += 1
                    changed_attributes.append(attr)
            else:
                if ov != int(cv):
                    changed_attributes.append(attr)
                    changes += 1
        return changes / len(sensitive_attributes),changed_attributes

    from itertools import combinations, product as iproduct

    def sensitive_attribute_intervention_analysis_inv_maps(max_k=2):
        """
        Intervene on combinations of sensitive attributes.
        Uses inv_cat_maps for interpretability.
        """

        results = []
        detailed_changes = []

        # Reverse maps: category -> encoded value
        cat_maps = {
            col: {v: k for k, v in inv.items()}
            for col, inv in inv_cat_maps.items()
        }

        feat_names = list(X_test.columns)

        # limit k
        max_k = min(max_k, len(sensitive_attributes))

        # generate combinations
        attr_combos = []
        for r in range(1, max_k + 1):
            attr_combos.extend(combinations(sensitive_attributes, r))

        for attrs in attr_combos:

            present_attrs = [a for a in attrs if a in feat_names and a in cat_maps]
            if not present_attrs:
                continue

            flip_total = 0
            flip_prot = 0
            flip_unprot = 0

            total = 0
            total_prot = 0
            total_unprot = 0

            for i in range(len(X_test)):

                x_orig = X_test.iloc[i].copy()
                original_pred = model.predict(pd.DataFrame([x_orig]))[0]

                # collect candidate values per attribute
                per_attr_candidates = []

                for attr in present_attrs:

                    current_val = x_orig[attr]
                    possible_vals = list(cat_maps[attr].keys())

                    candidates = [v for v in possible_vals if v != current_val]

                    if not candidates:
                        per_attr_candidates = []
                        break

                    per_attr_candidates.append([(attr, v) for v in candidates])

                if not per_attr_candidates:
                    continue

                total += 1
                if protected_mask[i]:
                    total_prot += 1
                    group = "protected"
                else:
                    total_unprot += 1
                    group = "unprotected"

                flipped = False

                # generate joint interventions
                for combo in iproduct(*per_attr_candidates):

                    x_cf = x_orig.copy()

                    for attr, val in combo:
                        x_cf[attr] = val

                    cf_pred = model.predict(pd.DataFrame([x_cf]))[0]

                    # log (you can trim this if too large)
                    detailed_changes.append({
                        "index": i,
                        "attributes": "+".join(present_attrs),
                        "group": group,
                        "original_pred": original_pred,
                        "cf_pred": cf_pred,
                        "changes": {a: v for a, v in combo}
                    })

                    if cf_pred != original_pred:
                        flipped = True
                        break

                if flipped:
                    flip_total += 1
                    if protected_mask[i]:
                        flip_prot += 1
                    else:
                        flip_unprot += 1

            results.append({
                "attribute": "+".join(present_attrs),
                "flip_rate_total": flip_total / max(1, total),
                "flip_rate_protected": flip_prot / max(1, total_prot),
                "flip_rate_unprotected": flip_unprot / max(1, total_unprot),
                "n_samples": total
            })

        return pd.DataFrame(results), pd.DataFrame(detailed_changes)

    cf_flip_df, res_df = sensitive_attribute_intervention_analysis_inv_maps(max_k=3)
    #cf_flip_df = counterfactual_flip_analysis()
    flip_path = f'edoc_paper/flip_analysis/{model.steps[1][0]}_fairness_flip_results_{dataset_name}_{encoding}_fixed_sensitive_attrs_prefix_{prefix}.csv'
    cf_flip_df.to_csv(flip_path, mode='a', index=False, header=not os.path.exists(flip_path))
    res_file_path = f'edoc_paper/flip_analysis/{model.steps[1][0]}_detailed_flip_results_{dataset_name}_{encoding}_fixed_sensitive_attrs_prefix_{prefix}.csv'
    res_df.to_csv(res_file_path, mode='a', index=False, header=not os.path.exists(flip_path))

    print('Prefix length', CONF['prefix_length'])
    import shap
#    explainer = shap.Explainer(predictive_model.model)
#    shap_values = explainer(test_df.iloc[:,1:-1])
#    shap.plots.bar(shap_values)
    def df_to_encoded(row, columns, cat_maps):
        encoded = []

        for col in columns:
            val = row[col]

            if col in cat_maps:
                encoded.append(cat_maps[col][val])
            else:
                encoded.append(float(val))

        return np.array(encoded)

    fairness_weight = 0.5 if CONF['fairness_opt'] else 0.0
    # ================================================================== #
    # Main loop over protected / unprotected cases                        #
    # ================================================================== #
    results      = []
    final_cf_dfs = pd.DataFrame()
    cf_dfs       = pd.DataFrame(columns=test_df.columns)

    for case in ['protected']:

        #mask = protected_cases if case == 'protected' else unprotected_cases
        mask = protected_mask if case == 'protected' else unprotected_mask
        test_cases = test_df[mask]
        test_cases = test_cases[test_cases['label'] != 'true']

        if test_cases.empty:
            print(f"No test cases found for {case}. Skipping.")
            continue

        def fitness_function(ga_instance, solution, solution_idx):

            cf_df = solution_to_df(solution, X_train.columns, inv_cat_maps)
            x_original_df = drop_columns(test_cases.iloc[[index_trace]])
            x_original = df_to_encoded(
                x_original_df.iloc[0],
                X_train.columns,
                cat_maps
            )
            original_solution = x_original
            prob = model.predict_proba(cf_df)[0, 1]
            pred = model.predict(cf_df)[0]

            try:
                prob = model.predict_proba(cf_df)[0, 1]
                pred = model.predict(cf_df)[0]
            except Exception:
                return -1.0

            if pred != DESIRED_CLASS:
                return -1.0

            pred = 1.0

            proximity = mixed_distance(solution, original_solution,X_train.columns, model, inv_cat_maps)
            sparsity = np.sum(solution != original_solution) / len(original_solution)
            d_plaus = manifold_distance(solution) / len(solution)
            conformance = declare_conformance(solution)

            if CONF['fairness_opt']:
                fairness, changed_attributes = sensitive_change_count(
                    original_solution,
                    solution,
                    sensitive_attributes
                )

                fitness = (
                        pred
                        - 0.5 * proximity
                        - 0.5 * sparsity
                        - 0.5 * d_plaus
                        - 1.0 * (1 - conformance)
                        - fairness_weight * fairness
                )
            else:
                fitness = (
                        pred
                        - 0.5 * proximity
                        - 0.5 * sparsity
                        - 0.5 * d_plaus
                        - 1.0 * (1 - conformance)
                )

            return fitness
        for index_trace in range(min(10, len(test_cases))):

            x_original          = test_cases.iloc[index_trace].values[1:-1]
            original_prediction = test_cases.iloc[index_trace][-1]
            trace_id            = test_cases.iloc[index_trace][0]
            x_original = test_cases.iloc[index_trace].values[1:-1]
            x_original_df = drop_columns(test_cases.iloc[[index_trace]])

            # Lock sensitive attribute indices to original values
            sensitive_indices = [
                encoded_columns.index(attr)
                for attr in sensitive_attributes
                if attr in encoded_columns
            ]

            # Build gene_space with sensitive attrs locked to their original value
            '''
            locked_gene_space = []
            for i, g in enumerate(gene_space):
                if i in sensitive_indices:
                    if encoded_columns[i] in encoder._label_dict:
                        locked_gene_space.append(int(x_original[i]))
                    else:
                        locked_gene_space.append(x_original[i])# single-value list = locked
                else:
                    locked_gene_space.append(g)
            # ---- fitness function (closure over x_original) ----
            '''

            # ---- collect valid (flipping) solutions each generation ----
            valid_solutions: list = []   # list of (fitness, np.ndarray)
            def on_generation(ga_instance):
                sol, fit, _ = ga_instance.best_solution()
                if fit > -1.0:
                    arr = np.array(sol).astype(X_test.values.dtype)
                    valid_solutions.append((float(fit), arr.copy()))

            start = time()
            ga = pygad.GA(
                num_generations=50,
                fitness_func=fitness_function,
                on_generation=on_generation,
                random_seed=42,
                save_best_solutions=False,
                sol_per_pop=100,
                num_parents_mating=40,
                num_genes=len(gene_space),
                gene_space=gene_space,
                gene_type=gene_types
            )
            ga.run()
            runtime = time() - start

            # ---------------------------------------------------------- #
            # Select top-k: sort by fitness, deduplicate, cap at TOP_K    #
            # ---------------------------------------------------------- #
            valid_solutions.sort(key=lambda x: x[0], reverse=True)
            seen_hashes: set = set()
            top_k_solutions: list = []

            for fit_val, sol in valid_solutions:
                key = tuple(sol.tolist())
                if key not in seen_hashes:
                    seen_hashes.add(key)
                    top_k_solutions.append((fit_val, sol))
                if len(top_k_solutions) >= TOP_K:
                    break

            # Fallback: use GA best if no valid solution was collected
            if not top_k_solutions:
                best_sol, best_fit, _ = ga.best_solution()
                best_sol = np.array(best_sol).astype(X_test.values.dtype)
                top_k_solutions = [(float(best_fit), best_sol)]

            # ---------------------------------------------------------- #
            # Diversity metrics for the top-k set                         #
            # ---------------------------------------------------------- #
            topk_arrays = [s for _, s in top_k_solutions]
            diversity   = pairwise_diversity(topk_arrays, n_features=len(encoded_columns))
            div_matrix  = pairwise_diversity_matrix(topk_arrays, n_features=len(encoded_columns))
            print(f"\n[Trace {index_trace} | {case}]  "
                  f"k={len(top_k_solutions)}  mean_pairwise_diversity={diversity:.4f}")
            # ---------------------------------------------------------- #
            # Record each CF in the top-k                                 #
            # ---------------------------------------------------------- #
            for k_rank, (fit_val, solution) in enumerate(top_k_solutions):
                #cf_pred = model.predict(solution)[0]
                cf_df = solution_to_df(solution, X_train.columns, inv_cat_maps)
                cf_pred = model.predict(cf_df)[0]
                x_original_df = drop_columns(test_cases.iloc[[index_trace]])
                x_original = x_original_df.iloc[0]
                x_orig = df_to_encoded(
                    x_original,
                    X_train.columns,
                    cat_maps
                )

                proximity    = mixed_distance(solution, x_orig,X_train.columns, model, inv_cat_maps)
                sparsity     = np.sum(solution != x_orig) / len(x_orig)
                sparsity_new = strict_sparsity(x_orig, solution, X_train.columns, model, inv_cat_maps)
                d_plaus      = manifold_distance(solution) / len(solution)
                conformance  = declare_conformance(solution)
                fairness,_     = sensitive_change_count(x_orig, solution, sensitive_attributes)
                # Build decoded trace row
                full_row       = np.zeros(len(test_df.columns))
                full_row[0]    = index_trace
                full_row[1:-1] = solution.copy()
                #full_row[-1]   = DESIRED_CLASS

                for prefix_col in [c for c in cf_df.columns if c.startswith("prefix_")]:
                    pos = prefix_col.split("_")[1]
                    if cf_df.at[0, prefix_col] == 0:
                        for col in [c for c in cf_df.columns if c.endswith(f"_{pos}")]:
                            cf_df.at[0, col] = 0

               # encoder.decode(cf_df)

                cf_df['trace_id'] = index_trace
                cf_df['label'] = cf_pred
                cf_df['group']            = case
                cf_df['predictive_model'] = CONF['predictive_model']
                cf_df['cf_rank']          = k_rank
                cf_df['top_k_diversity']  = diversity
                # Build original row
                orig_row = x_original.copy()
                orig_row = pd.concat([pd.Series([index_trace]), orig_row])
                orig_df = pd.DataFrame([orig_row], columns=test_df.columns)

                # Build counterfactual row
                #encoder.decode(orig_df)
                #encoder.decode(cf_df_temp)
                changed_attributes = {}

                for attr in orig_df.columns[1:-1]:
                    orig_val = orig_df.iloc[0][attr]
                    cf_val = cf_df.iloc[0][attr]

                    if orig_val != cf_val:
                        changed_attributes[attr] = {
                            "original": orig_val,
                            "counterfactual": cf_val
                        }

                cf_dfs = pd.concat([cf_dfs, cf_df], ignore_index=True)
                print(f"  CF#{k_rank}: pred={cf_pred}  prox={proximity:.3f}  "
                      f"spar={sparsity:.3f}  conf={conformance:.3f}  "
                      f"plaus={d_plaus:.3f}  unfair={fairness:.3f}  fit={fit_val:.3f}, changed attributes={changed_attributes}")
                results.append({
                    'dataset': dataset_name,
                    'encoding': encoding,
                    'model': CONF['predictive_model'],
                    'trace_id': index_trace,
                    'cf_rank': k_rank,  # rank within top-k (0 = best)
                    'prefix_length': CONF['prefix_length'],
                    'original_prediction': original_prediction,
                    'counterfactual_prediction': cf_pred,
                    'proximity': proximity,
                    'sparsity': sparsity,
                    'alternative_sparsity': sparsity_new,
                    'conformance': conformance,
                    'plausibility': d_plaus,
                    'unfairness': fairness,
                    'fitness': fit_val,
                    'time': runtime,
                    'case': case,
                    'fairness_weight': fairness_weight,
                    'top_k_diversity': diversity,
                    'changed_attributes': changed_attributes}) # mean pairwise diversity of set                })


    final_cf_dfs = pd.concat([final_cf_dfs, cf_dfs], ignore_index=True)
    results_df   = pd.DataFrame(results)
    prefix_length = CONF['prefix_length']
    # ---- Output paths ----
    suffix = 'with' if CONF['fairness_opt'] == True else 'without'
    file_path   = (f'edoc_paper/{dataset_name}/counterfactual_results_{suffix}_fairness_optimization_diff_models_{encoding}_{prefix_length}.csv')
    traces_path = (f'edoc_paper/{dataset_name}/counterfactual_traces_{suffix}_fairness_{encoding}_prefix_length_{prefix_length}.csv')

    results_df.to_csv(file_path,    mode='a', index=False, header=not os.path.exists(file_path))
    final_cf_dfs.to_csv(traces_path, mode='a', index=False, header=not os.path.exists(traces_path))

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    datasets = [
        d for d in os.listdir('/Users/andrei/Desktop/FBK/fairness_cfs/fairness_logs/')
        if d.endswith('.csv')
    ]
    #HOSPITAL MAX 5 Prefix
    #for dataset in datasets[4:]:
    datasets = ['hiring_log_high-xes.xes','hiring_log_medium-xes.xes','hiring_log_low-xes.xes']
    for dataset in datasets:
        for model in ['mlp']:
            for fairness in [False]:
                for enc in [EncodingType.SIMPLE_TRACE.value, EncodingType.COMPLEX.value]:
                    for prefix in [4]:
                        CONF = {
                            'data': '/Users/andrei/Desktop/FBK/fairness_cfs/fairness_logs/' + dataset,
                            'train_val_test_split': [0.7, 0.15, 0.15],
                            'output': os.path.join('..', 'output_data'),
                            'prefix_length_strategy': PrefixLengthStrategy.FIXED.value,
                            'prefix_length': prefix,
                            'padding': True,
                            'feature_selection': enc,
                            'task_generation_type': TaskGenerationType.ONLY_THIS.value,
                            'attribute_encoding': EncodingTypeAttribute.ONEHOT.value,
                            'labeling_type': LabelTypes.ATTRIBUTE_STRING.value,
                            'predictive_model': model,
                            'threshold': 13,
                            'top_k': 10,
                            # -------------------------------------------- #
                            # top_k_cfs: distinct counterfactuals per trace  #
                            # -------------------------------------------- #
                            'top_k_cfs': 5,
                            'hyperparameter_optimisation': False,
                            'hyperparameter_optimisation_target': HyperoptTarget.F1.value,
                            'hyperparameter_optimisation_epochs': 1,
                            'time_encoding': TimeEncodingType.NONE.value,
                            'fairness_opt': fairness,
                            'target_event': None,
                            'seed': 42,
                        }
                        run_pipeline(CONF)