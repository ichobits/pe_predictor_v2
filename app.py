import joblib
import numpy as np
import pandas as pd
import streamlit as st
from autogluon.tabular import TabularPredictor

st.set_page_config(page_title="GDM-PE Predictor", layout="wide")

MODEL_NAMES = ['RandomForestGini', 'RandomForestEntr', 'ExtraTreesGini', 'ExtraTreesEntr',
               'XGBoost', 'LightGBM', 'LightGBMLarge', 'LightGBMXT',
               'CatBoost', 'NeuralNetTorch', 'WeightedEnsemble_L2']

@st.cache_resource
def load_artifact(key):
    if key == 'asv':
        return (TabularPredictor.load('models/asv_only'),
                joblib.load('models/asv_scaler.pkl'),
                joblib.load('models/asv_cols.pkl'))
    if key == 'clin':
        return (TabularPredictor.load('models/clin_only'), None,
                joblib.load('models/clin_cols.pkl'))
    return (TabularPredictor.load('models/combined'),
            joblib.load('models/asv_scaler.pkl'),
            joblib.load('models/asv_cols.pkl'))

@st.cache_data
def load_taxonomy():
    return pd.read_csv('asv_taxonomy.csv')

def clr_transform(counts: np.ndarray) -> np.ndarray:
    x = np.log(counts + 1.0)
    return x - x.mean(axis=1, keepdims=True)


class AutogluonWrapper:
    """让 KernelExplainer 拿到类别1概率的包装器（借自旧项目）"""
    def __init__(self, predictor, feature_names, model_name):
        self.ag_model = predictor
        self.feature_names = feature_names
        self.model_name = model_name

    def predict_binary_prob(self, X):
        if isinstance(X, pd.Series):
            X = X.values.reshape(1, -1)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        return self.ag_model.predict_proba(X, model=self.model_name, as_multiclass=False)

# ---------- Default values (CLR + z-scored medians; clinical reference values) ----------
ASV_DEFAULTS_Z = {
    'ASV_21031': -0.18666755388698686,
    'ASV_24175': -0.17267762039173432,
    'ASV_10655': -0.23112077312008267,
    'ASV_23486': 0.07610394957138156,
    'ASV_4992':  -0.07120371874200937,
}
CLIN_META = {
    'Proteinuria_Dipstick':     ('Proteinuria dipstick (0-4)',       0.0,   4.0,   0.0),
    'Triglycerides':            ('Triglycerides (mmol/L)',           0.0,   10.0,  3.1),
    'Total_Cholesterol':        ('Total cholesterol (mmol/L)',       0.0,   15.0,  5.39),
    'Max_DBP_During_Pregnancy': ('Max DBP during pregnancy (mmHg)',  40.0,  140.0, 80.0),
    'BMI':                      ('Pre-pregnancy BMI (kg/m²)',       15.0,  50.0,  23.205),
    'Max_SBP_During_Pregnancy': ('Max SBP during pregnancy (mmHg)',  70.0,  220.0, 120.0),
    'D_Dimer':                  ('D-dimer (mg/L)',                   0.0,   20.0,  1.0),
    'Neonatal_Length':          ('Neonatal length (cm)',             30.0,  60.0,  50.0),
    'Total_Protein':            ('Total protein (g/L)',              30.0,  90.0,  70.0),
}

# ---------- UI ----------
st.title("Preeclampsia Risk Prediction in GDM")
st.caption("Integrative gut-microbiome (ASV) and clinical model — for research use only")

mode = st.sidebar.selectbox("Input data type",
                            ["Combined ASV + Clinical", "Clinical only", "ASV only"])
model_name = st.sidebar.selectbox("Model (RandomForestGini recommended)", MODEL_NAMES, index=0)
need_asv = mode != "Clinical only"
need_clin = mode != "ASV only"

asv_cols = joblib.load('models/asv_cols.pkl')
clin_cols = joblib.load('models/clin_cols.pkl')

asv_vals, clin_vals = [], {}
raw_mode = False
if need_asv:
    st.subheader("Fecal ASV features (n = 5)")
    with st.expander("📋 Taxonomic annotation of the 5 ASVs", expanded=False):
        st.dataframe(load_taxonomy(), use_container_width=True, hide_index=True)
    raw_mode = st.checkbox(
        "My inputs are raw read counts (apply CLR transformation + standardization automatically)",
        value=False,
        help="Leave unchecked to enter CLR- and z-score-transformed values (consistent with model training inputs).")
    c = st.columns(5)
    for i, col in enumerate(asv_cols):
        if raw_mode:
            asv_vals.append(c[i].number_input(col, min_value=0, value=100, step=10, key=col))
        else:
            asv_vals.append(c[i].number_input(col, value=float(ASV_DEFAULTS_Z.get(col, 0.0)),
                                              step=0.01, format="%.4f", key=col))

if need_clin:
    st.subheader("Clinical variables (n = 9)")
    c = st.columns(3)
    for i, col in enumerate(clin_cols):
        label, lo, hi, dv = CLIN_META.get(col, (col, 0.0, 100.0, 1.0))
        clin_vals[col] = c[i % 3].number_input(label, min_value=lo, max_value=hi,
                                               value=float(dv), key='clin_' + col)

if st.button("Predict", type="primary"):
    key = 'combined' if need_asv and need_clin else ('asv' if need_asv else 'clin')
    pred_obj, scaler, _ = load_artifact(key)
    parts = []
    if need_asv:
        if raw_mode:
            arr = clr_transform(np.array([asv_vals], dtype=float))
            arr = scaler.transform(arr)
        else:
            arr = np.array([asv_vals], dtype=float)
        parts.append(pd.DataFrame(arr, columns=asv_cols))
    if need_clin:
        parts.append(pd.DataFrame([clin_vals]))
    X = pd.concat(parts, axis=1)

    proba = pred_obj.predict_proba(X, model=model_name)
    p_pe = float(proba[1].iloc[0])
    st.metric("Predicted probability of PE", f"{p_pe:.1%}")
    st.progress(min(max(p_pe, 0.0), 1.0))
    if p_pe >= 0.5:
        st.error(f"Predicted class: PE risk ({model_name})")
    else:
        st.success(f"Predicted class: NP ({model_name})")

    with st.expander("SHAP explanation of this prediction (optional, may take ~1 min)"):
        import shap
        from streamlit_shap import st_shap

        shap.initjs()
        bg = pd.read_csv(f'models/bg_{key}.csv')          # 真正的背景：测试集中位数
        wrapper = AutogluonWrapper(pred_obj, list(X.columns), model_name)
        explainer = shap.KernelExplainer(wrapper.predict_binary_prob, bg)
        sv = explainer.shap_values(X)
        st_shap(shap.force_plot(explainer.expected_value, sv, X), height=400, width=1000)