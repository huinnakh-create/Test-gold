import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import date

st.set_page_config(page_title="Gold Fair Value Intelligence", page_icon="🟡", layout="wide")

st.title("🟡 Gold Fair Value Intelligence")
st.caption("نموذج اقتصادي كمي لتقدير القيمة العادلة للذهب — ليس توصية تداول ولا ضماناً للسعر.")

st.info(
    "الإصدار الأول يستخدم بيانات اقتصادية شهرية من FRED. "
    "القيمة الناتجة تقدير إحصائي مشروط بالمتغيرات المختارة، وليست قيمة جوهرية مؤكدة للذهب. "
    "بيانات مشتريات البنوك المركزية وتدفقات ETF ستُضاف في إصدار لاحق بعد ربط مصادرها."
)

with st.sidebar:
    st.header("الإعدادات")
    secret_key = st.secrets.get("FRED_API_KEY", "") if "FRED_API_KEY" in st.secrets else ""
    api_key = st.text_input("مفتاح FRED API", value=secret_key, type="password", help="أدخل مفتاحك من fred.stlouisfed.org/docs/api/api_key.html")
    train_pct = st.slider("نسبة بيانات التدريب", 60, 90, 80, 5)
    start_date = st.date_input("بداية البيانات", value=date(2006, 1, 1))
    st.caption("لا تُدخل المفتاح في ملفات عامة. عند النشر ضعه في Secrets.")

SERIES = {
    "GOLD": ("GOLDPMGBD228NLBM", "سعر الذهب LBMA PM (دولار/أونصة)"),
    "REAL10Y": ("DFII10", "العائد الحقيقي الأمريكي لعشر سنوات (%)"),
    "USD": ("DTWEXBGS", "مؤشر الدولار التجاري الواسع (بديل DXY)"),
    "BEI10": ("T10YIE", "تعويض التضخم لعشر سنوات (%)"),
    "VIX": ("VIXCLS", "مؤشر VIX")
}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred(series_id, key, observation_start):
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": key,
        "file_type": "json",
        "observation_start": observation_start,
        "observation_end": date.today().isoformat()
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if "observations" not in payload:
        raise ValueError(payload.get("error_message", "تعذر جلب البيانات"))
    frame = pd.DataFrame(payload["observations"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.set_index("date")["value"].replace(".", np.nan).dropna()

def load_data(key, start):
    series = {}
    for label, (sid, _) in SERIES.items():
        series[label] = fetch_fred(sid, key, start.isoformat()).resample("ME").mean()
    data = pd.concat(series, axis=1).dropna()
    data = data[(data["GOLD"] > 0) & (data["USD"] > 0) & (data["VIX"] > 0)]
    data["LOG_GOLD"] = np.log(data["GOLD"])
    data["LOG_USD"] = np.log(data["USD"])
    data["LOG_VIX"] = np.log(data["VIX"])
    return data

if not api_key:
    st.warning("أدخل مفتاح FRED API في الشريط الجانبي لبدء جلب البيانات.")
    st.markdown("**الحصول على مفتاح مجاني:** [FRED API Keys](https://fred.stlouisfed.org/docs/api/api_key.html)")
    st.stop()

try:
    with st.spinner("جارٍ تحميل البيانات الاقتصادية من FRED..."):
        data = load_data(api_key, start_date)
except Exception as exc:
    st.error(f"تعذر تحميل البيانات: {exc}")
    st.markdown("تحقق من مفتاح FRED واتصال الإنترنت وتوفر السلاسل المطلوبة.")
    st.stop()

if len(data) < 60:
    st.error("عدد المشاهدات المتاحة قليل جداً لبناء نموذج زمني موثوق. وسّع الفترة التاريخية.")
    st.stop()

features = ["REAL10Y", "LOG_USD", "BEI10", "LOG_VIX"]
X = data[features]
y = data["LOG_GOLD"]

split = max(24, int(len(data) * train_pct / 100))
if split >= len(data):
    split = len(data) - 1

model = LinearRegression()
model.fit(X.iloc[:split], y.iloc[:split])
pred_log = model.predict(X.iloc[split:])
actual_test = y.iloc[split:]
pred_test = np.exp(pred_log)
actual_test_price = np.exp(actual_test)

mae = mean_absolute_error(actual_test_price, pred_test)
rmse = np.sqrt(mean_squared_error(actual_test_price, pred_test))
r2 = r2_score(actual_test, pred_log)

latest = data.iloc[[-1]][features]
fair_value = float(np.exp(model.predict(latest)[0]))
actual = float(data["GOLD"].iloc[-1])
deviation = (actual / fair_value - 1) * 100

# Residual-based indicative interval from training residuals
train_resid = y.iloc[:split] - model.predict(X.iloc[:split])
resid_sd = float(train_resid.std(ddof=1))
lower = float(np.exp(np.log(fair_value) - 1.96 * resid_sd))
upper = float(np.exp(np.log(fair_value) + 1.96 * resid_sd))

st.subheader("القراءة الحالية")
c1, c2, c3, c4 = st.columns(4)
c1.metric("آخر سعر ذهب متاح من FRED", f"${actual:,.0f}/oz", help="سعر LBMA PM؛ قد لا يكون سعراً لحظياً.")
c2.metric("القيمة العادلة المقدّرة", f"${fair_value:,.0f}/oz")
c3.metric("الانحراف عن القيمة العادلة", f"{deviation:+.2f}%", help="موجب = السعر أعلى من تقدير النموذج؛ سالب = أدنى.")
c4.metric("نطاق التقدير الإرشادي 95%", f"${lower:,.0f} – ${upper:,.0f}")

if actual > upper:
    st.warning("السعر أعلى من النطاق الإرشادي للنموذج. هذا لا يعني تلقائياً أن البيع مناسب.")
elif actual < lower:
    st.success("السعر أدنى من النطاق الإرشادي للنموذج. هذا لا يعني تلقائياً أن الشراء مناسب.")
else:
    st.info("السعر داخل النطاق الإرشادي للنموذج.")

st.subheader("العوامل الاقتصادية المستخدمة")
factor_cols = st.columns(4)
latest_row = data.iloc[-1]
labels = [
    ("العائد الحقيقي 10 سنوات", f"{latest_row['REAL10Y']:.2f}%"),
    ("الدولار الواسع (بديل DXY)", f"{latest_row['USD']:.2f}"),
    ("تعويض التضخم 10 سنوات", f"{latest_row['BEI10']:.2f}%"),
    ("VIX", f"{latest_row['VIX']:.2f}")
]
for col, (label, value) in zip(factor_cols, labels):
    col.metric(label, value)

st.subheader("السعر الفعلي مقابل القيمة التي يتنبأ بها النموذج")
all_pred = np.exp(model.predict(X))
chart = go.Figure()
chart.add_trace(go.Scatter(x=data.index, y=data["GOLD"], name="السعر الفعلي", line=dict(width=2)))
chart.add_trace(go.Scatter(x=data.index, y=all_pred, name="القيمة الضمنية للنموذج", line=dict(width=2, dash="dash")))
chart.update_layout(height=480, xaxis_title="التاريخ", yaxis_title="دولار لكل أونصة", hovermode="x unified")
st.plotly_chart(chart, use_container_width=True)

st.subheader("اختبار النموذج خارج عينة التدريب")
m1, m2, m3 = st.columns(3)
m1.metric("MAE", f"${mae:,.0f}/oz")
m2.metric("RMSE", f"${rmse:,.0f}/oz")
m3.metric("R² على بيانات الاختبار", f"{r2:.3f}")

st.caption(
    f"عدد المشاهدات الشهرية: {len(data)} | بيانات التدريب: {split} | الاختبار: {len(data)-split}. "
    "مقاييس الاختبار لا تضمن أداءً مستقبلياً. النطاق الإرشادي مبني على تشتت بواقي التدريب ويفترض تقريباً ثباتها."
)

with st.expander("معاملات النموذج وحدوده"):
    coef = pd.DataFrame({"المتغير": features, "المعامل": model.coef_})
    st.dataframe(coef, use_container_width=True, hide_index=True)
    st.write(f"الثابت: {model.intercept_:.5f}")
    st.write(
        "هذا إصدار أولي: يستخدم متوسطات شهرية وانحداراً خطياً لوغاريتمياً. "
        "مؤشر الدولار المستخدم هو DTWEXBGS، وهو مؤشر الدولار التجاري الواسع وليس DXY حرفياً. "
        "سعر الذهب من سلسلة LBMA PM في FRED وليس سعراً لحظياً. "
        "لا يتضمن الإصدار الحالي مشتريات البنوك المركزية أو ETF أو الإنتاج وإعادة التدوير أو علاوة المخاطر الجيوسياسية مباشرة."
    )

st.download_button(
    "تنزيل البيانات المستخدمة CSV",
    data=data.to_csv().encode("utf-8-sig"),
    file_name="gold_fair_value_data.csv",
    mime="text/csv"
)
