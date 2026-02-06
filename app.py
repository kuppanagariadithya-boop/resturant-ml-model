import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Page Title
st.set_page_config(page_title="Restaurant Rating Prediction", layout="wide")

st.title("🍽️ Restaurant Rating Prediction System")
st.write("Machine Learning Project using Linear Regression")

# Load Data
@st.cache_data
def load_data():
    candidates = [
        Path("restaurant_data.csv"),
        Path("venv/Restuarent_Data.csv"),
    ]
    errors = []

    for path in candidates:
        if not path.exists():
            continue
        try:
            with path.open("rb") as handle:
                signature = handle.read(4)
        except OSError as exc:
            errors.append(f"{path}: {exc}")
            continue

        if signature[:2] == b"PK":
            try:
                return pd.read_excel(path)
            except ImportError:
                errors.append(f"{path}: missing dependency 'openpyxl' for Excel files")
            except Exception as exc:
                errors.append(f"{path}: {exc}")

        for encoding in ("utf-8", "latin1"):
            try:
                return pd.read_csv(path, encoding=encoding)
            except Exception as exc:
                errors.append(f"{path} ({encoding}): {exc}")

    raise RuntimeError("Unable to load data. " + "; ".join(errors))


data = load_data()
data = data.loc[:, ~data.columns.str.contains(r"^Unnamed")]
data.columns = [col.strip() for col in data.columns]
data = data.rename(
    columns={
        "Average cost": "Average_Cost",
        "Average_cost": "Average_Cost",
    }
)

# Show Dataset
st.subheader("📊 Dataset Preview")
st.dataframe(data.head())


# Handle Missing Values
data = data.fillna(0)

# Encode Categorical Data
encoder = LabelEncoder()
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = encoder.fit_transform(data[col])


# Separate X and Y
X = data.drop("Aggregate rating", axis=1)
y = data["Aggregate rating"]


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train Model
model = LinearRegression()
model.fit(X_train, y_train)


# Prediction
y_pred = model.predict(X_test)


# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# Show Performance
st.subheader("📈 Model Performance")

col1, col2 = st.columns(2)

col1.metric("Mean Squared Error", round(mse, 3))
col2.metric("R2 Score", round(r2, 3))


# =========================
# VISUALIZATION SECTION
# =========================

st.subheader("📉 Data Visualization")

# 1. Rating Distribution
st.write("### Rating Distribution")

fig1, ax1 = plt.subplots()
sns.histplot(data["Aggregate rating"], bins=10, kde=True, ax=ax1)
ax1.set_xlabel("Rating")
ax1.set_ylabel("Count")

st.pyplot(fig1)


# 2. Votes vs Rating
st.write("### Votes vs Rating")

fig2, ax2 = plt.subplots()
sns.scatterplot(
    x=data["Votes"],
    y=data["Aggregate rating"],
    ax=ax2
)

ax2.set_xlabel("Votes")
ax2.set_ylabel("Rating")

st.pyplot(fig2)


# 3. Cost vs Rating
st.write("### Cost vs Rating")

fig3, ax3 = plt.subplots()
sns.scatterplot(
    x=data["Average_Cost"],
    y=data["Aggregate rating"],
    ax=ax3
)

ax3.set_xlabel("Average Cost")
ax3.set_ylabel("Rating")

st.pyplot(fig3)


# =========================
# FEATURE IMPORTANCE
# =========================

st.subheader("📌 Feature Importance")

importance = pd.Series(model.coef_, index=X.columns)
importance = importance.sort_values(ascending=False)

fig4, ax4 = plt.subplots()
importance.plot(kind="bar", ax=ax4)

ax4.set_ylabel("Importance")

st.pyplot(fig4)


# =========================
# USER PREDICTION
# =========================

st.subheader("🔮 Predict New Rating")

st.write("Enter Restaurant Details:")

inputs = {}

for col in X.columns:
    value = st.number_input(f"{col}", value=0.0)
    inputs[col] = value


if st.button("Predict Rating"):
    input_df = pd.DataFrame([inputs])
    prediction = model.predict(input_df)

    st.success(f"⭐ Predicted Rating: {round(prediction[0],2)}")
