import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
    roc_curve, auc
)
import matplotlib.pyplot as plt
import joblib
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# PAGE CONFIG
st.set_page_config(
    page_title="Hybrid Machine Learning Platform",
    layout="wide"
)

st.title("Hybrid Machine Learning Platform")

st.markdown("""
### Welcome

This platform provides a complete end-to-end Machine Learning workflow.

Upload -> Preprocess -> Feature Selection -> Supervised Learning -> Unsupervised Learning -> Visualization -> Prediction
""")

st.divider()

st.subheader("Dashboard")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Modules", "7")
with col2:
    st.metric("Algorithms", "9+")
with col3:
    st.metric("Charts", "13+")
with col4:
    st.metric("Dataset", "CSV / Excel")

st.divider()

st.subheader("Machine Learning Workflow")
st.info("""
1. Upload Dataset
2. Data Preprocessing
3. Feature Selection
4. Supervised Learning
5. Unsupervised Learning
6. Visualization
7. Prediction
""")

st.divider()

# =====================================================
# UPLOAD DATASET
# =====================================================
st.header("Upload Dataset")

uploaded_file = st.file_uploader(
    "Choose CSV or Excel File",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            try:
                df = pd.read_csv(uploaded_file)
            except Exception:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="latin1")
        else:
            df = pd.read_excel(uploaded_file)

        st.success("Dataset Uploaded Successfully")
        st.session_state["df"] = df

    except Exception as e:
        st.error(f"Could not read the file: {e}")

# DATASET PREVIEW
if "df" in st.session_state:
    df = st.session_state["df"]

    st.divider()
    st.header("Dataset Preview")
    st.dataframe(df.head())

    st.divider()
    st.header("Dataset Information")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Rows", df.shape[0])
    with c2:
        st.metric("Columns", df.shape[1])
    with c3:
        st.metric("Missing Values", int(df.isnull().sum().sum()))

    st.subheader("Column Names")
    st.write(df.columns.tolist())

    st.subheader("Data Types")
    st.dataframe(df.dtypes.astype(str))

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum())

    st.subheader("Duplicate Rows")
    st.metric("Duplicates", int(df.duplicated().sum()))

else:
    st.warning("Please upload a dataset.")

# =====================================================
# DATA PREPROCESSING
# =====================================================
if "df" in st.session_state:

    st.divider()
    st.header("Data Preprocessing")

    df = st.session_state["df"].copy()

    # ------------------------------
    # Missing Value Handling
    # ------------------------------
    st.subheader("Missing Value Handling")

    option = st.selectbox(
        "Select Method",
        ["None", "Drop Missing Rows", "Fill Mean", "Fill Median", "Fill Mode"]
    )

    if st.button("Apply Missing Value Handling"):
        try:
            if option == "Drop Missing Rows":
                df = df.dropna()
            elif option == "Fill Mean":
                numeric = df.select_dtypes(include=np.number).columns
                for col in numeric:
                    df[col] = df[col].fillna(df[col].mean())
            elif option == "Fill Median":
                numeric = df.select_dtypes(include=np.number).columns
                for col in numeric:
                    df[col] = df[col].fillna(df[col].median())
            elif option == "Fill Mode":
                for col in df.columns:
                    mode_vals = df[col].mode()
                    if not mode_vals.empty:
                        df[col] = df[col].fillna(mode_vals[0])
                    else:
                        # Column is entirely missing - drop it rather than crash
                        df = df.drop(columns=[col])

            st.success("Missing Values Processed")
            st.session_state["df"] = df
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Missing value handling failed: {e}")

    # ------------------------------
    # Duplicate Rows
    # ------------------------------
    st.subheader("Duplicate Rows")

    duplicate = df.duplicated().sum()
    st.metric("Duplicate Rows", int(duplicate))

    if st.button("Remove Duplicate Rows"):
        df = df.drop_duplicates()
        st.success("Duplicates Removed")
        st.session_state["df"] = df
        st.dataframe(df.head())

    # ------------------------------
    # Remove Columns
    # ------------------------------
    st.subheader("Remove Columns")

    remove_cols = st.multiselect("Select Columns", df.columns, key="remove_cols")

    if st.button("Remove Selected Columns"):
        if remove_cols:
            df = df.drop(columns=remove_cols)
            st.session_state["df"] = df
            st.success("Columns Removed")
            st.dataframe(df.head())
        else:
            st.info("Select at least one column first.")

    # ------------------------------
    # Rename Column
    # ------------------------------
    st.subheader("Rename Column")

    old = st.selectbox("Select Column", df.columns, key="rename")
    new = st.text_input("New Name")

    if st.button("Rename"):
        if new.strip() != "":
            df = df.rename(columns={old: new})
            st.session_state["df"] = df
            st.success("Column Renamed")
            st.write(df.columns.tolist())
        else:
            st.info("Enter a new column name.")

    # ------------------------------
    # Label Encoding
    # ------------------------------
    st.subheader("Label Encoding")

    cat_cols = df.select_dtypes(include="object").columns.tolist()

    if len(cat_cols) > 0:
        encode_col = st.selectbox("Select Categorical Column", cat_cols, key="label_encoding")

        if st.button("Apply Label Encoding"):
            encoder = LabelEncoder()
            df[encode_col] = encoder.fit_transform(df[encode_col].astype(str))
            st.session_state["df"] = df
            st.success("Label Encoding Completed")
            st.dataframe(df.head())
    else:
        st.info("No categorical columns found.")

    # ------------------------------
    # Feature Scaling
    # ------------------------------
    st.subheader("Feature Scaling")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) == 0:
        st.info("No numeric columns available for scaling.")
    else:
        selected_cols = st.multiselect("Select Numeric Columns", numeric_cols, key="scaling")
        scaler_type = st.selectbox("Scaling Method", ["StandardScaler", "MinMaxScaler"])

        if st.button("Apply Scaling"):
            if len(selected_cols) > 0:
                if df[selected_cols].isnull().values.any():
                    st.error("Selected columns contain missing values. Handle missing values before scaling.")
                else:
                    scaler = StandardScaler() if scaler_type == "StandardScaler" else MinMaxScaler()
                    df[selected_cols] = scaler.fit_transform(df[selected_cols])
                    st.session_state["df"] = df
                    st.success("Scaling Completed")
                    st.dataframe(df.head())
            else:
                st.info("Select at least one numeric column.")

    # ------------------------------
    # Outlier Removal
    # ------------------------------
    st.subheader("Outlier Removal")

    numeric = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric) == 0:
        st.info("No numeric columns available for outlier removal.")
    else:
        outlier_col = st.selectbox("Select Column", numeric, key="outlier")

        if st.button("Remove Outliers"):
            Q1 = df[outlier_col].quantile(0.25)
            Q3 = df[outlier_col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            df = df[(df[outlier_col] >= lower) & (df[outlier_col] <= upper)]

            st.session_state["df"] = df
            st.success("Outliers Removed")
            st.write("Rows Remaining :", df.shape[0])
            st.dataframe(df.head())

    # ------------------------------
    # Download Dataset
    # ------------------------------
    st.subheader("Download Processed Dataset")

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Processed Dataset",
        data=csv,
        file_name="processed_dataset.csv",
        mime="text/csv"
    )

# =====================================================
# FEATURE SELECTION
# =====================================================
if "df" in st.session_state:

    st.divider()
    st.header("Feature Selection")

    df = st.session_state["df"]

    st.subheader("Select Target Column")
    target = st.selectbox("Target Column", df.columns, key="target_column")

    # If the target changed and previously chosen features now conflict
    # (feature == target, or feature no longer exists), clean them up
    # BEFORE rendering the multiselect widget, to avoid a Streamlit crash.
    if "feature_columns" in st.session_state:
        cleaned = [
            c for c in st.session_state["feature_columns"]
            if c != target and c in df.columns
        ]
        if cleaned != st.session_state["feature_columns"]:
            st.session_state["feature_columns"] = cleaned

    st.subheader("Select Feature Columns")
    features = st.multiselect(
        "Feature Columns",
        [col for col in df.columns if col != target],
        key="feature_columns"
    )

    X, y = None, None

    if len(features) > 0:
        try:
            X = df[features].copy()
            y = df[target].copy()

            feature_encoders = {}
            for col in X.select_dtypes(include="object").columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                feature_encoders[col] = le

            target_encoder = None
            if y.dtype == "object":
                target_encoder = LabelEncoder()
                y = target_encoder.fit_transform(y.astype(str))

            if X.isnull().values.any():
                st.warning("Selected features contain missing values. Consider handling them in preprocessing.")

            st.session_state["X"] = X
            st.session_state["y"] = y
            st.session_state["features"] = features
            st.session_state["target"] = target
            st.session_state["feature_encoders"] = feature_encoders
            st.session_state["target_encoder"] = target_encoder

            st.success("Features Selected Successfully")
            st.write("X Shape :", X.shape)
            st.write("y Shape :", y.shape)
            st.dataframe(X.head())
        except Exception as e:
            st.error(f"Feature selection failed: {e}")

    # ------------------------------
    # Correlation Matrix
    # ------------------------------
    st.subheader("Correlation Matrix")

    numeric_df = df.select_dtypes(include="number")

    if st.button("Generate Correlation Matrix"):
        if numeric_df.shape[1] < 2:
            st.info("Need at least 2 numeric columns for a correlation matrix.")
        else:
            correlation = numeric_df.corr()
            st.dataframe(correlation)
            fig = px.imshow(correlation, text_auto=True, aspect="auto")
            st.plotly_chart(fig, use_container_width=True)

    # ------------------------------
    # SelectKBest
    # ------------------------------
    st.subheader("SelectKBest")

if X is None:

    st.info("Please upload a dataset first.")

elif len(features) < 2:

    st.warning("Select at least 2 feature columns to use SelectKBest.")

else:

    k_best = st.slider(
        "Number of Features",
        min_value=1,
        max_value=len(features),
        value=min(2, len(features))
    )

    if st.button("Run SelectKBest"):

        selector = SelectKBest(
            score_func=f_classif,
            k=k_best
        )

        selector.fit(X, y)

        selected_features = X.columns[
            selector.get_support()
        ]

        st.success("Selected Features")

        st.write(selected_features.tolist())

    # ------------------------------
    # Feature Importance
    # ------------------------------
    st.subheader("Feature Importance")

    if len(features) > 0 and X is not None:
        if st.button("Generate Feature Importance"):
            try:
                fi_model = RandomForestClassifier(random_state=42)
                fi_model.fit(X, y)
                importance = pd.DataFrame({
                    "Feature": X.columns,
                    "Importance": fi_model.feature_importances_
                }).sort_values(by="Importance", ascending=False)
                st.dataframe(importance)
                st.bar_chart(importance.set_index("Feature"))
            except Exception as e:
                st.error(f"Feature importance failed: {e}")
    else:
        st.info("Select feature columns above first.")

    # ------------------------------
    # Download Selected Dataset
    # ------------------------------
    st.subheader("Download Selected Dataset")

    if len(features) > 0:
        selected_df = df[features + [target]]
        csv = selected_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Selected Dataset",
            data=csv,
            file_name="selected_dataset.csv",
            mime="text/csv"
        )

# =====================================================
# SUPERVISED LEARNING
# =====================================================
st.divider()
st.header("Supervised Learning")

if "X" in st.session_state and "y" in st.session_state:

    X = st.session_state["X"]
    y = st.session_state["y"]

    test_size = st.slider("Test Size (%)", 10, 40, 20)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size / 100, random_state=42
        )

        st.subheader("Dataset Split")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Training Samples", len(X_train))
            st.metric("Training Labels", len(y_train))
        with c2:
            st.metric("Testing Samples", len(X_test))
            st.metric("Testing Labels", len(y_test))

        st.subheader("Choose Machine Learning Model")

        model_name = st.selectbox(
            "Algorithm",
            ["Logistic Regression", "KNN", "Decision Tree",
             "Random Forest", "Support Vector Machine", "Naive Bayes"]
        )

        if st.button("Train Model"):
            try:
                if model_name == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000)
                elif model_name == "KNN":
                    model = KNeighborsClassifier()
                elif model_name == "Decision Tree":
                    model = DecisionTreeClassifier(random_state=42)
                elif model_name == "Random Forest":
                    model = RandomForestClassifier(random_state=42)
                elif model_name == "Support Vector Machine":
                    model = SVC(probability=True)
                else:
                    model = GaussianNB()

                if pd.Series(y_train).nunique() < 2:
                    st.error("Training data only contains a single class. Choose a different target column or split.")
                else:
                    model.fit(X_train, y_train)
                    prediction = model.predict(X_test)
                    accuracy = accuracy_score(y_test, prediction)

                    st.success("Model Trained Successfully")
                    st.metric("Accuracy", f"{accuracy*100:.2f}%")

                    result = pd.DataFrame({"Actual": y_test, "Predicted": prediction})
                    st.subheader("Prediction Sample")
                    st.dataframe(result.head(20))

                    st.session_state["trained_model"] = model
                    st.session_state["model_name"] = model_name
                    st.session_state["X_train"] = X_train
                    st.session_state["X_test"] = X_test
                    st.session_state["y_test"] = y_test
                    st.session_state["prediction"] = prediction
                    st.session_state["accuracy"] = accuracy
            except Exception as e:
                st.error(f"Model training failed: {e}")
    except Exception as e:
        st.error(f"Could not split the dataset: {e}")

else:
    st.info("Select target and feature columns in Feature Selection first.")

# =====================================================
# MODEL EVALUATION
# =====================================================
if "trained_model" in st.session_state:

    model = st.session_state["trained_model"]
    model_name = st.session_state["model_name"]
    X_train = st.session_state["X_train"]
    X_test = st.session_state["X_test"]
    y_test = st.session_state["y_test"]
    prediction = st.session_state["prediction"]
    accuracy = st.session_state["accuracy"]

    st.divider()
    st.subheader("Model Evaluation")

    precision = precision_score(y_test, prediction, average="weighted", zero_division=0)
    recall = recall_score(y_test, prediction, average="weighted", zero_division=0)
    f1 = f1_score(y_test, prediction, average="weighted", zero_division=0)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{accuracy*100:.2f}%")
    with col2:
        st.metric("Precision", f"{precision*100:.2f}%")
    with col3:
        st.metric("Recall", f"{recall*100:.2f}%")
    with col4:
        st.metric("F1 Score", f"{f1*100:.2f}%")

    st.subheader("Classification Report")
    report = classification_report(y_test, prediction, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, prediction)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Prediction Result")
    prediction_df = pd.DataFrame({"Actual": y_test, "Predicted": prediction})
    st.dataframe(prediction_df)

    st.subheader("Model Performance")
    performance = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Score": [accuracy, precision, recall, f1]
    })
    st.bar_chart(performance.set_index("Metric"))

    if model_name == "Random Forest":
        st.divider()
        st.subheader("Feature Importance")
        importance = pd.DataFrame({
            "Feature": X_train.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        st.dataframe(importance)
        st.bar_chart(importance.set_index("Feature"))

    if hasattr(model, "predict_proba"):
        st.divider()
        st.subheader("ROC Curve")

        try:
            probability = model.predict_proba(X_test)

            if len(model.classes_) == 2:
                fpr, tpr, threshold = roc_curve(y_test, probability[:, 1])
                roc_auc = auc(fpr, tpr)

                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
                ax.plot([0, 1], [0, 1], linestyle="--")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("ROC Curve is available only for binary classification.")

            st.divider()
            st.subheader("Prediction Probability")
            probability_df = pd.DataFrame(probability)
            st.dataframe(probability_df.head(10))
        except Exception as e:
            st.error(f"Could not compute probabilities: {e}")

    st.divider()
    st.subheader("Save Trained Model")

    try:
        joblib.dump(model, "trained_model.pkl")
        with open("trained_model.pkl", "rb") as file:
            st.download_button(
                label="Download Model",
                data=file,
                file_name="trained_model.pkl",
                mime="application/octet-stream"
            )
    except Exception as e:
        st.error(f"Could not save model: {e}")

# =====================================================
# MODEL COMPARISON
# =====================================================
if "X" in st.session_state and "y" in st.session_state:

    st.divider()
    st.header("Model Comparison")

    if st.button("Compare All Models"):
        try:
            X = st.session_state["X"]
            y = st.session_state["y"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            if pd.Series(y_train).nunique() < 2:
                st.error("Training data only contains a single class. Choose a different target column.")
            else:
                models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "KNN": KNeighborsClassifier(),
                    "Decision Tree": DecisionTreeClassifier(random_state=42),
                    "Random Forest": RandomForestClassifier(random_state=42),
                    "Support Vector Machine": SVC(),
                    "Naive Bayes": GaussianNB()
                }

                comparison = []
                for name, clf in models.items():
                    try:
                        clf.fit(X_train, y_train)
                        pred = clf.predict(X_test)

                        acc = accuracy_score(y_test, pred)
                        pre = precision_score(y_test, pred, average="weighted", zero_division=0)
                        rec = recall_score(y_test, pred, average="weighted", zero_division=0)
                        f1c = f1_score(y_test, pred, average="weighted", zero_division=0)

                        comparison.append({
                            "Model": name,
                            "Accuracy": round(acc * 100, 2),
                            "Precision": round(pre * 100, 2),
                            "Recall": round(rec * 100, 2),
                            "F1 Score": round(f1c * 100, 2)
                        })
                    except Exception as e:
                        st.warning(f"{name} failed: {e}")

                if comparison:
                    comparison_df = pd.DataFrame(comparison)
                    st.session_state["comparison_df"] = comparison_df
                    st.dataframe(comparison_df)
        except Exception as e:
            st.error(f"Model comparison failed: {e}")

    if "comparison_df" in st.session_state:
        comparison_df = st.session_state["comparison_df"]

        best = comparison_df.loc[comparison_df["Accuracy"].idxmax()]

        st.subheader("Best Model")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Model", best["Model"])
        with c2:
            st.metric("Accuracy", f'{best["Accuracy"]}%')

        st.subheader("Accuracy Comparison")
        chart = comparison_df.set_index("Model")
        st.bar_chart(chart["Accuracy"])

        csv = comparison_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Comparison Report",
            data=csv,
            file_name="Model_Comparison.csv",
            mime="text/csv"
        )

# =====================================================
# UNSUPERVISED LEARNING
# =====================================================
st.divider()
st.header("Unsupervised Learning")

if "df" in st.session_state:

    df = st.session_state["df"]
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_columns) < 2:
        st.warning("Dataset must contain at least two numeric columns.")
    else:
        selected_columns = st.multiselect(
            "Select Numeric Features",
            numeric_columns,
            default=numeric_columns[:2]
        )

        if len(selected_columns) >= 2 and not df[selected_columns].isnull().values.any():
            X_cluster = df[selected_columns]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_cluster)

            # ------------------------------
            # K-Means
            # ------------------------------
            st.subheader("K-Means")
            k = st.slider("Number of Clusters", 2, 10, 3)

            if st.button("Run KMeans"):
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(X_scaled)

                    silhouette = silhouette_score(X_scaled, labels)
                    davies = davies_bouldin_score(X_scaled, labels)
                    calinski = calinski_harabasz_score(X_scaled, labels)

                    st.subheader("KMeans Evaluation")
                    ecol1, ecol2, ecol3 = st.columns(3)
                    with ecol1:
                        st.metric("Silhouette Score", round(silhouette, 3))
                    with ecol2:
                        st.metric("Davies-Bouldin", round(davies, 3))
                    with ecol3:
                        st.metric("Calinski-Harabasz", round(calinski, 2))

                    cluster_df = X_cluster.copy()
                    cluster_df["Cluster"] = labels

                    st.success("KMeans Completed")
                    st.dataframe(cluster_df.head())
                    st.bar_chart(cluster_df["Cluster"].value_counts())

                    st.session_state["cluster_df"] = cluster_df
                    st.session_state["kmeans_score"] = silhouette
                    st.session_state["kmeans_model"] = kmeans
                    st.session_state["kmeans_labels"] = labels
                    st.session_state["cluster_selected_columns"] = selected_columns
                    st.session_state["X_scaled"] = X_scaled
                except Exception as e:
                    st.error(f"KMeans failed: {e}")

            # ------------------------------
            # Hierarchical Clustering
            # ------------------------------
            st.subheader("Hierarchical Clustering")
            linkage = st.selectbox("Linkage", ["ward", "complete", "average", "single"])

            if st.button("Run Hierarchical"):
                try:
                    hc = AgglomerativeClustering(n_clusters=k, linkage=linkage)
                    labels = hc.fit_predict(X_scaled)

                    silhouette = silhouette_score(X_scaled, labels)
                    davies = davies_bouldin_score(X_scaled, labels)
                    calinski = calinski_harabasz_score(X_scaled, labels)

                    st.subheader("Hierarchical Evaluation")
                    ecol1, ecol2, ecol3 = st.columns(3)
                    with ecol1:
                        st.metric("Silhouette Score", round(silhouette, 3))
                    with ecol2:
                        st.metric("Davies-Bouldin", round(davies, 3))
                    with ecol3:
                        st.metric("Calinski-Harabasz", round(calinski, 2))

                    cluster_df = X_cluster.copy()
                    cluster_df["Cluster"] = labels

                    st.success("Hierarchical Completed")
                    st.dataframe(cluster_df.head())
                    st.bar_chart(cluster_df["Cluster"].value_counts())

                    st.session_state["hierarchical_score"] = silhouette
                    st.session_state["hierarchical_labels"] = labels
                    st.session_state["hierarchical_cluster_df"] = cluster_df
                except Exception as e:
                    st.error(f"Hierarchical clustering failed: {e}")

            # ------------------------------
            # DBSCAN
            # ------------------------------
            st.subheader("DBSCAN")
            eps = st.slider("EPS", 0.1, 5.0, 0.5)
            min_samples = st.slider("Min Samples", 2, 20, 5)

            if st.button("Run DBSCAN"):
                try:
                    db = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = db.fit_predict(X_scaled)

                    real_clusters = set(labels) - {-1}

                    if len(real_clusters) > 1:
                        silhouette = silhouette_score(X_scaled, labels)
                        davies = davies_bouldin_score(X_scaled, labels)
                        calinski = calinski_harabasz_score(X_scaled, labels)

                        st.subheader("DBSCAN Evaluation")
                        ecol1, ecol2, ecol3 = st.columns(3)
                        with ecol1:
                            st.metric("Silhouette Score", round(silhouette, 3))
                        with ecol2:
                            st.metric("Davies-Bouldin", round(davies, 3))
                        with ecol3:
                            st.metric("Calinski-Harabasz", round(calinski, 2))

                        dbscan_score = silhouette
                    else:
                        dbscan_score = -1
                        st.warning(
                            "DBSCAN created only one cluster or only noise. Evaluation cannot be calculated. Try adjusting EPS / Min Samples."
                        )

                    cluster_df = X_cluster.copy()
                    cluster_df["Cluster"] = labels

                    st.success("DBSCAN Completed")
                    st.dataframe(cluster_df.head())
                    st.bar_chart(cluster_df["Cluster"].value_counts())

                    noise = int((labels == -1).sum())
                    st.metric("Noise Points", noise)

                    st.session_state["dbscan_score"] = dbscan_score
                    st.session_state["dbscan_labels"] = labels
                    st.session_state["dbscan_cluster_df"] = cluster_df
                except Exception as e:
                    st.error(f"DBSCAN failed: {e}")

            # ------------------------------
            # Best Clustering Algorithm
            # ------------------------------
            st.divider()
            st.subheader("Best Clustering Algorithm")

            scores = {
                "KMeans": st.session_state.get("kmeans_score", -1),
                "Hierarchical": st.session_state.get("hierarchical_score", -1),
                "DBSCAN": st.session_state.get("dbscan_score", -1)
            }

            best_algorithm = max(scores, key=scores.get)
            best_score = scores[best_algorithm]

            st.metric("Best Algorithm", best_algorithm)
            st.metric("Silhouette Score", round(best_score, 3))

            # ------------------------------
            # KMeans Visualization
            # ------------------------------
            if "kmeans_labels" in st.session_state:
                st.subheader("KMeans Cluster Visualization")

                cluster_cols = st.session_state.get("cluster_selected_columns", selected_columns)
                plot_df = pd.DataFrame(st.session_state["X_scaled"], columns=cluster_cols)
                plot_df["Cluster"] = st.session_state["kmeans_labels"]

                fig = px.scatter(
                    plot_df, x=cluster_cols[0], y=cluster_cols[1],
                    color=plot_df["Cluster"].astype(str), title="KMeans Clustering"
                )
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Cluster Centers")
                center_df = pd.DataFrame(
                    st.session_state["kmeans_model"].cluster_centers_,
                    columns=cluster_cols
                )
                st.dataframe(center_df)

                st.subheader("Cluster Distribution")
                st.bar_chart(plot_df["Cluster"].value_counts())

                st.subheader("Cluster Summary")
                summary = st.session_state["cluster_df"].groupby("Cluster").mean(numeric_only=True)
                st.dataframe(summary)

                st.subheader("Download Cluster Centers")
                csv = center_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Cluster Centers",
                    data=csv,
                    file_name="cluster_centers.csv",
                    mime="text/csv"
                )

            # ------------------------------
            # Hierarchical Visualization
            # ------------------------------
            if "hierarchical_labels" in st.session_state:
                st.subheader("Hierarchical Cluster Visualization")

                plot_df = pd.DataFrame(X_scaled, columns=selected_columns)
                plot_df["Cluster"] = st.session_state["hierarchical_labels"]

                fig = px.scatter(
                    plot_df, x=selected_columns[0], y=selected_columns[1],
                    color=plot_df["Cluster"].astype(str), title="Hierarchical Clustering"
                )
                st.plotly_chart(fig, use_container_width=True)

            # ------------------------------
            # DBSCAN Visualization
            # ------------------------------
            if "dbscan_labels" in st.session_state:
                st.subheader("DBSCAN Cluster Visualization")

                plot_df = pd.DataFrame(X_scaled, columns=selected_columns)
                plot_df["Cluster"] = st.session_state["dbscan_labels"]

                fig = px.scatter(
                    plot_df, x=selected_columns[0], y=selected_columns[1],
                    color=plot_df["Cluster"].astype(str), title="DBSCAN Clustering"
                )
                st.plotly_chart(fig, use_container_width=True)

            # ------------------------------
            # Download Clustered Dataset
            # ------------------------------
            st.divider()
            st.subheader("Download Clustered Dataset")

            if "cluster_df" in st.session_state:
                csv = st.session_state["cluster_df"].to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Clustered Dataset",
                    data=csv,
                    file_name="clustered_dataset.csv",
                    mime="text/csv"
                )
            else:
                st.info("Run a clustering algorithm first.")

            # ------------------------------
            # Clustering Comparison
            # ------------------------------
            st.divider()
            st.header("Clustering Comparison")

            comparison = pd.DataFrame({
                "Algorithm": ["KMeans", "Hierarchical", "DBSCAN"],
                "Silhouette Score": [scores["KMeans"], scores["Hierarchical"], scores["DBSCAN"]]
            })

            st.dataframe(comparison)
            st.bar_chart(comparison.set_index("Algorithm"))

            # ------------------------------
            # Clustering Report
            # ------------------------------
            st.divider()
            st.header("Clustering Report")

            st.success("Clustering Analysis Completed Successfully")
            st.write("Best Algorithm :", best_algorithm)
            st.write("Silhouette Score :", round(best_score, 3))
            st.write("Features Used")
            st.write(selected_columns)
            st.write("Number of Samples")
            st.write(len(df))
            st.write("Number of Features")
            st.write(len(selected_columns))
        elif len(selected_columns) >= 2:
            st.warning("Selected columns contain missing values. Handle them in preprocessing before clustering.")

# =====================================================
# VISUALIZATION DASHBOARD
# =====================================================
st.divider()
st.header("Visualization Dashboard")

if "df" in st.session_state:
    df = st.session_state["df"]
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    all_columns = df.columns.tolist()

    # Histogram
    st.subheader("Histogram")
    if numeric_columns:
        hist_column = st.selectbox("Select Column", numeric_columns, key="hist")
        if st.button("Generate Histogram"):
            fig = px.histogram(df, x=hist_column, nbins=30, title=f"Histogram - {hist_column}")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric columns available.")

    # Bar Chart
    st.subheader("Bar Chart")
    bar_column = st.selectbox("Select Column", all_columns, key="bar")
    if st.button("Generate Bar Chart"):
        chart = df[bar_column].value_counts().reset_index()
        chart.columns = [bar_column, "Count"]
        fig = px.bar(chart, x=bar_column, y="Count", title=f"Bar Chart - {bar_column}")
        st.plotly_chart(fig, use_container_width=True)

    # Line Chart
    st.subheader("Line Chart")
    if numeric_columns:
        line_column = st.selectbox("Select Numeric Column", numeric_columns, key="line")
        if st.button("Generate Line Chart"):
            fig = px.line(df, y=line_column, title=f"Line Chart - {line_column}")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric columns available.")

    # Data Summary
    st.subheader("Data Summary")
    st.dataframe(df.describe())

    # Scatter Plot
    st.divider()
    st.subheader("Scatter Plot")
    if len(numeric_columns) >= 2:
        x_axis = st.selectbox("Select X Axis", numeric_columns, key="scatter_x")
        y_axis = st.selectbox("Select Y Axis", numeric_columns, key="scatter_y")
        color_column = st.selectbox("Color By", ["None"] + all_columns, key="scatter_color")

        if st.button("Generate Scatter Plot"):
            if color_column == "None":
                fig = px.scatter(df, x=x_axis, y=y_axis, title="Scatter Plot")
            else:
                fig = px.scatter(df, x=x_axis, y=y_axis, color=color_column, title="Scatter Plot")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need at least 2 numeric columns for a scatter plot.")

    # Box Plot
    st.subheader("Box Plot")
    if numeric_columns:
        box_column = st.selectbox("Select Numeric Column", numeric_columns, key="box_plot")
        if st.button("Generate Box Plot"):
            fig = px.box(df, y=box_column, title=f"Box Plot - {box_column}")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric columns available.")

    # Pie Chart
    st.subheader("Pie Chart")
    pie_column = st.selectbox("Select Categorical Column", all_columns, key="pie_chart")
    if st.button("Generate Pie Chart"):
        pie_df = df[pie_column].value_counts().reset_index()
        pie_df.columns = [pie_column, "Count"]
        fig = px.pie(pie_df, names=pie_column, values="Count", title=f"Pie Chart - {pie_column}")
        st.plotly_chart(fig, use_container_width=True)

    # Area Chart
    st.subheader("Area Chart")
    if numeric_columns:
        area_column = st.selectbox("Select Numeric Column", numeric_columns, key="area_chart")
        if st.button("Generate Area Chart"):
            fig = px.area(df, y=area_column, title=f"Area Chart - {area_column}")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric columns available.")

    # Bubble Chart
    st.subheader("Bubble Chart")
    if len(numeric_columns) >= 3:
        bubble_x = st.selectbox("Bubble X", numeric_columns, key="bubble_x")
        bubble_y = st.selectbox("Bubble Y", numeric_columns, key="bubble_y")
        bubble_size = st.selectbox("Bubble Size", numeric_columns, key="bubble_size")
        if st.button("Generate Bubble Chart"):
            fig = px.scatter(df, x=bubble_x, y=bubble_y, size=bubble_size, title="Bubble Chart")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need at least 3 numeric columns for a bubble chart.")

    # Correlation Matrix / Heatmap
    st.divider()
    st.subheader("Correlation Matrix")
    if len(numeric_columns) >= 2:
        corr = df[numeric_columns].corr()
        st.dataframe(corr)

        st.subheader("Correlation Heatmap")
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r",
            aspect="auto", title="Correlation Heatmap"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need at least 2 numeric columns for a correlation matrix.")

    # Pair Plot
    st.subheader("Pair Plot")
    if len(numeric_columns) >= 2:
        pair_columns = st.multiselect(
            "Select Columns (2-5 Columns)", numeric_columns, default=numeric_columns[:3]
        )
        if len(pair_columns) >= 2:
            fig = px.scatter_matrix(df, dimensions=pair_columns, title="Pair Plot")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least 2 numeric columns.")
    else:
        st.info("Need at least 2 numeric columns for a pair plot.")

    # Violin Plot
    st.subheader("Violin Plot")
    if numeric_columns:
        violin_column = st.selectbox("Select Numeric Column", numeric_columns, key="violin")
        if st.button("Generate Violin Plot"):
            fig = px.violin(df, y=violin_column, box=True, points="all", title=f"Violin Plot - {violin_column}")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric columns available.")

    # Distribution Plot
    st.subheader("Distribution Plot")
    if numeric_columns:
        dist_column = st.selectbox("Select Column", numeric_columns, key="distribution")
        if st.button("Generate Distribution Plot"):
            fig = px.histogram(df, x=dist_column, marginal="box", nbins=30, title=f"Distribution - {dist_column}")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric columns available.")

    # Count Plot
    st.subheader("Count Plot")
    count_column = st.selectbox("Select Column", all_columns, key="count_plot")
    if st.button("Generate Count Plot"):
        count_df = df[count_column].value_counts().reset_index()
        count_df.columns = [count_column, "Count"]
        fig = px.bar(count_df, x=count_column, y="Count", title=f"Count Plot - {count_column}")
        st.plotly_chart(fig, use_container_width=True)

    # Visualization Summary
    st.divider()
    st.header("Visualization Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    with summary_col1:
        st.metric("Rows", df.shape[0])
    with summary_col2:
        st.metric("Columns", df.shape[1])
    with summary_col3:
        st.metric("Numeric Columns", len(numeric_columns))

    st.subheader("Dataset Statistics")
    st.dataframe(df.describe())

    st.subheader("Missing Value Report")
    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values,
        "Percentage": (df.isnull().sum().values / len(df)) * 100
    })
    st.dataframe(missing_df)

    st.subheader("Column Data Types")
    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values
    })
    st.dataframe(dtype_df)

    st.divider()
    st.header("Visualization Report")
    st.success("Visualization Completed Successfully")
    st.write("Charts Available")

    charts = [
        "Histogram", "Bar Chart", "Line Chart", "Scatter Plot", "Box Plot",
        "Pie Chart", "Area Chart", "Bubble Chart", "Correlation Heatmap",
        "Pair Plot", "Violin Plot", "Distribution Plot", "Count Plot"
    ]
    st.dataframe(pd.DataFrame({"Visualization": charts}))

    st.divider()
    st.header("Dashboard Statistics")
    dc1, dc2, dc3, dc4 = st.columns(4)
    with dc1:
        st.metric("Visualizations", 13)
    with dc2:
        st.metric("Algorithms", 9)
    with dc3:
        st.metric("Modules", 7)
    with dc4:
        st.metric("Status", "Completed")

else:
    st.warning("Please upload a dataset first.")

# =====================================================
# PREDICTION
# =====================================================
st.divider()
st.header("Prediction")

st.write("Upload a trained model (.pkl) and a dataset for prediction.")

model_file = st.file_uploader("Upload Trained Model", type=["pkl"], key="prediction_model")
prediction_file = st.file_uploader(
    "Upload Prediction Dataset", type=["csv", "xlsx", "xls"], key="prediction_dataset"
)

prediction_input_df = None
if prediction_file is not None:
    try:
        if prediction_file.name.endswith(".csv"):
            prediction_input_df = pd.read_csv(prediction_file)
        else:
            prediction_input_df = pd.read_excel(prediction_file)

        st.success("Prediction Dataset Loaded Successfully")
        st.dataframe(prediction_input_df.head())

    except Exception as e:
        st.error(f"Could not read prediction file: {e}")

pred_model = None
if model_file is not None:
    try:
        pred_model = joblib.load(model_file)
        st.success("Model Loaded Successfully")
        st.write("Model Type")
        st.code(type(pred_model).__name__)

    except Exception as e:
        st.error(f"Could not load model: {e}")

if prediction_input_df is not None:
    st.subheader("Prediction Dataset Information")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Rows", prediction_input_df.shape[0])
    with c2:
        st.metric("Columns", prediction_input_df.shape[1])
    with c3:
        st.metric("Missing Values", int(prediction_input_df.isnull().sum().sum()))

    st.dataframe(prediction_input_df.dtypes.astype(str))

# ------------------------------
# Run Prediction
# ------------------------------
if pred_model is not None and prediction_input_df is not None:

    st.divider()
    st.subheader("Run Prediction")

    training_features = st.session_state.get("features")
    feature_encoders = st.session_state.get("feature_encoders", {})
    target_encoder = st.session_state.get("target_encoder")

    if st.button("Predict"):
        try:
            input_df = prediction_input_df.copy()

            # Align columns to the ones the model was trained on, if we know them
            if training_features:
                missing_cols = [c for c in training_features if c not in input_df.columns]
                if missing_cols:
                    st.error(f"Prediction dataset is missing required columns: {missing_cols}")
                    st.stop()
                input_df = input_df[training_features]

            # Encode categoricals the same way training data was encoded
            for col in input_df.select_dtypes(include="object").columns:
                if col in feature_encoders:
                    le = feature_encoders[col]
                    known = set(le.classes_)
                    input_df[col] = input_df[col].astype(str).apply(
                        lambda v: le.transform([v])[0] if v in known else -1
                    )
                else:
                    fallback_encoder = LabelEncoder()
                    input_df[col] = fallback_encoder.fit_transform(input_df[col].astype(str))

            prediction = pred_model.predict(input_df)

            if target_encoder is not None:
                try:
                    prediction = target_encoder.inverse_transform(prediction.astype(int))
                except Exception:
                    pass  # keep numeric prediction if inverse mapping fails

            prediction_result = prediction_input_df.copy()
            prediction_result["Prediction"] = prediction

            st.session_state["prediction_result"] = prediction_result
            st.session_state["batch_prediction"] = prediction

            st.success("Prediction Completed Successfully")
            st.subheader("Prediction Result")
            st.dataframe(prediction_result)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # ------------------------------
    # Prediction Probability
    # ------------------------------
    if hasattr(pred_model, "predict_proba"):
        if st.button("Prediction Probability"):
            try:
                input_df = prediction_input_df.copy()

                if training_features:
                    missing_cols = [c for c in training_features if c not in input_df.columns]
                    if missing_cols:
                        st.error(f"Prediction dataset is missing required columns: {missing_cols}")
                        st.stop()
                    input_df = input_df[training_features]

                for col in input_df.select_dtypes(include="object").columns:
                    if col in feature_encoders:
                        le = feature_encoders[col]
                        known = set(le.classes_)
                        input_df[col] = input_df[col].astype(str).apply(
                            lambda v: le.transform([v])[0] if v in known else -1
                        )
                    else:
                        fallback_encoder = LabelEncoder()
                        input_df[col] = fallback_encoder.fit_transform(input_df[col].astype(str))

                probability = pred_model.predict_proba(input_df)
                probability_df = pd.DataFrame(probability, columns=pred_model.classes_)

                st.subheader("Prediction Probability")
                st.dataframe(probability_df)

            except Exception as e:
                st.error(f"Could not compute probabilities: {e}")
    else:
        st.info("Selected model does not support probability prediction.")

# ------------------------------
# Prediction Statistics
# ------------------------------
if "prediction_result" in st.session_state:

    st.divider()
    st.subheader("Prediction Statistics")

    prediction_count = (
        st.session_state["prediction_result"]["Prediction"]
        .value_counts()
        .reset_index()
    )
    prediction_count.columns = ["Prediction", "Count"]

    st.dataframe(prediction_count)
    st.bar_chart(prediction_count.set_index("Prediction"))

    st.subheader("Prediction Summary")
    st.metric("Total Predictions", len(st.session_state["prediction_result"]))
    st.metric("Unique Classes", len(prediction_count))

# ------------------------------
# Download Prediction
# ------------------------------
st.divider()
st.header("Download Prediction Result")

if "prediction_result" in st.session_state:
    csv = st.session_state["prediction_result"].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Prediction CSV",
        data=csv,
        file_name="prediction_result.csv",
        mime="text/csv"
    )
else:
    st.info("Prediction result not available.")

# ------------------------------
# Prediction Report
# ------------------------------
if "prediction_result" in st.session_state:

    st.divider()
    st.header("Prediction Report")

    result = st.session_state["prediction_result"]

    report_col1, report_col2, report_col3 = st.columns(3)
    with report_col1:
        st.metric("Rows Predicted", len(result))
    with report_col2:
        st.metric("Columns", result.shape[1])
    with report_col3:
        st.metric("Prediction Classes", result["Prediction"].nunique())

    st.subheader("Prediction Distribution")
    distribution = result["Prediction"].value_counts()
    st.bar_chart(distribution)

    st.subheader("Export Final Dataset")
    export_csv = result.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Export Final Dataset",
        export_csv,
        "Hybrid_ML_Output.csv",
        "text/csv"
    )

# =====================================================
# APPLICATION SUMMARY
# =====================================================
st.divider()
st.header("Application Summary")

summary = {
    "Module": [
        "Upload Dataset", "Data Preprocessing", "Feature Selection",
        "Supervised Learning", "Unsupervised Learning", "Visualization", "Prediction"
    ],
    "Status": ["Completed"] * 7
}
summary_df = pd.DataFrame(summary)
st.dataframe(summary_df)

# =====================================================
# DASHBOARD STATISTICS
# =====================================================
st.divider()
st.header("Project Statistics")

pc1, pc2, pc3, pc4 = st.columns(4)
with pc1:
    st.metric("Modules", "7")
with pc2:
    st.metric("Algorithms", "9")
with pc3:
    st.metric("Charts", "13")
with pc4:
    st.metric("Status", "Completed")

# =====================================================
# FOOTER
# =====================================================
st.divider()
st.markdown("""
### Hybrid Machine Learning Platform

Version : 1.0

Developed Using

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- Plotly
- Matplotlib

Developed by : Y Nandhakumar
Artificial Intelligence & Data Science
""")