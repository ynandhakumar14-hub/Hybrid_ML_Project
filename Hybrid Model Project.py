import streamlit as st

st.set_page_config(
    page_title="Hybrid Machine Learning System",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Hybrid Machine Learning System")

st.markdown("""
### 👋 Welcome
Welcome to the **Hybrid Machine Learning Platform**.
This application provides a complete Machine Learning workflow from dataset upload to prediction.
""")

st.divider()

# Dashboard Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📂 Modules", "7")

with col2:
    st.metric("🤖 Algorithms", "9")

with col3:
    st.metric("📊 Charts", "12")

with col4:
    st.metric("📁 File Types", "CSV / Excel")

st.divider()

st.header("📌 Project Workflow")

st.markdown("""
1. 📂 **Upload Dataset** — load a CSV or Excel file
2. 🧹 **Data Preprocessing** — handle missing values, duplicates, encoding, and scaling
3. 🎯 **Feature Selection** — choose target/features and rank importance
4. 🤖 **Supervised Learning** — train and evaluate classification models
5. 🧠 **Unsupervised Learning** — cluster data with KMeans, Hierarchical, or DBSCAN
6. 📊 **Visualization** — explore the data with interactive charts
7. 🔮 **Prediction** — apply a trained model to new data
""")

st.divider()

st.header("📚 Project Modules")

col1, col2 = st.columns(2)

with col1:
    st.success("📂 Upload Dataset")
    st.success("🧹 Data Preprocessing")
    st.success("🎯 Feature Selection")
    st.success("🤖 Supervised Learning")

with col2:
    st.success("🧠 Unsupervised Learning")
    st.success("📊 Visualization")
    st.success("🔮 Prediction")

st.divider()

st.header("🤖 Supervised Algorithms")

st.write("""
- Logistic Regression
- Decision Tree
- Random Forest
- KNN
- Support Vector Machine (SVM)
- Naive Bayes
""")

st.header("🧠 Unsupervised Algorithms")

st.write("""
- K-Means Clustering
- Hierarchical Clustering
- DBSCAN
""")

st.divider()

st.header("🛠 Technologies Used")

tech1, tech2, tech3 = st.columns(3)

with tech1:
    st.info("🐍 Python")
    st.info("📊 Pandas")
    st.info("🔢 NumPy")

with tech2:
    st.info("🤖 Scikit-Learn")
    st.info("📈 Matplotlib")
    st.info("📉 Plotly")

with tech3:
    st.info("🌐 Streamlit")
    st.info("📁 CSV")
    st.info("📗 Excel")

st.divider()

st.header("✨ Key Features")

st.write("""
✅ Upload CSV & Excel Dataset
✅ Automatic Data Preprocessing
✅ Feature Selection
✅ Supervised Learning Models
✅ Unsupervised Learning Models
✅ Interactive Visualization Dashboard
✅ Machine Learning Prediction
✅ Download Results
""")

st.divider()

st.header("👨‍💻 Developer")

st.success("""
Name : Nandhakumar
Department : Artificial Intelligence & Data Science
Project : Hybrid Machine Learning Platform
Version : 1.0
""")