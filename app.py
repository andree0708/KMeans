"""Aplicación Streamlit — segmentación de clientes con K-Means."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Mall_Customers.csv"
MODELS_DIR = ROOT / "models"
FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]

PERFILES = {
    "Clientes de alto valor": "Fidelización VIP, lanzamientos y experiencias premium.",
    "Clientes con potencial": "Captación: eventos, marcas premium y first-visit offers.",
    "Clientes promedio": "Cross-selling y comunicación regular.",
    "Clientes de bajo valor": "Campañas de bajo costo y ofertas de entrada.",
    "Clientes sensibles / alto gasto": "Cupones, 2x1 y financiación; categorías de margen.",
}


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def train_artifacts(df: pd.DataFrame):
    X = df[FEATURES]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertias = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    model = KMeans(n_clusters=5, init="k-means++", n_init=10, random_state=42)
    labels = model.fit_predict(X_scaled)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "kmeans_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(FEATURES, MODELS_DIR / "features.joblib")
    return scaler, model, X_scaled, labels, inertias


def etiquetar_centroides(df: pd.DataFrame, centroids_df: pd.DataFrame) -> dict:
    inc_med = df["Annual Income (k$)"].median()
    score_med = df["Spending Score (1-100)"].median()

    def etiqueta(row):
        alto_ing = row["Annual Income (k$)"] >= inc_med
        alto_gasto = row["Spending Score (1-100)"] >= score_med
        if alto_ing and alto_gasto:
            return "Clientes de alto valor"
        if alto_ing and not alto_gasto:
            return "Clientes con potencial"
        if (not alto_ing) and alto_gasto:
            return "Clientes sensibles / alto gasto"
        return "Clientes de bajo valor"

    perfiles = centroids_df.apply(etiqueta, axis=1)
    dist = (
        ((centroids_df["Annual Income (k$)"] - inc_med) / df["Annual Income (k$)"].std()) ** 2
        + (
            (centroids_df["Spending Score (1-100)"] - score_med)
            / df["Spending Score (1-100)"].std()
        )
        ** 2
    ) ** 0.5
    perfiles.iloc[int(dist.idxmin())] = "Clientes promedio"
    return perfiles.to_dict()


def main():
    st.set_page_config(page_title="Segmentación K-Means", layout="wide")
    st.title("Segmentación de clientes — Mall Customers")
    st.caption("Taller final de minería de datos · K-Means")

    df = load_data()
    scaler, model, _X_scaled, labels, inertias = train_artifacts(df)
    df = df.copy()
    df["Cluster"] = labels

    centroids = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=FEATURES)
    mapa = etiquetar_centroides(df, centroids)
    df["Perfil"] = df["Cluster"].map(mapa)
    centroids["Perfil"] = centroids.index.map(mapa)

    tab_codo, tab_clusters, tab_nuevo = st.tabs(
        ["Método del codo", "Clusters", "Clasificar cliente nuevo"]
    )

    with tab_codo:
        st.subheader("Inercia según el número de grupos")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(range(1, 11), inertias, marker="o")
        ax.axvline(5, color="red", linestyle="--", label="K = 5")
        ax.set_xlabel("K")
        ax.set_ylabel("Inercia (WCSS)")
        ax.set_title("Método del codo")
        ax.legend()
        st.pyplot(fig)
        st.write(
            "El codo se observa en **K = 5**: a partir de ahí la inercia baja poco "
            "y no justifica más grupos."
        )

    with tab_clusters:
        st.subheader("Grupos obtenidos (K = 5)")
        fig, ax = plt.subplots(figsize=(7, 5))
        for perfil, grupo in df.groupby("Perfil"):
            ax.scatter(
                grupo["Annual Income (k$)"],
                grupo["Spending Score (1-100)"],
                label=perfil,
                s=50,
            )
        ax.scatter(
            centroids["Annual Income (k$)"],
            centroids["Spending Score (1-100)"],
            c="black",
            marker="X",
            s=180,
            label="Centroides",
        )
        ax.set_xlabel("Ingreso anual (k$)")
        ax.set_ylabel("Puntaje de gasto (1-100)")
        ax.legend(fontsize=8)
        st.pyplot(fig)

        resumen = (
            df.groupby("Perfil")[FEATURES + ["Age"]]
            .mean()
            .round(1)
            .assign(Clientes=df.groupby("Perfil").size())
        )
        st.dataframe(resumen, use_container_width=True)

    with tab_nuevo:
        st.subheader("Asignar un cliente al cluster más cercano")
        col1, col2 = st.columns(2)
        with col1:
            ingreso = st.number_input("Ingreso anual (k$)", min_value=0.0, max_value=200.0, value=60.0)
        with col2:
            gasto = st.slider("Puntaje de gasto (1-100)", min_value=1, max_value=100, value=50)

        if st.button("Clasificar", type="primary"):
            x_new = scaler.transform(pd.DataFrame([[ingreso, gasto]], columns=FEATURES))
            cluster = int(model.predict(x_new)[0])
            perfil = mapa[cluster]
            st.success(f"Cluster **{cluster}** — {perfil}")
            st.info(PERFILES.get(perfil, ""))

            fig, ax = plt.subplots(figsize=(7, 5))
            for p, grupo in df.groupby("Perfil"):
                ax.scatter(
                    grupo["Annual Income (k$)"],
                    grupo["Spending Score (1-100)"],
                    label=p,
                    s=40,
                    alpha=0.7,
                )
            ax.scatter([ingreso], [gasto], c="red", s=160, marker="*", label="Cliente nuevo")
            ax.set_xlabel("Ingreso anual (k$)")
            ax.set_ylabel("Puntaje de gasto (1-100)")
            ax.legend(fontsize=8)
            st.pyplot(fig)


if __name__ == "__main__":
    main()
