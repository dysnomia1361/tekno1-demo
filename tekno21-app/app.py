from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from matplotlib import colormaps
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torchvision import transforms

from model import AttentionRollout, GradCAM, HybridCNNViT

# ============ CONFIGURACIÓN (debe coincidir con el entrenamiento) ============
BASE = Path(__file__).parent
SAMPLES = BASE / "samples"
WEIGHTS = BASE / "best_hybrid.pt"

MODEL_ARCH = dict(embed_dim=256, depth=4, heads=8, mlp_ratio=4, dropout=0.1)
IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TEST_METRICS = None   # opcional, p. ej. {"f1_macro": 0.93}: cifras del test completo del notebook
# =============================================================================

st.set_page_config(page_title="Clasificador de ACV en CT", page_icon="🧠", layout="wide")

# ---------- Verificaciones iniciales ----------
if not WEIGHTS.exists():
    st.error(f"No se encontró {WEIGHTS.name} junto a app.py.")
    st.stop()
if not (SAMPLES / "manifest.csv").exists():
    st.error("No se encontró samples/manifest.csv.")
    st.stop()


# ---------- Carga de datos y modelo ----------
@st.cache_data
def load_manifest():
    m = pd.read_csv(SAMPLES / "manifest.csv")
    names = m.drop_duplicates("label").sort_values("label")["label_name"].tolist()
    return m, names


manifest, class_names = load_manifest()


@st.cache_resource
def load_model():
    model = HybridCNNViT(num_classes=len(class_names), **MODEL_ARCH, pretrained=False)
    state = torch.load(WEIGHTS, map_location="cpu")
    state = state.get("state_dict", state)          # acepta state_dict directo o checkpoint completo
    model.load_state_dict(state)
    model.eval()
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return model, tfm, AttentionRollout(model, IMG_SIZE), GradCAM(model, model.cnn[-1], IMG_SIZE)


model, tfm, rollout, gradcam = load_model()


# ---------- Utilidades ----------
def overlay(img: Image.Image, cam: np.ndarray, alpha=0.45) -> Image.Image:
    base = np.asarray(img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))) / 255.0
    heat = colormaps["jet"](cam)[..., :3]
    return Image.fromarray((((1 - alpha) * base + alpha * heat) * 255).astype(np.uint8))


def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    cmn = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm[i, j]}\n({cmn[i, j]:.0%})", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner="Calculando predicciones sobre las muestras...")
def predict_samples():
    chunks = []
    for s in range(0, len(manifest), 16):
        files = manifest["file"].iloc[s:s + 16]
        x = torch.stack([tfm(Image.open(SAMPLES / f).convert("RGB")) for f in files])
        with torch.no_grad():
            chunks.append(torch.softmax(model(x), dim=1).numpy())
    probs = np.concatenate(chunks)

    df = pd.DataFrame({
        "Archivo": manifest["file"],
        "Real": manifest["label_name"],
        "Predicción": [class_names[i] for i in probs.argmax(1)],
        "Confianza": probs.max(1),
    })
    df["Resultado"] = np.where(df["Real"] == df["Predicción"], "✅ Correcto", "❌ Error")
    for i, n in enumerate(class_names):
        df[f"P({n})"] = probs[:, i]

    agreement = None
    if "pred_notebook" in manifest.columns:
        agreement = float((df["Predicción"] == manifest["pred_notebook"]).mean())
    return df, agreement


def show_maps(img: Image.Image):
    x = tfm(img).unsqueeze(0)
    cam_r, _ = rollout(x)
    cam_g, _ = gradcam(x)
    c1, c2, c3 = st.columns(3)
    c1.image(img, caption="Original", use_container_width=True)
    c2.image(overlay(img, cam_r[0]), caption="Attention rollout (Transformer)", use_container_width=True)
    c3.image(overlay(img, cam_g[0]), caption="Grad-CAM (CNN)", use_container_width=True)


# ---------- Encabezado ----------
st.title("🧠 Clasificación de accidente cerebrovascular en CT")
st.caption("Prototipo de investigación: modelo híbrido CNN (ResNet18) + Transformer entrenado con Tekno21.")
st.warning("Solo demostrativo. **No es un dispositivo médico** y no debe usarse para diagnóstico.")

tab1, tab2 = st.tabs(["📊 Predicciones vs. labels reales", "🔍 Probar una imagen"])

# ---------- Pestaña 1: tabla comparativa ----------
with tab1:
    df, agreement = predict_samples()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Muestras", len(df))
    m2.metric("Accuracy", f"{accuracy_score(df['Real'], df['Predicción']):.1%}")
    m3.metric("F1 macro", f"{f1_score(df['Real'], df['Predicción'], average='macro', labels=class_names, zero_division=0):.3f}")
    m4.metric("Errores", int((df['Resultado'] == '❌ Error').sum()))

    caption = ("Las muestras son un subconjunto estratificado del test set, así que estas cifras "
               "no coinciden necesariamente con las del test completo")
    if TEST_METRICS:
        caption += f" (F1 macro completo: {TEST_METRICS['f1_macro']:.3f})"
    st.caption(caption + ".")
    if agreement is not None:
        st.caption(f"Verificación de despliegue: la app coincide con el notebook en {agreement:.0%} de las muestras.")

    f1c, f2c = st.columns([2, 1])
    sel_classes = f1c.multiselect("Filtrar por clase real", class_names, default=class_names)
    outcome = f2c.radio("Mostrar", ["Todos", "Solo correctos", "Solo errores"], horizontal=True)

    view = df[df["Real"].isin(sel_classes)]
    if outcome == "Solo correctos":
        view = view[view["Resultado"].str.startswith("✅")]
    elif outcome == "Solo errores":
        view = view[view["Resultado"].str.startswith("❌")]
    view = view.reset_index(drop=True)

    prob_cols = [f"P({n})" for n in class_names]
    styled = (view.style
              .apply(lambda r: ["background-color: rgba(220,53,69,0.18)" if r["Resultado"].startswith("❌") else ""
                                for _ in r], axis=1)
              .format({"Confianza": "{:.1%}", **{c: "{:.1%}" for c in prob_cols}}))

    st.markdown(f"**{len(view)} imágenes.** Selecciona una fila para ver la imagen y sus mapas de interpretabilidad.")
    event = st.dataframe(styled, use_container_width=True, hide_index=True,
                         on_select="rerun", selection_mode="single-row")

    st.download_button("⬇️ Descargar tabla (CSV)", view.to_csv(index=False).encode("utf-8"),
                       "comparacion_predicciones.csv", "text/csv")

    if event.selection.rows:
        r = view.iloc[event.selection.rows[0]]
        st.subheader(f"Detalle: {r['Archivo']}")
        st.markdown(f"**Real:** {r['Real']} &nbsp;|&nbsp; **Predicción:** {r['Predicción']} "
                    f"({r['Confianza']:.1%}) &nbsp;|&nbsp; {r['Resultado']}")
        show_maps(Image.open(SAMPLES / r["Archivo"]).convert("RGB"))

    st.divider()
    cc1, cc2 = st.columns(2)
    with cc1:
        st.subheader("Matriz de confusión (todas las muestras)")
        st.pyplot(plot_confusion(df["Real"], df["Predicción"]))
    with cc2:
        st.subheader("Métricas por clase")
        rep = classification_report(df["Real"], df["Predicción"], labels=class_names,
                                    output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(3), use_container_width=True)

# ---------- Pestaña 2: imagen propia ----------
with tab2:
    uploaded = st.file_uploader("Sube una imagen de CT cerebral", type=["png", "jpg", "jpeg"])
    true_opt = st.selectbox("Label real (opcional, para comparar)", ["— desconocido —"] + class_names)

    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        with torch.no_grad():
            probs = torch.softmax(model(tfm(img).unsqueeze(0)), dim=1)[0].numpy()
        pred = int(probs.argmax())

        c1, c2 = st.columns(2)
        c1.metric("Clase predicha", class_names[pred], f"{probs[pred]:.1%} de confianza")
        if true_opt != "— desconocido —":
            c2.metric("Label real", true_opt, "✅ coincide" if true_opt == class_names[pred] else "❌ no coincide")
        if probs[pred] < 0.60:
            st.error("Confianza baja: resultado poco fiable; la imagen podría estar fuera de la distribución de entrenamiento.")
        st.bar_chart(pd.DataFrame({"Probabilidad": probs}, index=class_names))
        show_maps(img)
    else:
        st.info("Sube una imagen para ver la predicción y los mapas de interpretabilidad.")