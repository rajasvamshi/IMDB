import os
import time
import joblib
import pandas as pd
import streamlit as st

# 🔐 Optional: load API key from Streamlit secrets (for cloud deploy)
try:
    key_from_secrets = None
    try:
        key_from_secrets = st.secrets["OPENAI_API_KEY"]
    except Exception:
        key_from_secrets = None

    if key_from_secrets and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = key_from_secrets
except Exception:
    # ignore any secrets-related errors in local dev
    pass

# ==============================
# Model loading
# ==============================

BASE_TFIDF = None
BASE_LR = None
if os.path.exists("models/tfidf_vec.pkl") and os.path.exists("models/tfidf_lr.pkl"):
    BASE_TFIDF = joblib.load("models/tfidf_vec.pkl")
    BASE_LR = joblib.load("models/tfidf_lr.pkl")

HAS_NB = False
NB_VEC = None
NB_MODEL = None
if os.path.exists("models/tfidf_vec_for_nb.pkl") and os.path.exists("models/nb_proxy.pkl"):
    NB_VEC = joblib.load("models/tfidf_vec_for_nb.pkl")
    NB_MODEL = joblib.load("models/nb_proxy.pkl")
    HAS_NB = True

LLM_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL = "gpt-4o-mini"


# ==============================
# LLM helper functions
# ==============================

def _extract_text_from_response(resp) -> str:
    """Safely extract text from Responses API result."""
    try:
        out = resp.output[0].content[0].text
        if hasattr(out, "value"):
            return out.value
        return str(out)
    except Exception:
        try:
            return str(resp.output[0].content[0])
        except Exception:
            return str(resp)


def call_llm_label(text: str, method: str = "zero_shot"):
    """
    Ask the LLM for a POSITIVE/NEGATIVE label.
    Returns (label_int_or_None, info_string).
    """
    if not LLM_AVAILABLE:
        return None, "LLM disabled (no OPENAI_API_KEY)"

    try:
        from openai import OpenAI
        client = OpenAI()

        if method == "zero_shot":
            prompt = f"""Classify the sentiment of this movie review as POSITIVE or NEGATIVE.
IMPORTANT: Reply with exactly one word ONLY: POSITIVE or NEGATIVE.

Review:
\"\"\"{text}\"\"\"

Answer:
"""
        elif method == "few_shot":
            prompt = (
                "You classify movie review sentiment as POSITIVE or NEGATIVE.\n\n"
                "Example 1:\n"
                "Review: \"Amazing acting, great direction and strong performances.\"\n"
                "Label: POSITIVE\n\n"
                "Example 2:\n"
                "Review: \"Terrible pacing, very boring and a complete waste of time.\"\n"
                "Label: NEGATIVE\n\n"
                f"Now classify this review as POSITIVE or NEGATIVE:\n\"{text}\"\n\n"
                "Reply with exactly one word: POSITIVE or NEGATIVE."
            )
        elif method == "chain_of_thought":
            prompt = f"""You are an expert movie sentiment analyst.

1. Briefly explain (step by step) whether the overall sentiment of this review is POSITIVE or NEGATIVE.
2. On the VERY LAST LINE, output exactly one of:
   FINAL_LABEL: POSITIVE
   FINAL_LABEL: NEGATIVE

Rules:
- Do NOT mention both POSITIVE and NEGATIVE in the FINAL_LABEL line.
- The last line MUST start with FINAL_LABEL:.

Review:
\"\"\"{text}\"\"\"

Now reason step-by-step, then end with the FINAL_LABEL line.
"""
        else:
            prompt = f"""Classify the sentiment of this movie review as POSITIVE or NEGATIVE.
IMPORTANT: Reply with exactly one word ONLY: POSITIVE or NEGATIVE.

Review:
\"\"\"{text}\"\"\"

Answer:
"""

        t0 = time.time()
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            max_output_tokens=128 if method == "chain_of_thought" else 32,
            temperature=0.0,
        )
        latency = time.time() - t0
        out = _extract_text_from_response(resp)
        upper = out.upper()

        label_int = None

        if method == "chain_of_thought":
            lines = [ln.strip().upper() for ln in upper.splitlines() if ln.strip()]
            final_line = lines[-1] if lines else ""
            if "FINAL_LABEL" in final_line:
                if "POSITIVE" in final_line and "NEGATIVE" not in final_line:
                    label_int = 1
                elif "NEGATIVE" in final_line and "POSITIVE" not in final_line:
                    label_int = 0

            if label_int is None:
                for ln in reversed(lines):
                    if "POSITIVE" in ln and "NEGATIVE" not in ln:
                        label_int = 1
                        break
                    if "NEGATIVE" in ln and "POSITIVE" not in ln:
                        label_int = 0
                        break
        else:
            if "POSITIVE" in upper and "NEGATIVE" not in upper:
                label_int = 1
            elif "NEGATIVE" in upper and "POSITIVE" not in upper:
                label_int = 0

        if label_int == 1:
            return 1, f"POSITIVE ({latency:.2f}s, {method})"
        if label_int == 0:
            return 0, f"NEGATIVE ({latency:.2f}s, {method})"

        short = upper.replace("\n", " ")[:80]
        return None, f"Ambiguous: {short} ({latency:.2f}s)"

    except Exception as e:
        return None, f"LLM error: {e}"


def call_llm_summary(text: str, sentiment_hint: str | None = None) -> str:
    """Use LLM to generate a short explanation/summary of the sentiment."""
    if not LLM_AVAILABLE:
        return "LLM disabled (no OPENAI_API_KEY)."

    try:
        from openai import OpenAI
        client = OpenAI()

        sentiment_line = (
            f"The current sentiment prediction is: {sentiment_hint}.\n"
            if sentiment_hint
            else ""
        )

        prompt = f"""You are an AI assistant that explains movie review sentiment clearly.

{sentiment_line}Summarize the overall sentiment of the following movie review in 2–3 sentences.
Mention key phrases that make it positive or negative, but do NOT repeat the full text.

Review:
\"\"\"{text}\"\"\""""

        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            max_output_tokens=128,
            temperature=0.3,
        )
        return _extract_text_from_response(resp)
    except Exception as e:
        return f"LLM summary error: {e}"


def call_llm_chat(user_message: str, history: list[dict]) -> str:
    """Chat-style LLM reply using previous turns as context."""
    if not LLM_AVAILABLE:
        return "LLM disabled (no OPENAI_API_KEY)."

    try:
        from openai import OpenAI
        client = OpenAI()

        convo_lines = []
        for turn in history:
            convo_lines.append(f"User: {turn['user']}")
            convo_lines.append(f"Assistant: {turn['bot']}")

        convo_text = "\n".join(convo_lines[-10:])
        prompt = f"""You are a helpful AI assistant specialized in movies and sentiment analysis.
You explain things clearly and briefly.

Conversation so far:
{convo_text}

User now says:
{user_message}

Reply in 2–4 sentences.
"""

        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            max_output_tokens=128,
            temperature=0.5,
        )
        return _extract_text_from_response(resp)
    except Exception as e:
        return f"LLM chat error: {e}"


# ==============================
# Streamlit setup + state
# ==============================

st.set_page_config(
    page_title="Sentiment Comparison Bot",
    page_icon="🎬",
    layout="centered",
)

if "history" not in st.session_state:
    st.session_state.history = []
if "results" not in st.session_state:
    st.session_state.results = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pending_load_review" in st.session_state:
    st.session_state.input_text = st.session_state.pending_load_review
    del st.session_state.pending_load_review

st.title("🎬 Sentiment Analysis — Baseline vs Deep Proxy vs LLM")
st.caption("Compare classical models and LLM prompting with explanations, charts, and chatbot.")

tab_single, tab_chat = st.tabs(["🔍 Single Review", "💬 Chatbot"])


# ==============================
# TAB 1: Single Review
# ==============================

with tab_single:
    st.subheader("Single Review Analysis")

    method = st.selectbox(
        "LLM prompting method:",
        ["zero_shot", "few_shot", "chain_of_thought"],
    )

    selected_models = st.multiselect(
        "Select models to run:",
        ["Baseline", "Deep (NB)", "LLM"],
        default=["Baseline", "Deep (NB)", "LLM"],
    )

    want_summary = st.checkbox("Generate LLM explanation / summary", value=False)

    text = st.text_area(
        "Paste an IMDb-style movie review:",
        height=200,
        key="input_text",
    )

    analyze_btn = st.button("Analyze Review")

    if analyze_btn:
        if not text.strip():
            st.warning("Please enter a review.")
        else:
            word_count = len(text.split())
            if word_count < 5:
                st.warning(
                    "⚠️ Very short input — classical models may be unreliable; "
                    "LLM may handle context better."
                )

            base_pred = base_prob = None
            nb_pred = nb_prob = None
            llm_int = None
            llm_pred_label = "N/A"
            llm_info = "Not run"

            # Baseline
            if "Baseline" in selected_models and BASE_TFIDF is not None and BASE_LR is not None:
                X = BASE_TFIDF.transform([text])
                base_prob = float(BASE_LR.predict_proba(X)[0, 1])
                base_pred = "POSITIVE" if base_prob >= 0.5 else "NEGATIVE"
            elif "Baseline" in selected_models:
                st.info("Baseline model not found. Run baseline training first.")

            # Deep (NB)
            if "Deep (NB)" in selected_models:
                if HAS_NB and NB_VEC is not None and NB_MODEL is not None:
                    X_nb = NB_VEC.transform([text])
                    if hasattr(NB_MODEL, "predict_proba"):
                        nb_prob = float(NB_MODEL.predict_proba(X_nb)[0, 1])
                        nb_pred = "POSITIVE" if nb_prob >= 0.5 else "NEGATIVE"
                    else:
                        raw = NB_MODEL.predict(X_nb)[0]
                        nb_pred = "POSITIVE" if raw == 1 else "NEGATIVE"
                        nb_prob = None
                else:
                    st.info("NB proxy not trained. Run: python src/train_nb_proxy.py")

            # LLM
            if "LLM" in selected_models:
                llm_int, llm_info = call_llm_label(text, method=method)
                llm_pred_label = {1: "POSITIVE", 0: "NEGATIVE"}.get(llm_int, "N/A")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.subheader("Baseline (TF-IDF + LR)")
                if base_pred is not None:
                    st.metric("Prediction", base_pred, f"p(pos)={base_prob:.3f}")
                    if 0.45 <= base_prob <= 0.55:
                        st.warning("Borderline / low confidence", icon="⚠️")
                elif "Baseline" in selected_models:
                    st.info("Baseline not available.")

            with c2:
                st.subheader("Deep Model (NB Proxy)")
                if "Deep (NB)" in selected_models:
                    if nb_pred is not None:
                        if nb_prob is not None:
                            st.metric("Prediction", nb_pred, f"p(pos)={nb_prob:.3f}")
                            if 0.45 <= nb_prob <= 0.55:
                                st.warning("Borderline / low confidence", icon="⚠️")
                        else:
                            st.metric("Prediction", nb_pred, "prob=N/A")
                    else:
                        st.info("NB proxy not available.")
                else:
                    st.info("Deep model not selected.")

            with c3:
                st.subheader("LLM (GPT)")
                if "LLM" in selected_models:
                    st.metric("Prediction", llm_pred_label, llm_info)
                else:
                    st.info("LLM not selected.")

            # Confidence chart
            conf_rows = []
            if base_prob is not None:
                conf_rows.append({"Model": "Baseline", "p_pos": base_prob})
            if nb_prob is not None:
                conf_rows.append({"Model": "Deep (NB)", "p_pos": nb_prob})

            if conf_rows:
                st.markdown("### 📊 Sentiment confidence (p(pos))")
                df_conf = pd.DataFrame(conf_rows).set_index("Model")
                st.bar_chart(df_conf)

            # LLM explanation
            if want_summary and "LLM" in selected_models:
                st.markdown("### 📝 LLM explanation / summary")
                hint = llm_pred_label if llm_pred_label != "N/A" else None
                summary_text = call_llm_summary(text, sentiment_hint=hint)
                st.write(summary_text)

            # History
            row = {
                "review": text,
                "baseline_label": base_pred,
                "baseline_p_pos": base_prob,
                "deep_label": nb_pred,
                "deep_p_pos": nb_prob,
                "llm_label": llm_pred_label,
                "llm_info": llm_info,
                "llm_method": method,
                "word_count": word_count,
            }
            st.session_state.results.append(row)
            st.session_state.history = (st.session_state.history + [row])[-5:]

    st.divider()

    st.subheader("🕒 History (last 5 analyses)")
    if st.session_state.history:
        labels = []
        for i, r in enumerate(st.session_state.history):
            snippet = r["review"][:60] + "…" if len(r["review"]) > 60 else r["review"]
            labels.append(
                f"{i+1}. {snippet} | B:{r['baseline_label']} "
                f"D:{r['deep_label']} L:{r['llm_label']}"
            )

        index = st.selectbox(
            "Select past run:",
            range(len(labels)),
            format_func=lambda i: labels[i],
        )
        sel = st.session_state.history[index]

        st.write("**Review:**")
        st.write(sel["review"])

        st.write("**Predictions:**")
        st.write(f"- Baseline: {sel['baseline_label']} (p(pos)={sel['baseline_p_pos']})")
        st.write(f"- Deep (NB): {sel['deep_label']} (p(pos)={sel['deep_p_pos']})")
        st.write(f"- LLM: {sel['llm_label']} ({sel['llm_info']})")

        if st.button("Load this review into input"):
            st.session_state.pending_load_review = sel["review"]
            st.rerun()
    else:
        st.caption("No history yet.")

    st.divider()

    st.subheader("⬇️ Download all results")
    if st.session_state.results:
        df_all = pd.DataFrame(st.session_state.results)
        st.download_button(
            "Download CSV",
            df_all.to_csv(index=False),
            "sentiment_results.csv",
            mime="text/csv",
        )
    else:
        st.caption("No analyses yet.")

    st.divider()

    # ==============================
    # Model metrics comparison section
    # ==============================

    st.subheader("📈 Model Comparison (Accuracy, Precision, Recall, F1-score)")

    metrics = {}

    def load_metric_csv(path, model_name):
        if os.path.exists(path):
            df = pd.read_csv(path)
            metrics[model_name] = df
        else:
            metrics[model_name] = None

    load_metric_csv("results/baseline_eval.csv", "Baseline (TF-IDF + LR)")
    load_metric_csv("results/deep_eval.csv", "Deep Model (NB Proxy)")
    load_metric_csv("results/llm_eval_summary.csv", "LLM (GPT)")

    rows = []
    for model_name, df_m in metrics.items():
        if df_m is not None:
            row = {"Model": model_name}
            for _, r in df_m.iterrows():
                m = r["metric"]
                v = r["value"]
                if m in ["accuracy", "precision", "recall", "f1"]:
                    row[m] = v
            rows.append(row)

    if rows:
        df_table = pd.DataFrame(rows).set_index("Model")
        st.dataframe(df_table, use_container_width=True)

        st.markdown("#### 📊 Accuracy Comparison")
        if "accuracy" in df_table.columns:
            st.bar_chart(df_table[["accuracy"]])

        if all(c in df_table.columns for c in ["precision", "recall", "f1"]):
            st.markdown("#### 🔍 Precision / Recall / F1 Comparison")
            st.bar_chart(df_table[["precision", "recall", "f1"]])
        st.caption(
            "Metrics loaded from evaluation CSVs (baseline_eval.csv, "
            "deep_eval.csv, llm_eval_summary.csv)."
        )
    else:
        st.info(
            "Metrics files not found in results/. Please run:\n"
            "python src/evaluate.py --data_dir data/aclImdb --models_dir models --llm_csv results/llm_eval_clean.csv --out_dir results"
        )


# ==============================
# TAB 2: Chatbot
# ==============================

with tab_chat:
    st.subheader("Chat with the Sentiment Bot")

    if not LLM_AVAILABLE:
        st.info("LLM is disabled (no OPENAI_API_KEY). Set your key to enable the chatbot.")
    else:
        for turn in st.session_state.chat_history:
            st.markdown(f"**You:** {turn['user']}")
            st.markdown(f"**Bot:** {turn['bot']}")

        user_msg = st.text_input(
            "Type your message about movies or reviews:",
            key="chat_input",
        )

        send_btn = st.button("Send")
        if send_btn and user_msg.strip():
            reply = call_llm_chat(user_msg.strip(), st.session_state.chat_history)
            st.session_state.chat_history.append(
                {"user": user_msg.strip(), "bot": reply}
            )
            st.rerun()
