import pandas as pd
import streamlit as st

from feedback import predict_feedback, train_model

st.set_page_config(page_title="Customer feedback analysis", layout="wide")
st.title("Customer feedback analysis")
st.caption("Train a TensorFlow sentiment classifier on labeled feedback, then score a feedback CSV.")

with st.expander("1. Train a model", expanded=True):
    training_file = st.file_uploader("Labeled training CSV", type="csv", key="training")
    if training_file:
        try:
            training = pd.read_csv(training_file)
            st.dataframe(training.head(), use_container_width=True)
            text_col = st.selectbox("Feedback text column", training.columns, key="train_text")
            label_col = st.selectbox("Sentiment label column", training.columns, key="train_label")
            epochs = st.slider("Training epochs", 2, 30, 8)
            if st.button("Train classifier", type="primary"):
                with st.spinner("Training model..."):
                    st.session_state.model, history = train_model(training, text_col, label_col, epochs)
                st.success("Model trained. Validation accuracy is shown below.")
                st.line_chart(pd.DataFrame(history)[["accuracy", "val_accuracy"]])
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(str(exc))

with st.expander("2. Analyze feedback", expanded=True):
    feedback_file = st.file_uploader("Feedback CSV", type="csv", key="feedback")
    if feedback_file:
        try:
            feedback = pd.read_csv(feedback_file)
            feedback_col = st.selectbox("Feedback text column", feedback.columns, key="feedback_text")
            if st.button("Analyze feedback"):
                if "model" not in st.session_state:
                    st.warning("Train a model first.")
                else:
                    result = predict_feedback(st.session_state.model, feedback, feedback_col)
                    st.metric("Feedback rows", len(result))
                    st.bar_chart(result["predicted_sentiment"].value_counts())
                    st.dataframe(result, use_container_width=True)
                    st.download_button("Download predictions", result.to_csv(index=False), "feedback_predictions.csv", "text/csv")
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(str(exc))

