# Customer feedback analysis dashboard

Upload a labeled CSV, train a three-class TensorFlow model, then upload another CSV for batch sentiment predictions and a chart. Training labels must be `negative`, `neutral`, or `positive`; the file needs at least 12 nonempty feedback rows and all three classes. The app reports validation accuracy from a held-out 20% of the uploaded training data. Small datasets give unreliable accuracy, so use a substantial representative dataset for meaningful results.

## Run

```bash
python -m venv .venv
# Activate the virtual environment, then:
pip install -r requirements.txt
streamlit run app.py
```

The app accepts CSV columns of your choice. Training happens locally when you press **Train classifier**; uploaded files are not included in this repository. No pretrained model or customer data is shipped.

