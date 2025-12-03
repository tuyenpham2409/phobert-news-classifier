import pandas as pd

try:
    df = pd.read_csv(r"c:\Users\DELL\Downloads\NLP Project\NLP Project\Data\vnexpress_processed_vncorenlp_for_phobert.csv")
    label_mapping = df[['label', 'label_id']].drop_duplicates().sort_values('label_id')
    print(label_mapping)
except Exception as e:
    print(e)
