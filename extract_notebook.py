import json

def extract():
    with open('d:/CODE/Amazon_Classifier/notebooks/02_full_pipeline_feature_engineering_and_training.ipynb', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open('d:/CODE/Amazon_Classifier/notebooks/02_code.py', 'w', encoding='utf-8') as f:
        for cell in data['cells']:
            if cell.get('cell_type') == 'code':
                f.write(''.join(cell.get('source', [])) + '\n\n')

extract()
