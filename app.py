from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pdfplumber
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def train_or_load_model(data):
    X = data[['year', 'msp', 'irrigated area', 'rain fall', 'fertilizer usage']]
    y = data['production']
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    return model

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pdf' not in request.files or 'year' not in request.form:
        return jsonify({'error': 'PDF file and prediction year are required.'}), 400

    try:
        target_year = int(request.form['year'])
    except ValueError:
        return jsonify({'error': 'Invalid year format'}), 400

    pdf_file = request.files['pdf']
    file_path = os.path.join(UPLOAD_FOLDER, pdf_file.filename)
    pdf_file.save(file_path)

    try:
        # Extract table from PDF
        with pdfplumber.open(file_path) as pdf:
            all_tables = []
            for page in pdf.pages:
                table = page.extract_table()
                if table:
                    all_tables.extend(table)

        if not all_tables:
            return jsonify({'error': 'No tables found in PDF'}), 400

        df = pd.DataFrame(all_tables[1:], columns=all_tables[0])
        df.columns = [col.strip().lower() if col else '' for col in df.columns]

        column_renames = {
            'irrigatedarea': 'irrigated area',
            'irrigated_area': 'irrigated area',
            'rainfall': 'rain fall',
            'fertilizerusage': 'fertilizer usage',
            'fertilizer_usage': 'fertilizer usage'
        }
        df.rename(columns=column_renames, inplace=True)

        required_cols = ['year', 'production', 'msp', 'irrigated area', 'rain fall', 'fertilizer usage']
        for col in required_cols:
            if col not in df.columns:
                return jsonify({'error': f'Missing required column: \"{col}\"'}), 400

        # Type conversions
        df['year'] = df['year'].astype(int)
        df['production'] = df['production'].str.replace(',', '').astype(float)
        df['msp'] = df['msp'].str.replace(',', '').astype(float)
        df['irrigated area'] = df['irrigated area'].str.replace(',', '').astype(float)
        df['rain fall'] = df['rain fall'].str.replace(',', '').astype(float)
        df['fertilizer usage'] = df['fertilizer usage'].str.replace(',', '').astype(float)

        # Filter data before the target year
        train_df = df[df['year'] < target_year]
        if train_df.empty:
            return jsonify({'error': f'No data available for years before {target_year}'}), 400

        model = train_or_load_model(train_df)

        # Predict using average of known features
        input_row = pd.DataFrame([{
            'year': target_year,
            'msp': train_df['msp'].mean(),
            'irrigated area': train_df['irrigated area'].mean(),
            'rain fall': train_df['rain fall'].mean(),
            'fertilizer usage': train_df['fertilizer usage'].mean()
        }])

        prediction = model.predict(input_row)

        # Save processed data
        df['source_pdf'] = pdf_file.filename
        df.to_csv(f'uploads/processed_{pdf_file.filename}.csv', index=False)

        with open('uploads/predictions.csv', 'a') as f:
            f.write(f"{pdf_file.filename},{target_year},{round(prediction[0], 2)}\n")

        return jsonify({
            'message': 'Prediction successful',
            'predicted_year': target_year,
            'predicted_production': float(round(prediction[0], 2))  # in tonnes
        })

    except Exception as e:
        return jsonify({'error': f'Failed to process PDF: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)














