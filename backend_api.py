"""
==============================================
RETAIL DEMAND FORECASTING - BACKEND API
==============================================
Flask REST API for Machine Learning Model
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables
model = None
le_season = None
feature_columns = None
df = None

print("=" * 60)
print("RETAIL DEMAND FORECASTING - BACKEND API")
print("=" * 60)

def generate_data():
    """Generate synthetic dataset"""
    print("\nGenerating synthetic dataset...")
    np.random.seed(42)
    
    n_records = 12000
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = np.random.choice(date_range, size=n_records, replace=True)
    
    n_stores = 10
    n_items = 50
    store_ids = np.random.randint(1, n_stores + 1, size=n_records)
    item_ids = np.random.randint(1, n_items + 1, size=n_records)
    
    base_price = np.random.uniform(10, 100, size=n_records)
    promotion = np.random.choice([0, 1], size=n_records, p=[0.7, 0.3])
    price = base_price * (1 - promotion * 0.2)
    
    holidays = pd.to_datetime(['2021-01-01', '2021-07-04', '2021-12-25',
                              '2022-01-01', '2022-07-04', '2022-12-25',
                              '2023-01-01', '2023-07-04', '2023-12-25'])
    
    df_new = pd.DataFrame({
        'date': dates,
        'store_id': store_ids,
        'item_id': item_ids,
        'price': price
    })
    
    df_new['date'] = pd.to_datetime(df_new['date'])
    df_new['year'] = df_new['date'].dt.year
    df_new['month'] = df_new['date'].dt.month
    df_new['day'] = df_new['date'].dt.day
    df_new['day_of_week'] = df_new['date'].dt.dayofweek
    df_new['week_of_year'] = df_new['date'].dt.isocalendar().week
    df_new['quarter'] = df_new['date'].dt.quarter
    df_new['holiday'] = df_new['date'].isin(holidays).astype(int)
    df_new['season'] = df_new['month'].apply(lambda x: 'Winter' if x in [12, 1, 2] 
                                       else 'Spring' if x in [3, 4, 5]
                                       else 'Summer' if x in [6, 7, 8]
                                       else 'Fall')
    df_new['promotion'] = promotion
    
    base_demand = 50
    trend = df_new['year'] * 5 + df_new['month'] * 2
    seasonality = 20 * np.sin(2 * np.pi * df_new['month'] / 12)
    weekly_pattern = 15 * (df_new['day_of_week'] >= 5)
    price_effect = -0.5 * (df_new['price'] - 50)
    promotion_effect = 30 * df_new['promotion']
    holiday_effect = 40 * df_new['holiday']
    store_effect = np.random.normal(0, 5, size=len(df_new))
    item_effect = np.random.normal(0, 10, size=len(df_new))
    noise = np.random.normal(0, 10, size=len(df_new))
    
    df_new['sales'] = (base_demand + trend + seasonality + weekly_pattern + 
                   price_effect + promotion_effect + holiday_effect + 
                   store_effect + item_effect + noise)
    df_new['sales'] = np.maximum(df_new['sales'], 0).astype(int)
    
    df_new = df_new.sort_values('date').reset_index(drop=True)
    df_new.to_csv('retail_demand_data.csv', index=False)
    
    print("✓ Dataset generated successfully")
    return df_new

def train_model():
    """Train ML model"""
    global model, le_season, feature_columns, df
    
    print("\nTraining machine learning model...")
    
    # Load or generate data
    if os.path.exists('retail_demand_data.csv'):
        df = pd.read_csv('retail_demand_data.csv')
        df['date'] = pd.to_datetime(df['date'])
        print("✓ Dataset loaded")
        
        # Check if date features exist, if not create them
        if 'year' not in df.columns:
            print("Creating date features...")
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['day_of_week'] = df['date'].dt.dayofweek
            df['week_of_year'] = df['date'].dt.isocalendar().week
            df['quarter'] = df['date'].dt.quarter
            
        # Check if season exists
        if 'season' not in df.columns:
            df['season'] = df['month'].apply(lambda x: 'Winter' if x in [12, 1, 2] 
                                           else 'Spring' if x in [3, 4, 5]
                                           else 'Summer' if x in [6, 7, 8]
                                           else 'Fall')
    else:
        df = generate_data()
    
    # Feature engineering
    le_season = LabelEncoder()
    df['season_encoded'] = le_season.fit_transform(df['season'])
    
    feature_columns = ['store_id', 'item_id', 'price', 'promotion', 'holiday',
                      'year', 'month', 'day', 'day_of_week', 'week_of_year',
                      'quarter', 'season_encoded']
    
    X = df[feature_columns]
    y = df['sales']
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    with open('models/rf_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/le_season.pkl', 'wb') as f:
        pickle.dump(le_season, f)
    with open('models/feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_columns, f)
    
    print("✓ Model trained and saved")

# API Endpoints

@app.route('/')
def home():
    return jsonify({
        'message': 'Retail Demand Forecasting API',
        'status': 'running',
        'version': '1.0'
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'dataset_loaded': df is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/stats')
def get_stats():
    if df is None:
        return jsonify({'error': 'Dataset not loaded'}), 500
    
    stats = {
        'total_records': len(df),
        'date_range': {
            'start': df['date'].min().isoformat(),
            'end': df['date'].max().isoformat()
        },
        'stores': {
            'count': int(df['store_id'].nunique()),
            'ids': sorted(df['store_id'].unique().tolist())
        },
        'items': {
            'count': int(df['item_id'].nunique()),
            'ids': sorted(df['item_id'].unique().tolist())
        },
        'sales': {
            'mean': float(df['sales'].mean()),
            'median': float(df['sales'].median()),
            'min': int(df['sales'].min()),
            'max': int(df['sales'].max()),
            'total': int(df['sales'].sum())
        },
        'price': {
            'mean': float(df['price'].mean()),
            'min': float(df['price'].min()),
            'max': float(df['price'].max())
        }
    }
    
    return jsonify(stats)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Parse date
        pred_date = pd.to_datetime(data['date'])
        
        # Get season
        month = pred_date.month
        season = ('Winter' if month in [12, 1, 2] 
                 else 'Spring' if month in [3, 4, 5]
                 else 'Summer' if month in [6, 7, 8]
                 else 'Fall')
        
        # Prepare features
        features = {
            'store_id': data['store_id'],
            'item_id': data['item_id'],
            'price': data['price'],
            'promotion': data.get('promotion', 0),
            'holiday': data.get('holiday', 0),
            'year': pred_date.year,
            'month': pred_date.month,
            'day': pred_date.day,
            'day_of_week': pred_date.dayofweek,
            'week_of_year': pred_date.isocalendar().week,
            'quarter': (pred_date.month - 1) // 3 + 1,
            'season_encoded': le_season.transform([season])[0]
        }
        
        X = pd.DataFrame([features])[feature_columns]
        prediction = model.predict(X)[0]
        prediction = max(0, prediction)
        
        return jsonify({
            'prediction': float(prediction),
            'input': data,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        data = request.get_json()
        
        store_id = data['store_id']
        item_id = data['item_id']
        start_date = pd.to_datetime(data['start_date'])
        days = data['days']
        price = data.get('price', df['price'].mean())
        promotion = data.get('promotion', 0)
        holiday = data.get('holiday', 0)
        
        forecast_dates = pd.date_range(start=start_date, periods=days, freq='D')
        predictions = []
        
        for date in forecast_dates:
            month = date.month
            season = ('Winter' if month in [12, 1, 2] 
                     else 'Spring' if month in [3, 4, 5]
                     else 'Summer' if month in [6, 7, 8]
                     else 'Fall')
            
            features = {
                'store_id': store_id,
                'item_id': item_id,
                'price': price,
                'promotion': promotion,
                'holiday': holiday,
                'year': date.year,
                'month': date.month,
                'day': date.day,
                'day_of_week': date.dayofweek,
                'week_of_year': date.isocalendar().week,
                'quarter': (date.month - 1) // 3 + 1,
                'season_encoded': le_season.transform([season])[0]
            }
            
            X = pd.DataFrame([features])[feature_columns]
            pred = max(0, model.predict(X)[0])
            
            predictions.append({
                'date': date.strftime('%Y-%m-%d'),
                'predicted_sales': float(pred),
                'day_of_week': date.day_name()
            })
        
        pred_values = [p['predicted_sales'] for p in predictions]
        
        summary = {
            'total_predicted_sales': sum(pred_values),
            'average_daily_sales': np.mean(pred_values),
            'min_daily_sales': min(pred_values),
            'max_daily_sales': max(pred_values),
            'std_dev': np.std(pred_values)
        }
        
        return jsonify({
            'forecast': predictions,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Initialize
train_model()

print("\n" + "=" * 60)
print("Starting Flask server...")
print("API will be available at: http://localhost:5000")
print("=" * 60)
print("\nAvailable endpoints:")
print("  GET  /health     - Health check")
print("  GET  /stats      - Statistics")
print("  POST /predict    - Single prediction")
print("  POST /forecast   - Multi-day forecast")
print("=" * 60 + "\n")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)