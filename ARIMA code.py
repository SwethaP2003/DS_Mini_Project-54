from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mysql.connector
from datetime import datetime, timedelta
import warnings
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress ARIMA warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'crop_prediction'
}

def get_database_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as err:
        logger.error(f"Database Connection Error: {err}")
        return None

def train_arima_model(data, order=(1,1,1)):
    """
    Train ARIMA model and return the fitted model
    """
    try:
        model = ARIMA(data, order=order)
        fitted_model = model.fit()
        logger.info("ARIMA model training completed")
        logger.info(f"Model summary:\n{fitted_model.summary()}")
        return fitted_model
    except Exception as e:
        logger.error(f"Error training ARIMA model: {e}")
        raise
@app.route('/predict', methods=['POST'])
def predict_crop_yield():
    try:
        # Parse request data
        data = request.get_json()
        crop_type = data.get('crop_type', 'wheat')
        forecast_periods = data.get('periods', 12)
        
        logger.info(f"Received prediction request for crop: {crop_type}, periods: {forecast_periods}")

        # Database connection
        conn = get_database_connection()
        if not conn:
            return jsonify({'error': 'Database connection failed'}), 500

        # Fetch historical data
        cursor = conn.cursor(dictionary=True)
        query = """
        SELECT 
            date, 
            yield_amount
        FROM crop_data 
        WHERE crop_type = %s 
        ORDER BY date
        """
        cursor.execute(query, (crop_type,))
        historical_data = cursor.fetchall()
        cursor.close()
        conn.close()

        if not historical_data:
            return jsonify({'error': 'No historical data found for the specified crop type'}), 404

        # Convert to DataFrame and ensure proper data types
        df = pd.DataFrame(historical_data)
        df['date'] = pd.to_datetime(df['date'])
        df['yield_amount'] = pd.to_numeric(df['yield_amount'], errors='coerce')
        
        # Drop any rows with NaN values
        df = df.dropna()
        
        if len(df) < 10:  # Minimum data points needed for ARIMA
            return jsonify({'error': 'Insufficient data points for prediction'}), 400
            
        df.set_index('date', inplace=True)
        
        # Convert to numpy array
        yield_data = np.asarray(df['yield_amount'].values, dtype=np.float64)
        
        # Print data overview
        logger.info("\nHistorical Data Overview:")
        logger.info(f"Total records: {len(df)}")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        logger.info("\nYield Statistics:")
        logger.info(df['yield_amount'].describe())

        # Split data for validation
        train_size = int(len(yield_data) * 0.8)
        train_data = yield_data[:train_size]
        test_data = yield_data[train_size:]

        # Train ARIMA model
        try:
            model = ARIMA(train_data, order=(2,1,2))
            fitted_model = model.fit()
        except Exception as e:
            logger.error(f"ARIMA model training failed: {e}")
            return jsonify({'error': 'Failed to train ARIMA model'}), 500

        # Make predictions on test data
        try:
            test_predictions = fitted_model.forecast(steps=len(test_data))
            
            # Calculate metrics
            mse = mean_squared_error(test_data, test_predictions)
            mae = mean_absolute_error(test_data, test_predictions)
            rmse = np.sqrt(mse)
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return jsonify({'error': 'Failed to calculate prediction metrics'}), 500

        # Generate future dates
        last_date = df.index.max()
        future_dates = [last_date + timedelta(days=30*i) for i in range(1, forecast_periods+1)]
        
        # Make future predictions
        try:
            future_predictions = fitted_model.forecast(steps=forecast_periods)
            
            # Calculate prediction intervals (95% confidence)
            forecast_std = np.std(yield_data) * np.sqrt(1 + np.arange(forecast_periods) / len(yield_data))
            conf_interval = 1.96 * forecast_std
            
            # Prepare forecast data
            forecast_data = [
                {
                    'date': date.strftime('%Y-%m-%d'),
                    'predicted_yield': float(pred),
                    'lower_ci': float(max(0, pred - ci)),  # Ensure non-negative yields
                    'upper_ci': float(pred + ci)
                }
                for date, pred, ci in zip(future_dates, future_predictions, conf_interval)
            ]
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return jsonify({'error': 'Failed to generate predictions'}), 500

        # Print predictions to console
        logger.info("\nModel Performance Metrics:")
        logger.info(f"MSE: {mse:.2f}")
        logger.info(f"MAE: {mae:.2f}")
        logger.info(f"RMSE: {rmse:.2f}")
        
        logger.info("\nFuture Predictions:")
        for pred in forecast_data:
            logger.info(
                f"Date: {pred['date']}, "
                f"Predicted Yield: {pred['predicted_yield']:.2f}, "
                f"CI: [{pred['lower_ci']:.2f}, {pred['upper_ci']:.2f}]"
            )

        response_data = {
            'crop_type': crop_type,
            'predictions': forecast_data,
            'metrics': {
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse)
            },
            'model_info': {
                'type': 'ARIMA',
                'order': '(2,1,2)',
                'training_size': train_size,
                'test_size': len(test_data)
            }
        }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e)
        }), 500

@app.route('/analytics', methods=['GET'])
def get_analytics():
    try:
        conn = get_database_connection()
        if not conn:
            return jsonify({'error': 'Database connection failed'}), 500

        cursor = conn.cursor(dictionary=True)
        
        # Total records
        cursor.execute("SELECT COUNT(*) as total_records FROM crop_data")
        total_records = cursor.fetchone()['total_records']
        
        # Crop distribution and statistics
        cursor.execute("""
            SELECT 
                crop_type, 
                COUNT(*) as record_count,
                ROUND(AVG(yield_amount), 2) as avg_yield,
                ROUND(MIN(yield_amount), 2) as min_yield,
                ROUND(MAX(yield_amount), 2) as max_yield,
                ROUND(STDDEV(yield_amount), 2) as yield_std
            FROM crop_data 
            GROUP BY crop_type
        """)
        crop_distribution = cursor.fetchall()
        
        # Time series analysis
        cursor.execute("""
            SELECT 
                DATE_FORMAT(date, '%Y-%m') as month,
                crop_type,
                ROUND(AVG(yield_amount), 2) as avg_yield,
                COUNT(*) as sample_size
            FROM crop_data
            GROUP BY month, crop_type
            ORDER BY month, crop_type
        """)
        time_series = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        # Print analytics to console
        logger.info("\nAnalytics Summary:")
        logger.info(f"Total Records: {total_records}")
        
        logger.info("\nCrop Distribution:")
        for crop in crop_distribution:
            logger.info(
                f"Crop: {crop['crop_type']}, "
                f"Records: {crop['record_count']}, "
                f"Avg Yield: {crop['avg_yield']}, "
                f"Std: {crop['yield_std']}"
            )

        return jsonify({
            'total_records': total_records,
            'crop_distribution': crop_distribution,
            'time_series': time_series
        })
    
    except Exception as e:
        logger.error(f"Analytics Error: {e}")
        return jsonify({
            'error': 'Failed to retrieve analytics',
            'details': str(e)
        }), 500

@app.route('/upload', methods=['POST'])
def upload_data():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Read CSV file
        df = pd.read_csv(file)
        
        # Validate CSV columns
        required_columns = ['date', 'crop_type', 'yield_amount']
        if not all(col in df.columns for col in required_columns):
            return jsonify({'error': 'Invalid CSV format'}), 400
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df.sort_values('date', inplace=True)
        
        # Database connection
        conn = get_database_connection()
        cursor = conn.cursor()
        
        # Insert data
        insert_query = """
        INSERT INTO crop_data 
        (date, crop_type, yield_amount) 
        VALUES (%s, %s, %s)
        """
        
        records_uploaded = 0
        for _, row in df.iterrows():
            cursor.execute(insert_query, (
                row['date'].strftime('%Y-%m-%d'), 
                row['crop_type'], 
                row['yield_amount']
            ))
            records_uploaded += 1
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"\nData Upload Summary:")
        logger.info(f"Records uploaded: {records_uploaded}")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        return jsonify({
            'records_uploaded': records_uploaded,
            'message': 'Data uploaded successfully'
        })
    
    except Exception as e:
        logger.error(f"Upload Error: {e}")
        return jsonify({
            'error': 'File upload failed',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    logger.info("Starting Crop Prediction API with ARIMA modeling...")
    app.run(debug=True, host='0.0.0.0', port=5000)