import pandas as pd
import sqlite3
import os
import time
import schedule
import logging
import requests
from sklearn.metrics import mean_squared_error
from evidently import Report, Regression, Dataset
from evidently.presets import DataDriftPreset, RegressionPreset
from evidently import DataDefinition
from prometheus_client import start_http_server, Gauge

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configurações
DB_PATH = os.getenv("DB_PATH", "/app/data/monitoring.db")
REFERENCE_DATA_PATH = os.getenv("REFERENCE_DATA_PATH", "/app/data/preprocessed/final_subsampled_dataset.parquet")
REPORT_OUTPUT_PATH = os.getenv("REPORT_OUTPUT_PATH", "/app/reports")
MIN_DATA_POINTS = int(os.getenv("MIN_DATA_POINTS", "50"))

# Configurações GitHub Actions
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_OWNER = os.getenv("REPO_OWNER", "Tomas7025")
REPO_NAME = os.getenv("REPO_NAME", "proj_LAIA")

# Métricas Prometheus
DRIFT_DETECTED_GAUGE = Gauge('evidently_drift_detected', '1 if drift is detected, 0 otherwise')
DATA_DRIFT_SCORE = Gauge('evidently_data_drift_score', 'Share of drifted features')
TARGET_DRIFT_SCORE = Gauge('evidently_target_drift_score', 'Drift score for the target')
MSE_SCORE = Gauge('mse_score','MSE score')

def trigger_retraining_workflow():
    """Dispara um workflow no GitHub Actions via repository_dispatch"""
    if not GITHUB_TOKEN:
        logger.error("GITHUB_TOKEN not set. Cannot trigger workflow.")
        return

    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/dispatches"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "event_type": "retrain_model",
        "client_payload": {
            "reason": "drift_detected",
            "timestamp": time.time()
        }
    }

    try:
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 204:
            logger.info("🚀 Successfully triggered retraining workflow on GitHub!")
        else:
            logger.error(f"Failed to trigger workflow: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"Error triggering workflow: {e}")

def load_reference_data():
    logger.info("DEBUG 1")
    """Carrega os dados de treino (referência)"""
    try:
        if os.path.exists(REFERENCE_DATA_PATH):
            logger.info("DEBUG 2")
            df = pd.read_parquet(REFERENCE_DATA_PATH)
            df['predicted_duration'] = df['duration']
            logger.info("DEBUG 3")
            logger.info(f"Reference data loaded. Shape: {df.shape}")
            logger.info("DEBUG 4")
            return df
        else:
            logger.info("DEBUG 5")
            logger.error(f"Reference data not found at {REFERENCE_DATA_PATH}")
            return None
    except Exception as e:
        logger.info("DEBUG 6")
        logger.error(f"Error loading reference data: {e}")
        return None

def fetch_current_data():
    """
    Lê os dados do SQLite.
    Faz o JOIN entre predições e labels para ter o dataset completo.
    """
    logger.info("DEBUG 8")
    try:
        logger.info("DEBUG 9")
        conn = sqlite3.connect(DB_PATH)
        logger.info("DEBUG 10")
        
        # Query para juntar Features + Predição + Label (Target)
        # Assumindo que tens tabelas 'predictions' e 'labels' com 'request_id' comum
        query = """
        SELECT 
            p.*, 
            l.real_duration as real_duration
        FROM predictions p
        INNER JOIN labels l ON p.data_id = l.label_id AND p.request_id=l.request_id
        WHERE p.was_checked = 0;
        """
        
        df = pd.read_sql_query(query, conn)
        df = df[['VendorID', 'passenger_count', 'trip_distance',
                'pickup_hour', 'pickup_day_of_week', 'pickup_month', 'pickup_day',
                'PULocationID', 'DOLocationID', 'predicted_duration', 'real_duration']]
        df["duration"] = df["real_duration"]
        df.drop('real_duration',axis=1,inplace=True)
        logger.info(f"Columns: {df.columns}")
        logger.info(f'Dataframe: {df}')
        logger.info(f'Missing Values: {df.isnull().sum()}')
        logger.info(f"DEBUG 11 - Retrieved {df.shape[0]} rows and {df.shape[1]} columns from database")
        
        conn.close()
        
        # Limpeza básica (remover colunas de metadados se necessário)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        return df
    except Exception as e:
        logger.error(f"Error fetching data from SQLite: {e}")
        return pd.DataFrame()

def calculate_metrics(df):

    mse = mean_squared_error(df['duration'],df['predicted_duration'])

    MSE_SCORE.set(mse)

    return

def run_analysis():
    logger.info("Starting monitoring cycle...")
    
    reference_data = load_reference_data()
    logger.info("DEBUG 7")
    current_data = fetch_current_data()

    calculate_metrics(current_data)
    
    if reference_data is None:
        logger.warning("Skipping analysis: No reference data.")
        return

    if current_data.empty or len(current_data) < MIN_DATA_POINTS:
        logger.info(f"Not enough data to analyze. Current rows with labels: {len(current_data)}")
        return

    conn = sqlite3.connect(DB_PATH)

    bool_update = """
        UPDATE predictions
        SET was_checked = 1
        WHERE was_checked = 0;
    """
    
    conn.execute(bool_update)
    logger.info("DEBUG 12")

    conn.commit()
    conn.close()

    logger.info(f"Analyzing {len(current_data)} rows of production data...")

    # 1. Configurar o Relatório do Evidently
    # Usamos Presets para facilitar: Drift de Dados e Qualidade de Regressão (RMSE, MAE, etc)
    
    data_definition = DataDefinition(
        numerical_columns = ["trip_distance"],
        categorical_columns = ["VendorID","passenger_count","pickup_hour","pickup_day_of_week","pickup_month","pickup_day","PULocationID","DOLocationID"],
        regression = [Regression(target='duration',prediction='predicted_duration')],
    )

    report = Report(metrics=[
        DataDriftPreset()
    ])

    try:
        # 2. Correr o Relatório
        # Nota: Garante que as colunas do current_data batem certo com o reference_data
        # O Evidently ignora colunas extra no current, mas precisa das features do reference.
        report.data_definition = data_definition

        # ???
        reference_data_ev = Dataset.from_pandas(
            pd.DataFrame(reference_data),
            data_definition=data_definition,
        )
        
        current_data_ev = Dataset.from_pandas(
            pd.DataFrame(current_data),
            data_definition=data_definition,
        )
        # ???

        my_eval = report.run(reference_data=reference_data_ev, current_data=current_data_ev)
        my_eval.save_json('drift_report.json')

        # 3. Guardar o Relatório HTML
        os.makedirs(REPORT_OUTPUT_PATH, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        report_path = os.path.join(REPORT_OUTPUT_PATH, f"drift_report_{timestamp}.html")
        # report.save_html(report_path)
        logger.info(f"Report saved to {report_path}")

        # 4. Verificar Resultados (Lógica de Decisão)
        metrics = my_eval.dict()

        logger.info(f"Passed Dict")
        logger.info(metrics)
        # Extrair métricas específicas para o Prometheus
        # A estrutura exata depende do preset, mas vamos tentar extrair o share de drift
        try:
            # Data Drift
            data_drift_share = metrics['metrics'][0]['value']['share']
            DATA_DRIFT_SCORE.set(data_drift_share)
            data_drift = data_drift_share > 0
            
            # Concept drift
            # target_drift_score = int(metrics['metrics'][1]['value']['share'] > 0.5)
            # TARGET_DRIFT_SCORE.set(target_drift_score)
            
            # Decisão Global de Drift ???
            # dataset_drift = True if data_drift_share == 1 or target_drift_score == 1 else False
            
            if data_drift:
                DRIFT_DETECTED_GAUGE.set(1)
                logger.warning("🚨 DRIFT DETECTED! 🚨")
                trigger_retraining_workflow()
            else:
                DRIFT_DETECTED_GAUGE.set(0)
                logger.info("✅ No significant drift detected.")
                
        except KeyError as e:
            logger.error(f"Could not extract metrics from report: {e}")

    except Exception as e:
        logger.error(f"Failed to run Evidently report: {e}")

def main():
    logger.info("Evidently Monitoring Service Started")
    
    # Iniciar servidor Prometheus na porta 8000
    start_http_server(8000)
    logger.info("Prometheus metrics exposed on port 8000")
    
    # Agendar para correr a cada X tempo (ex: 1 minuto para testes, 1 hora em prod)
    schedule.every(1).minutes.do(run_analysis)
    
    # Correr uma vez logo ao arranque para testar
    run_analysis()
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()
