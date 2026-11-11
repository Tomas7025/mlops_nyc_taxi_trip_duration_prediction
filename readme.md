# End-to-End MLOps Pipeline for Taxi Trip Duration Prediction

Project for the course "Advanced Artificial Intelligence Laboratory" at the University of Coimbra, 2025/26.

## Summary

This work consisted on the implementation of an end-to-end MLOps pipeline to predict taxi trip duration. This included data preprocessing (feature engineering/extraction and preprocessing), model training, model and metric tracking, data versioning, a CI/CD pipeline, performance monitoring and drift detection.

![Project Architecture](images\mlops-architecture.png)

## Goals

- Deploy and serve a model that predicts taxi trip duration based on historical data;
- Implement an automatic MLOps pipeline.

## Description

This project used data from [New York City Taxi dataset](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), specifically data from 2011 and 2012 for training and data from 2013 for testing.

### Feature Engineering / Extraction

The duration of a taxi trip was calculated as `tpep_dropoff_datetime - tpep_pickup_datetime`, with the first column being dropped after calculation. 
Afterwards, we extracted `pickup_hour`, `pickup_day`, `pickup_day_of_week` and `pickup_month` from the `tpep_pickup_datetime` column. Because nearly every timestamp is unique, the raw value gives the models little to generalise from, whereas these components capture recurring patterns (such as rush hours, weekends and seasonal demand) that we expected the models to learn more easily. 
Finally, `tpep_pickup_datetime` was dropped alongside other columns not provided by the professors during the testing of the deployed models (these were `RatecodeID`, `store_and_fwd_flag`, `payment_type`, `fare_amount`, `extra`, `mta_tax`, `tip_amount`, `tolls_amount`, `improvement_surcharge`, `total_amount`,`congestion_surcharge` and `airport_fee`).

### Preprocessing

Due to resource limitations, only 2.5% of the original data was used to train the models. This subsample was done randomly, but ensuring the same number of registries per month, day and hour was kept, as to avoid biasing the data to certain times. 
Next, the numerical columns were normalized using a StandardScaler, and the numerical column `VendorID` was one hot encoded. The remaining categorical columns were not encoded due to the variability of classes present in them. 
Finally, we treated training cases where `trip_distance=0` as missing data, and changed this to the mean of this column.

### Model Training

Two models were trained on this data: a LinearRegressor and a Histogram-based Gradient Boosting classifier. For the histogram-based classifier a GridSearch was performed among the following parameters:

| Parameters            | Values    |
|-----------------------|-----------|
| learning_rate         | 0.01, 0.1 |
| l2_regularization     | 0.01, 0.1 |
| max_leaf_nodes        | 15, 31    |
| max_iter              | 200, 400  |

These models are trained on the train data and validated on the validation data. The model with the best score (lowest MSE) in the validation data is given the proper alias in MLFlow.

### MLFlow

MLFlow is used to track all models, their respective scores, their parameters, and the pipeline used for data preprocessing (which is then used on the deployement). Tags are used to move models between stages.

### DVC

DVC was used for data versioning, hosting the data in on of the VMs.

### CI/CD pipeline

The GitHub Actions CI/CD pipeline was divided into four workflows: **Continuous Integration**, **Continuous Delivery**, **Continuous Staging**, and **Continuous Deployment**. This pipeline ran on self-hosted runners, with the first three workflows running on VM-1 (the training VM) and the last running on VM-2 (the deployment VM). This division was made to ensure the training environment was kept separate from the deployment environment.

Regarding the workflows, **Continuous Integration** ran on every commit to the master branch. It started by running unit tests, before building and pushing the training and serving Docker images. 

**Continuous Delivery** ran only if the previous workflow was successful. It started by getting the training Docker image from the GitHub Container Registry (GHCR), installing DVC and pulling the datasets saved there, and running the training container. Afterwards, the quality of the best model is checked, and if its MSE is above the defined threshold, the workflow fails. Otherwise, we pull the serving Docker image and clean any serving containers if they exist, before running the serving container. This is followed by the end-to-end tests and, if they are successful, the staging image is tagged as serving:staging and is pushed to GHCR. Finally, we promote the model to staging by changing its alias.

Afterwards, **Continuous Staging** started by getting the serving:staging Docker image from the GHCR and cleaning any serving containers, if any exist. We then run the serving container and run the end-to-end tests again, and if everything is successful, we promote the model to production and change the container tag to serving:production.

Finally, **Continuous Deployment** started by pulling the Docker image now tagged as serving:production from the GHCR, and then stops and deletes the serving container from the previous step and runs the recently pulled one. 
This container is then available to receive prediction requests, waiting for them on the 9001 port.

### Monitoring

The deployed service was monitored using Prometheus, Graphana and an Alertmanager. 

**Instrumentation**
The Flask API serves predictions on port `9001` and exposes metrics on port `8080` (`/metrics`) using `prometheus_client`:
- **Counters:** `http_requests_total` (labelled by status code, used for the error rate) and `model_predictions_total` (labelled by vendor)
- **Histograms:** API latency, input trip distance, and predicted trip duration (custom 0–120 min buckets for accurate quantiles such as P95)
- **Gauges:** `model_last_prediction`, the most recent model output

**Collection & visualisation**
- Prometheus scrapes the API every 5 seconds. 
- Grafana dashboards and data sources are provisioned automatically from YAML/JSON files, so the setup is reproducible and version-controlled.

**Alerting**
- Prometheus alert rules cover availability (error rate > 10%) and data drift.
- Alertmanager sent alerts to the team's Discord channel. Drift alerts also triggered the Continuous Training pipeline in GitHub Actions, as will be explained next.

### Drift Detection

To check for data drift, EvidentlyAI was used to compare the training data with the data provided to the deployed model. To detect drift, the package uses Wasserstein (for numerical columns) and Jensen–Shannon (for categorical columns) statistical tests. If any drift was detected, we trigger the GitHub training pipeline to retrain the model and change the serving image.

To check for concept drift, we logged MSE into a Gauge object in Prometheus and logged a Counter that is increased when MSE goes over a certain threshold. If the MSE went over the threshold 5 or more times in 5 hours, we once again triggered the GitHub training pipeline to retrain the model and change the serving image.

