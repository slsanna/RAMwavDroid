#!/usr/bin/env python3
import argparse
import ast
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from implementations.classifier_rf import RandomForestTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def loadCsvData(csvPath, targetColumn):
    logging.info(f"Loading CSV data from: {csvPath}")
    dataFrame = pd.read_csv(csvPath)
    logging.info(f"Loaded {len(dataFrame)} samples from CSV")
    
    columnsToIgnore = [
        'filename', 'file', 'path', 'id', 'name', 'apk_id',
        'sample_rate', 'num_samples', 'duration_sec', 'duration',
        'length', 'size', 'timestamp', 'date',
        'num_regions', 'region_types', 'dl_embedding_dim'
    ]
    
    featureColumns = dataFrame.columns[~dataFrame.columns.str.lower().isin(columnsToIgnore)]
    featureColumns = featureColumns[featureColumns != targetColumn]
    logging.info(f"Found {len(featureColumns)} potential feature columns")
    
    processedColumns = []
    
    for col in featureColumns:
        columnData = dataFrame[col]
        
        if columnData.dtype == 'object' and '[' in str(columnData.iloc[0]):
            logging.info(f"Processing array column: {col}")
            parsedArrays = columnData.apply(_parseArrayString)
            maxLength = max(len(arr) for arr in parsedArrays)

            # Padding with 0 maybe change?
            matrix = np.array([np.pad(arr, (0, maxLength - len(arr))) for arr in parsedArrays])
            processedColumns.append(matrix)
        else:
            numericData = pd.to_numeric(columnData, errors='coerce').fillna(0).values.reshape(-1, 1)
            processedColumns.append(numericData)
    
    if processedColumns:
        featuresMatrix = np.hstack(processedColumns)
        logging.info(f"Created feature matrix with shape: {featuresMatrix.shape}")
    else:
        raise ValueError("No valid features found")
    
    targetData = dataFrame[targetColumn].values
    logging.info(f"Target classes: {np.unique(targetData)}")
    return featuresMatrix, targetData

def _parseArrayString(arrayStr):
    try:
        return ast.literal_eval(str(arrayStr))
    except (ValueError, SyntaxError):
        return np.fromstring(str(arrayStr).strip(' []'), sep=' ')

def createClassifier(classifierType):
    if classifierType == 'rf':
        return RandomForestTrainer()
    else:
        raise ValueError(f"Classifier type '{classifierType}' not supported. Available: rf")

def main():
    parser = argparse.ArgumentParser(description='Generic classifier launcher')
    parser.add_argument('--csv', required=True, help='CSV file path')
    parser.add_argument('--target', default='category', help='Target column name')
    parser.add_argument('--classifier', default='rf', choices=['rf'], help='Classifier type')
    parser.add_argument('--testSize', type=float, default=0.2, help='Test split size')
    parser.add_argument('--modelPath', default='../../models/model.joblib', help='Model save path')
    
    args = parser.parse_args()
    
    featuresMatrix, targetValues = loadCsvData(args.csv, args.target)
    
    xTrain, xTest, yTrain, yTest = train_test_split(
        featuresMatrix, targetValues, 
        test_size=args.testSize, 
        random_state=42
    )
    
    logging.info(f"Dataset split - Train: {xTrain.shape[0]} samples, Test: {xTest.shape[0]} samples")
    logging.info(f"Train classes: {np.unique(yTrain, return_counts=True)}")
    logging.info(f"Test classes: {np.unique(yTest, return_counts=True)}")
    
    logging.info(f"Creating {args.classifier} classifier")
    trainer = createClassifier(args.classifier)
    
    trainer.train(xTrain, yTrain)
    trainer.evaluate(xTest, yTest)
    trainer.saveModel(args.modelPath)

if __name__ == "__main__":
    main()