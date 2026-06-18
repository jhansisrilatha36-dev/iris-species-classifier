"""
Iris Flower Classification - Model Training and Evaluation Pipeline
Author: AI Assistant
Description:
    This script loads the standard Iris dataset, performs preprocessing,
    trains five different machine learning classifiers, evaluates them,
    saves the models and scaler, and generates high-quality visualizations.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Set style for professional-looking plots
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

# Import machine learning classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def create_directories():
    """Ensure that all target directories exist."""
    for folder in ['dataset', 'models', 'images']:
        os.makedirs(folder, exist_ok=True)
    print("Directories verified: dataset/, models/, images/")

def load_and_prepare_data():
    """
    Loads Iris dataset from sklearn, saves raw data to CSV,
    and returns features (X), targets (y), and metadata.
    """
    print("Loading Iris dataset from Scikit-Learn...")
    iris = load_iris()
    
    # Create a DataFrame for visual inspection and saving
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    # Map numerical targets to scientific species names
    species_map = {i: name for i, name in enumerate(iris.target_names)}
    df['species'] = df['target'].map(species_map)
    
    # Save the raw dataset
    csv_path = os.path.join('dataset', 'iris.csv')
    df.to_csv(csv_path, index=False)
    print(f"Raw dataset successfully saved to {csv_path}")
    print(f"Dataset shape: {df.shape}")
    print("\nDataset Summary Statistics:")
    print(df.describe().to_string())
    
    # Separate features and target
    X = df[iris.feature_names]
    y = df['target']
    
    return X, y, iris.feature_names, iris.target_names, df

def generate_eda_plots(df, feature_names):
    """Generates and saves exploratory data analysis plots."""
    print("Generating Exploratory Data Analysis (EDA) plots...")
    
    # 1. Feature Distribution Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    colors = ['#4A90E2', '#50E3C2', '#F5A623', '#D0021B']
    
    for i, feature in enumerate(feature_names):
        sns.histplot(
            data=df, 
            x=feature, 
            hue='species', 
            kde=True, 
            ax=axes[i], 
            element='step', 
            palette='muted',
            alpha=0.6
        )
        axes[i].set_title(f'Distribution of {feature.title()}')
        axes[i].set_xlabel('Measurement (cm)')
    
    plt.tight_layout()
    dist_path = os.path.join('images', 'feature_distribution.png')
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f"Saved: {dist_path}")
    
    # 2. Correlation Heatmap
    plt.figure(figsize=(8, 6))
    # Correlation is calculated only on features
    corr_matrix = df[feature_names].corr()
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='coolwarm', 
        vmin=-1, 
        vmax=1, 
        fmt=".3f", 
        linewidths=0.5,
        square=True
    )
    plt.title('Feature Correlation Matrix', pad=15)
    plt.tight_layout()
    heatmap_path = os.path.join('images', 'correlation_heatmap.png')
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Saved: {heatmap_path}")
    
    # 3. Pair Plot
    # Seaborn pairplot returns a Grid object and handles saving natively
    pair_grid = sns.pairplot(
        df.drop(columns=['target']), 
        hue='species', 
        palette='Set2', 
        diag_kind='kde',
        markers=["o", "s", "D"]
    )
    pair_grid.fig.suptitle('Pairwise Feature Relationships by Species', y=1.02)
    pair_grid.savefig(os.path.join('images', 'pair_plot.png'), dpi=300)
    plt.close()
    print("Saved: images/pair_plot.png")

def train_and_evaluate_models(X, y, target_names):
    """
    Splits the data, standardizes it, trains 5 models,
    evaluates them, and saves the metrics and files.
    """
    # 1. Train-test split (80-20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 2. Standardize features
    print("Fitting StandardScaler and transforming features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler object for online prediction pipeline
    scaler_path = os.path.join('models', 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # Define the 5 classifiers
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Support Vector Machine': SVC(probability=True, random_state=42)
    }
    
    results = {}
    confusion_matrices = {}
    
    print("\nTraining and evaluating models:")
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        
        results[name] = {
            'Accuracy': float(acc),
            'Precision': float(precision),
            'Recall': float(recall),
            'F1-Score': float(f1)
        }
        
        # Store confusion matrix
        confusion_matrices[name] = confusion_matrix(y_test, y_pred)
        
        # Save model file
        filename_map = {
            'Logistic Regression': 'logistic_regression.joblib',
            'K-Nearest Neighbors': 'knn.joblib',
            'Decision Tree': 'decision_tree.joblib',
            'Random Forest': 'random_forest.joblib',
            'Support Vector Machine': 'svm.joblib'
        }
        model_filename = filename_map[name]
        model_path = os.path.join('models', model_filename)
        joblib.dump(model, model_path)
        print(f"    Saved model file to {model_path}")
    
    # Save metrics JSON for Streamlit usage
    metrics_path = os.path.join('models', 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Evaluation metrics saved to {metrics_path}")
    
    # Output metrics as a neat markdown table to console
    print("\n=== Model Performance Summary ===")
    metrics_df = pd.DataFrame(results).T
    print(metrics_df.to_markdown())
    
    # Identify the best model based on F1-Score
    best_model_name = metrics_df['F1-Score'].idxmax()
    print(f"\nBest Performing Model: {best_model_name} (F1-Score: {results[best_model_name]['F1-Score']:.4f})")
    
    # 3. Create Accuracy Comparison Plot
    generate_comparison_plots(results, confusion_matrices, target_names)

def generate_comparison_plots(results, confusion_matrices, target_names):
    """Generates charts for comparing accuracy and plotting confusion matrices."""
    # Accuracy Comparison Bar Chart
    names = list(results.keys())
    accuracies = [results[name]['Accuracy'] * 100 for name in names]
    
    plt.figure(figsize=(10, 6))
    colors = ['#4A90E2', '#50E3C2', '#FF6B6B', '#F5A623', '#9B51E0']
    bars = plt.bar(names, accuracies, color=colors, width=0.6, edgecolor='none', alpha=0.95)
    
    # Styling details
    plt.title('Comparison of Model Accuracy (%)', pad=20)
    plt.ylabel('Accuracy (%)')
    plt.ylim(80, 105)  # Focus on the relevant range
    
    # Adding values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.0, 
            yval + 0.5, 
            f"{yval:.2f}%", 
            ha='center', 
            va='bottom', 
            fontweight='bold'
        )
        
    plt.tight_layout()
    comparison_path = os.path.join('images', 'model_comparison.png')
    plt.savefig(comparison_path, dpi=300)
    plt.close()
    print(f"Saved: {comparison_path}")
    
    # Confusion Matrices Grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (name, cm) in enumerate(confusion_matrices.items()):
        sns.heatmap(
            cm, 
            annot=True, 
            fmt="d", 
            cmap="Blues", 
            xticklabels=target_names, 
            yticklabels=target_names,
            cbar=False,
            ax=axes[i],
            annot_kws={"size": 12, "weight": "bold"}
        )
        axes[i].set_title(f'{name} Confusion Matrix')
        axes[i].set_ylabel('True Label')
        axes[i].set_xlabel('Predicted Label')
    
    # Turn off the last unused subplot in the 2x3 grid
    axes[-1].axis('off')
    
    plt.tight_layout()
    cm_path = os.path.join('images', 'confusion_matrices.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved: {cm_path}")

def main():
    print("="*50)
    print("Starting Iris Flower Classification Training Pipeline")
    print("="*50)
    
    create_directories()
    X, y, feature_names, target_names, df = load_and_prepare_data()
    generate_eda_plots(df, feature_names)
    train_and_evaluate_models(X, y, target_names)
    
    print("\n" + "="*50)
    print("Training Pipeline Executed Successfully!")
    print("="*50)

if __name__ == '__main__':
    main()
