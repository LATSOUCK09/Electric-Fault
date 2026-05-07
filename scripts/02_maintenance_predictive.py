"""
PREDICTIVE MAINTENANCE MODEL
=============================
Purpose: Predict equipment failures 30/60/90 days in advance
- Feature engineering for maintenance history
- XGBoost classification model
- Generate maintenance alerts for high-risk equipment
- Provide actionable insights for maintenance teams

Author: Electric Fault Team
Date: 2026-05-07
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from pathlib import Path
from datetime import datetime, timedelta

# Create directories
Path("outputs/reports").mkdir(parents=True, exist_ok=True)
Path("outputs/dashboards").mkdir(parents=True, exist_ok=True)
Path("outputs/models").mkdir(parents=True, exist_ok=True)


class PredictiveMaintenanceModel:
    """Predictive maintenance model using XGBoost"""
    
    def __init__(self, data_file="data/processed/cleaned_data.csv"):
        self.data_file = data_file
        self.df = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.target = None
        
    def load_data(self):
        """Load cleaned data"""
        try:
            self.df = pd.read_csv(self.data_file)
            print(f"✅ Data loaded: {len(self.df)} records")
            return self.df
        except FileNotFoundError:
            print(f"❌ ERROR: File not found: {self.data_file}")
            print("   Run 01_data_cleaning.py first!")
            exit(1)
    
    def create_target(self, forecast_days=30):
        """
        Create target variable: will this equipment fail in next N days?
        
        Logic: Equipment fails if it has high maintenance frequency or long periods without maintenance
        """
        print(f"\n🎯 CREATING TARGET VARIABLE (forecast: {forecast_days} days)")
        print("=" * 60)
        
        # Group by equipment to analyze patterns
        equipment_stats = self.df.groupby('codeEquipement').agg({
            'codeOT': 'count',  # Number of interventions
            'coutMaintenance': ['sum', 'mean'],
            'dateCreation': 'min',
            'dateFinPrevue': 'max'
        }).reset_index()
        equipment_stats.columns = ['codeEquipement', 'num_interventions', 
                                  'total_cost', 'avg_cost', 'first_date', 'last_date']
        
        # Calculate intervention frequency (interventions per year)
        equipment_stats['days_active'] = (equipment_stats['last_date'] - equipment_stats['first_date']).dt.days + 1
        equipment_stats['intervention_freq'] = (equipment_stats['num_interventions'] / 
                                               (equipment_stats['days_active'] / 365.25))
        
        # Define failure risk based on:
        # - High intervention frequency (> median)
        # - High average cost (> 75th percentile)
        freq_threshold = equipment_stats['intervention_freq'].median()
        cost_threshold = equipment_stats['avg_cost'].quantile(0.75)
        
        equipment_stats['failure_risk'] = (
            (equipment_stats['intervention_freq'] > freq_threshold) &
            (equipment_stats['avg_cost'] > cost_threshold)
        ).astype(int)
        
        # Merge back to main dataframe
        self.df = self.df.merge(equipment_stats[['codeEquipement', 'failure_risk']], 
                               on='codeEquipement', how='left')
        
        self.target = 'failure_risk'
        print(f"✅ Target created: {self.df[self.target].sum()} high-risk equipments detected")
        print(f"   Frequency threshold: {freq_threshold:.2f} interventions/year")
        print(f"   Cost threshold: {cost_threshold:.0f}")
    
    def create_features(self):
        """Create features for the model"""
        print("\n⚙️  CREATING FEATURES")
        print("=" * 60)
        
        # Numeric features
        numeric_features = []
        
        # 1. Equipment age (in years)
        if 'dateCreation' in self.df.columns:
            self.df['equipment_age_years'] = (datetime.now() - pd.to_datetime(self.df['dateCreation'])).dt.days / 365.25
            numeric_features.append('equipment_age_years')
        
        # 2. Days since last maintenance
        if 'dateFinPrevue' in self.df.columns:
            self.df['days_since_last_maint'] = (datetime.now() - pd.to_datetime(self.df['dateFinPrevue'])).dt.days
            numeric_features.append('days_since_last_maint')
        
        # 3. Intervention duration
        if 'duree_intervention_jours' in self.df.columns:
            numeric_features.append('duree_intervention_jours')
        
        # 4. Cost metrics
        cost_features = ['coutMaintenance', 'CoutMainDoeuvre', 'CoutMateriel', 'CoutService']
        for f in cost_features:
            if f in self.df.columns:
                numeric_features.append(f)
        
        # 5. Maintenance frequency per equipment (count OT by equipment)
        eq_intervention_count = self.df.groupby('codeEquipement').size().reset_index(name='eq_intervention_count')
        self.df = self.df.merge(eq_intervention_count, on='codeEquipement', how='left')
        numeric_features.append('eq_intervention_count')
        
        # Categorical features (to encode)
        categorical_features = []
        cat_cols = ['zone', 'fonction', 'familleEquipement', 'typeIntervention', 'classeIntervention']
        for col in cat_cols:
            if col in self.df.columns:
                categorical_features.append(col)
        
        # Encode categorical features
        for col in categorical_features:
            le = LabelEncoder()
            self.df[col + '_encoded'] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
            numeric_features.append(col + '_encoded')
        
        # Remove NaN from numeric features
        for f in numeric_features:
            if f in self.df.columns:
                self.df[f].fillna(self.df[f].mean(), inplace=True)
        
        self.feature_columns = numeric_features
        print(f"✅ Features created: {len(self.feature_columns)}")
        for i, f in enumerate(self.feature_columns, 1):
            print(f"   {i}. {f}")
    
    def train_model(self, test_size=0.2, random_state=42):
        """Train XGBoost model"""
        print("\n🤖 TRAINING XGBOOST MODEL")
        print("=" * 60)
        
        # Prepare data
        X = self.df[self.feature_columns].values
        y = self.df[self.target].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"✅ Train set: {len(X_train)} samples")
        print(f"✅ Test set: {len(X_test)} samples")
        print(f"   Class distribution - Train: {np.mean(y_train):.2%} positive")
        print(f"   Class distribution - Test: {np.mean(y_test):.2%} positive")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        print("\n⏳ Training model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        print("\n📊 MODEL PERFORMANCE")
        print("-" * 60)
        print(classification_report(y_test, y_pred, target_names=['Normal', 'High Risk']))
        print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.X_test_scaled = X_test_scaled
    
    def generate_alerts(self, risk_threshold=0.6):
        """Generate maintenance alerts for high-risk equipment"""
        print("\n🚨 GENERATING MAINTENANCE ALERTS")
        print("=" * 60)
        
        # Get predictions for all data
        X_all = self.df[self.feature_columns].values
        X_all_scaled = self.scaler.transform(X_all)
        risk_scores = self.model.predict_proba(X_all_scaled)[:, 1]
        
        self.df['risk_score'] = risk_scores
        self.df['risk_level'] = pd.cut(risk_scores, 
                                       bins=[0, 0.33, 0.66, 1.0], 
                                       labels=['LOW', 'MEDIUM', 'HIGH'])
        
        # Create alerts dataframe
        alerts = self.df[self.df['risk_score'] > risk_threshold].copy()
        alerts = alerts[['codeOT', 'codeEquipement', 'zone', 'familleEquipement', 
                        'typeIntervention', 'risk_score', 'risk_level', 
                        'coutMaintenance', 'dateFinPrevue']].drop_duplicates()
        alerts = alerts.sort_values('risk_score', ascending=False)
        
        print(f"✅ Generated {len(alerts)} alerts (risk > {risk_threshold})")
        print(f"\n   HIGH RISK: {len(alerts[alerts['risk_level']=='HIGH'])}")
        print(f"   MEDIUM RISK: {len(alerts[alerts['risk_level']=='MEDIUM'])}")
        print(f"   LOW RISK: {len(alerts[alerts['risk_level']=='LOW'])}")
        
        # Save alerts
        alerts.to_csv('outputs/reports/maintenance_alerts.csv', index=False)
        print(f"\n✅ Alerts saved: outputs/reports/maintenance_alerts.csv")
        
        # Display top 10 high-risk equipment
        print("\n📋 TOP 10 HIGH-RISK EQUIPMENT:")
        print("-" * 60)
        top_alerts = alerts.head(10)
        for idx, row in top_alerts.iterrows():
            print(f"{row['codeEquipement']:12} | Risk: {row['risk_score']:.2%} | Zone: {row['zone']:8} | Family: {row['familleEquipement']}")
    
    def save_model(self):
        """Save trained model"""
        model_path = 'outputs/models/maintenance_model.pkl'
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"\n✅ Model saved: {model_path}")
        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
    
    def create_visualizations(self):
        """Create feature importance and model performance plots"""
        print("\n📈 CREATING VISUALIZATIONS")
        print("=" * 60)
        
        # Feature importance
        plt.figure(figsize=(12, 6))
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)
        
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Importance Score')
        plt.title('Top 15 Feature Importance - Predictive Maintenance Model')
        plt.tight_layout()
        plt.savefig('outputs/dashboards/feature_importance.png', dpi=300)
        print("✅ Feature importance plot saved")
        
        # ROC Curve
        plt.figure(figsize=(10, 6))
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        auc = roc_auc_score(self.y_test, self.y_pred_proba)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Equipment Failure Prediction')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/dashboards/roc_curve.png', dpi=300)
        print("✅ ROC curve saved")
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'High Risk'],
                   yticklabels=['Normal', 'High Risk'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix - Equipment Failure Prediction')
        plt.tight_layout()
        plt.savefig('outputs/dashboards/confusion_matrix.png', dpi=300)
        print("✅ Confusion matrix saved")
        
        plt.close('all')


def main():
    """Main execution"""
    print("\n" + "="*60)
    print("🤖 PREDICTIVE MAINTENANCE MODEL - XGBOOST")
    print("="*60)
    
    model = PredictiveMaintenanceModel()
    
    # Execute pipeline
    model.load_data()
    model.create_target(forecast_days=30)
    model.create_features()
    model.train_model()
    model.generate_alerts(risk_threshold=0.6)
    model.save_model()
    model.create_visualizations()
    
    print("\n✅ PREDICTIVE MAINTENANCE MODEL COMPLETE!")
    print("📊 Outputs saved to outputs/ directory")
    print("📋 Alerts saved to: outputs/reports/maintenance_alerts.csv")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
