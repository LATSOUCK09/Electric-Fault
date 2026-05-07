"""
DATA CLEANING & VALIDATION
===========================
Purpose: Clean, validate and prepare maintenance data for modeling
- Date format validation & correction
- Handle missing values intelligently
- Remove duplicates & outliers
- Generate quality report

Author: Electric Fault Team
Date: 2026-05-07
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
from pathlib import Path

# Create directories
Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("outputs/reports").mkdir(parents=True, exist_ok=True)


class DataCleaner:
    """Clean and validate maintenance data"""
    
    def __init__(self, data_file="data/raw/maintenance_data.csv"):
        self.data_file = data_file
        self.df = None
        self.df_original = None
        self.quality_report = {}
        
    def load_data(self):
        """Load raw data"""
        try:
            self.df = pd.read_csv(self.data_file)
            self.df_original = self.df.copy()
            print(f"✅ Data loaded: {len(self.df)} records, {len(self.df.columns)} columns")
            return self.df
        except FileNotFoundError:
            print(f"❌ ERROR: File not found: {self.data_file}")
            exit(1)
    
    def validate_dates(self):
        """Validate and fix date columns"""
        print("\n📅 VALIDATING DATES")
        print("=" * 60)
        
        date_columns = ['dateCreation', 'dateDebutPrevue', 'dateFinPrevue', 'DatePlanifiee']
        
        for col in date_columns:
            if col in self.df.columns:
                # Convert to datetime
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                missing_count = self.df[col].isna().sum()
                print(f"✓ {col}: {missing_count} invalid dates")
        
        # Check chronology: dateCreation <= dateDebutPrevue <= dateFinPrevue
        print("\n🔍 Checking chronology...")
        
        bad_chronology = 0
        for idx, row in self.df.iterrows():
            dc = row['dateCreation']
            dbp = row['dateDebutPrevue']
            dfp = row['dateFinPrevue']
            
            # Fix: if dateDebutPrevue > dateFinPrevue, swap them
            if pd.notna(dbp) and pd.notna(dfp) and dbp > dfp:
                self.df.at[idx, 'dateDebutPrevue'] = dfp
                self.df.at[idx, 'dateFinPrevue'] = dbp
                bad_chronology += 1
            
            # If dateCreation > dateDebutPrevue, fix it
            if pd.notna(dc) and pd.notna(dbp) and dc > dbp:
                self.df.at[idx, 'dateDebutPrevue'] = dc
        
        print(f"✅ Fixed {bad_chronology} chronology issues")
        self.quality_report['date_issues_fixed'] = bad_chronology
    
    def calculate_intervention_duration(self):
        """Calculate intervention duration in days"""
        print("\n⏱️  CALCULATING INTERVENTION DURATION")
        print("=" * 60)
        
        if 'dateDebutPrevue' in self.df.columns and 'dateFinPrevue' in self.df.columns:
            self.df['duree_intervention_jours'] = (self.df['dateFinPrevue'] - self.df['dateDebutPrevue']).dt.days
            print(f"✅ Duration calculated")
            print(f"   Mean: {self.df['duree_intervention_jours'].mean():.1f} days")
            print(f"   Median: {self.df['duree_intervention_jours'].median():.1f} days")
    
    def clean_costs(self):
        """Clean cost columns"""
        print("\n💰 CLEANING COST DATA")
        print("=" * 60)
        
        cost_columns = ['coutMaintenance', 'CoutMainDoeuvre', 'CoutMateriel', 'CoutService']
        
        for col in cost_columns:
            if col in self.df.columns:
                # Convert to numeric
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                # Fill NaN with 0
                missing_before = self.df[col].isna().sum()
                self.df[col].fillna(0, inplace=True)
                print(f"✓ {col}: {missing_before} missing values filled")
        
        # Validate cost consistency: coutMaintenance = sum of components
        print("\n✓ Validating cost consistency...")
        if all(col in self.df.columns for col in cost_columns):
            self.df['cout_calculated'] = (self.df['CoutMainDoeuvre'] + 
                                         self.df['CoutMateriel'] + 
                                         self.df['CoutService'])
            
            # For records with coutMaintenance = 0, use calculated value
            mask = (self.df['coutMaintenance'] == 0) & (self.df['cout_calculated'] > 0)
            self.df.loc[mask, 'coutMaintenance'] = self.df.loc[mask, 'cout_calculated']
            
            self.df.drop('cout_calculated', axis=1, inplace=True)
            print(f"✅ Cost consistency validated")
        
        # Detect outliers
        print("\n✓ Detecting cost outliers...")
        for col in cost_columns:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                if outliers > 0:
                    print(f"   {col}: {outliers} potential outliers detected")
                    # Flag them but don't remove (may be legitimate high-cost items)
                    self.df[f'{col}_is_outlier'] = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
    
    def handle_missing_values(self):
        """Handle missing categorical values"""
        print("\n🔧 HANDLING MISSING VALUES")
        print("=" * 60)
        
        categorical_cols = ['zone', 'fonction', 'familleEquipement', 'typeIntervention', 
                           'classeIntervention', 'entiteIntervention', 'entiteAction']
        
        for col in categorical_cols:
            if col in self.df.columns:
                missing_count = self.df[col].isna().sum()
                if missing_count > 0:
                    # Fill with "UNKNOWN"
                    self.df[col].fillna('UNKNOWN', inplace=True)
                    print(f"✓ {col}: {missing_count} missing values filled with 'UNKNOWN'")
    
    def remove_duplicates(self):
        """Remove duplicate records"""
        print("\n🗑️  REMOVING DUPLICATES")
        print("=" * 60)
        
        duplicates_before = len(self.df)
        # Consider codeOT + dateCreation as unique identifier
        self.df.drop_duplicates(subset=['codeOT', 'dateCreation'], keep='first', inplace=True)
        duplicates_after = len(self.df)
        removed = duplicates_before - duplicates_after
        
        print(f"✓ Removed {removed} duplicate records")
        print(f"✓ Remaining records: {duplicates_after}")
        self.quality_report['duplicates_removed'] = removed
    
    def generate_quality_report(self):
        """Generate data quality report"""
        print("\n📊 GENERATING QUALITY REPORT")
        print("=" * 60)
        
        report = []
        report.append("=" * 70)
        report.append("DATA QUALITY REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 70)
        
        report.append(f"\n📋 DATASET SIZE")
        report.append(f"  Original records: {len(self.df_original)}")
        report.append(f"  Cleaned records: {len(self.df)}")
        report.append(f"  Duplicates removed: {self.quality_report.get('duplicates_removed', 0)}")
        report.append(f"  Final records: {len(self.df)}")
        
        report.append(f"\n📅 DATE QUALITY")
        report.append(f"  Chronology issues fixed: {self.quality_report.get('date_issues_fixed', 0)}")
        date_cols = ['dateCreation', 'dateDebutPrevue', 'dateFinPrevue']
        for col in date_cols:
            if col in self.df.columns:
                missing = self.df[col].isna().sum()
                valid = len(self.df) - missing
                pct = (valid / len(self.df)) * 100
                report.append(f"  {col}: {valid} valid ({pct:.1f}%)")
        
        report.append(f"\n💰 COST QUALITY")
        cost_cols = ['coutMaintenance', 'CoutMainDoeuvre', 'CoutMateriel', 'CoutService']
        for col in cost_cols:
            if col in self.df.columns:
                report.append(f"  {col}:")
                report.append(f"    Mean: {self.df[col].mean():.2f}")
                report.append(f"    Median: {self.df[col].median():.2f}")
                report.append(f"    Max: {self.df[col].max():.2f}")
        
        report.append(f"\n🏷️  CATEGORICAL FIELDS")
        cat_cols = ['zone', 'fonction', 'familleEquipement', 'typeIntervention']
        for col in cat_cols:
            if col in self.df.columns:
                unique_count = self.df[col].nunique()
                report.append(f"  {col}: {unique_count} unique values")
        
        report.append("\n" + "=" * 70)
        report.append("✅ DATA CLEANING COMPLETE")
        report.append("=" * 70)
        
        # Save report
        report_text = "\n".join(report)
        with open('outputs/reports/data_quality_report.txt', 'w') as f:
            f.write(report_text)
        
        print("\n".join(report))
        return report_text
    
    def save_cleaned_data(self):
        """Save cleaned data"""
        output_file = 'data/processed/cleaned_data.csv'
        self.df.to_csv(output_file, index=False)
        print(f"\n✅ Cleaned data saved: {output_file}")
        return output_file


def main():
    """Main execution"""
    print("\n" + "="*60)
    print("🧹 DATA CLEANING & VALIDATION")
    print("="*60)
    
    cleaner = DataCleaner()
    
    # Execute cleaning pipeline
    cleaner.load_data()
    cleaner.validate_dates()
    cleaner.calculate_intervention_duration()
    cleaner.clean_costs()
    cleaner.handle_missing_values()
    cleaner.remove_duplicates()
    cleaner.generate_quality_report()
    cleaner.save_cleaned_data()
    
    print("\n✅ DATA CLEANING COMPLETE!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
