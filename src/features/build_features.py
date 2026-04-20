"""
Feature engineering for Pharma SFE.
Creates doctor-level Analytical Base Table (ABT).
"""

import pandas as pd
import numpy as np
import os


def build_doctor_features(doctors, visits, rx):
    """One row per doctor, all features needed for downstream modeling."""

    visits['visit_date'] = pd.to_datetime(visits['visit_date'])
    visits['visit_month'] = visits['visit_date'].dt.to_period('M').astype(str)
    rx['month'] = pd.to_datetime(rx['month']).dt.to_period('M').astype(str)

    # 1. Rx-based features
    rx_features = rx.groupby('doctor_id').agg(
        total_rx_24m=('our_brand_rx', 'sum'),
        avg_monthly_rx=('our_brand_rx', 'mean'),
        rx_volatility=('our_brand_rx', 'std'),
        avg_our_share=('our_share', 'mean'),
        max_monthly_rx=('our_brand_rx', 'max'),
        months_active=('our_brand_rx', lambda x: (x > 0).sum())
    ).reset_index()

    # 2. Momentum (last 3 months vs previous 3)
    recent_months = sorted(rx['month'].unique())[-6:]
    last_3 = recent_months[-3:]
    prev_3 = recent_months[:3]

    rx_recent = rx[rx['month'].isin(last_3)].groupby('doctor_id')['our_brand_rx'].sum().rename('rx_last_3m')
    rx_prev = rx[rx['month'].isin(prev_3)].groupby('doctor_id')['our_brand_rx'].sum().rename('rx_prev_3m')

    momentum = pd.concat([rx_recent, rx_prev], axis=1).fillna(0).reset_index()
    momentum['momentum_ratio'] = (momentum['rx_last_3m'] + 1) / (momentum['rx_prev_3m'] + 1)
    momentum['is_bloomer'] = (momentum['momentum_ratio'] > 1.2).astype(int)
    momentum['is_decliner'] = (momentum['momentum_ratio'] < 0.8).astype(int)

    # 3. Visit features
    visit_features = visits.groupby('doctor_id').agg(
        total_visits_24m=('visit_id', 'count'),
        avg_visit_duration=('duration_mins', 'mean'),
        total_samples=('samples_left', 'sum'),
        unique_months_visited=('visit_month', 'nunique')
    ).reset_index()
    visit_features['visit_frequency'] = visit_features['total_visits_24m'] / 24

    # 4. Message mix
    message_mix = visits.pivot_table(
        index='doctor_id',
        columns='message_delivered',
        values='visit_id',
        aggfunc='count',
        fill_value=0
    )
    message_mix.columns = [f'msg_{c.lower().replace(" ", "_")}_count' for c in message_mix.columns]
    message_mix = message_mix.reset_index()

    # 5. Coverage rate
    coverage = visits.groupby('doctor_id')['visit_month'].nunique().rename('months_covered').reset_index()
    coverage['coverage_rate'] = coverage['months_covered'] / 24

    # Merge onto doctors
    abt = doctors.copy()
    for df in [rx_features, momentum, visit_features, message_mix, coverage]:
        abt = abt.merge(df, on='doctor_id', how='left')

    abt = abt.fillna(0)

    # 6. Derived ratios
    abt['rx_per_visit'] = abt['total_rx_24m'] / abt['total_visits_24m'].replace(0, 1)
    abt['samples_per_visit'] = abt['total_samples'] / abt['total_visits_24m'].replace(0, 1)

    # 7. Value deciles
    abt['value_decile'] = pd.qcut(
        abt['total_rx_24m'],
        q=10,
        labels=range(1, 11),
        duplicates='drop'
    ).astype(int)
    abt['is_high_value'] = (abt['value_decile'] >= 8).astype(int)

    return abt


def main():
    print("Loading raw data...")
    doctors = pd.read_csv('data/raw/doctors.csv')
    visits = pd.read_csv('data/raw/rep_visits.csv')
    rx = pd.read_csv('data/raw/prescriptions.csv')

    print(f"  Doctors: {len(doctors)}")
    print(f"  Visits: {len(visits)}")
    print(f"  Rx: {len(rx)}")

    print("\nBuilding ABT...")
    abt = build_doctor_features(doctors, visits, rx)

    os.makedirs('data/processed', exist_ok=True)
    abt.to_csv('data/processed/abt.csv', index=False)

    print(f"\nABT shape: {abt.shape}")
    print(f"Saved: data/processed/abt.csv")


if __name__ == '__main__':
    main()