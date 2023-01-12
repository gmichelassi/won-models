from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import shuffle

import numpy as np
import pandas as pd

FILE_NAME = './data.csv'
RANDOM_STATE = 10

RAW_DATA_UNUSED_COLUMNS = ['company_id', 'company_created_at', 'company_created_at_day', 'first_employee_invitation_created_at', 'first_employee_invitation_created_at_day', 'first_boss_invitation_created_at', 'first_boss_invitation_created_at_day', 'first_employment_accepted_invitation_created_at', 'first_employment_accepted_invitation_created_at_day', 'first_work_schedule_created_at', 'first_work_schedule_created_at_day', 'vacations_count', 'first_employment_accepted_invitation_created_at_days_count', 'reports_count']
HIGHLY_CORRELATED_COLUMNS = ['first_work_schedule_created_at_days_count']
COLUMNS_TO_REMOVE = [*RAW_DATA_UNUSED_COLUMNS, *HIGHLY_CORRELATED_COLUMNS]

UNDERSAMPLER = RandomUnderSampler(random_state=RANDOM_STATE)


def load_data() -> tuple[pd.DataFrame, np.ndarray]:
    original_data = pd.read_csv(FILE_NAME)
    data: pd.DataFrame = original_data.drop(COLUMNS_TO_REMOVE, axis=1)

    data = data.query('timetrackings_count < 5000')

    x = data.loc[:, data.columns != 'won?']
    y = ['won' if current_label == 1 else 'lost' for current_label in data['won?'].to_numpy()]

    return x, data['won?'].to_numpy()


def preprocess_data(x: pd.DataFrame, y: np.ndarray, balance_classes: bool = True) -> tuple[np.ndarray, np.ndarray]:
    features = x.to_numpy()
    labels = y

    if balance_classes:
        features, labels = UNDERSAMPLER.fit_resample(features, labels)

    features, labels = shuffle(features, labels)

    return features, labels
