
# AI Borrower Affordability & Repayment Analytics

A Dash + scikit-learn prototype for borrower affordability assessment, repayment analytics, default-risk prediction, payment forecasting and loan simulation.

## 1. Open in PyCharm
Open the `loan_affordability_ai` folder as a project.

## 2. Create a virtual environment
In PyCharm: Settings/Preferences -> Project -> Python Interpreter -> Add Interpreter -> Virtualenv.

Or terminal:
`python3 -m venv .venv`

Activate it:
- macOS/Linux: `source .venv/bin/activate`
- Windows: `.venv\\Scripts\\activate`

## 3. Install dependencies
`pip install -r requirements.txt`

## 4. Run the application
Run `app.py` from PyCharm, or:
`python app.py`

Open the address printed by Dash, normally `http://127.0.0.1:8050/`.

## 5. Jupyter analysis
Open the `notebooks` folder in PyCharm/Jupyter and run:
1. 01_data_analysis.ipynb
2. 02_feature_engineering.ipynb
3. 03_machine_learning.ipynb

The project includes a generated `data/borrowers.csv` so it works immediately. The dashboard also supports uploading a compatible CSV.

## Important
This is a decision-support/academic prototype, not a production credit underwriting system. Before real lending use, validate models on representative historical data, monitor bias/fairness, calibrate probabilities, secure personal data, establish credit policy thresholds and comply with applicable lending/data-protection regulations.
