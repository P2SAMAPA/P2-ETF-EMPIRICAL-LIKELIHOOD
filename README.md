# Empirical Likelihood Engine

Nonparametric likelihood ratio for the mean of ETF returns (Owen, 1988). Provides exact confidence intervals without distributional assumptions. Uses Wilks’ theorem: \(-2\log R(\mu) \sim \chi^2(1)\). The score is the lower bound of a 90% confidence interval for the mean – a conservative estimate of expected return.

- **Algorithm:** Empirical likelihood ratio, solved via Lagrange multiplier root‑finding
- **Confidence level:** 90% (configurable)
- **Windows:** 63, 252, 504, 1008, 2016 days (best per ETF)
- **Output:** top 3 ETFs per universe by EL lower bound

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
