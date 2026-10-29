# Statistical Analysis of Urban Traffic and Travel Time

A complete, production-grade Statistical Inference Lab project web application built with Python and Streamlit. This application covers all 10 core practical modules required for college curriculum, big data analytics, and statistical inference courses.

---

## 🌟 Modules Covered

1. **Web Scraping and CSV Creation:** Scraping public municipal traffic indices with error handling (HTTP 403, 404, table fallbacks).
2. **Data Cleaning & Exploratory Data Analysis (EDA):** Mean, median, mode, variance, standard deviation, quartiles, IQR, Tukey outlier detection, histograms, box plots, and frequency charts.
3. **Probability Distributions & Sampling:** Simple Random Sampling (SRS), Systematic Sampling, and Stratified Sampling with CSV export.
4. **Central Limit Theorem (CLT):** Simulation of sampling distributions across sample sizes ($n=5$ to $120$) and comparison between theoretical vs observed standard error.
5. **Confidence Intervals & Hypothesis Testing:** One-sample Z-test and t-test, assumptions, hypothesis formulation ($H_0$ vs $H_1$), p-value computation, and decision boundaries.
6. **Chi-Square Test of Independence:** Contingency tables, observed vs expected counts, degrees of freedom, $chi^2$ statistic, and association tests.
7. **Covariance & Correlation:** Covariance matrix, Pearson correlation heatmap, strength and direction interpretation.
8. **Correlation Hypothesis Testing:** Pearson linear ($r$) vs Spearman rank ($ho$) non-parametric test with exact p-values.
9. **Two-Way ANOVA:** Factorial ANOVA testing the effects of Peak Hours and Weather Conditions on Travel Time with interaction analysis.
10. **Multiple Linear Regression:** Ordinary Least Squares ($R^2$, RMSE, MAE, feature coefficients) and live prediction calculator.

---

## 🚀 Running Locally

### 1. Clone or Download the repository
```bash
git clone https://github.com/YOUR_USERNAME/urban-traffic-statistical-analysis.git
cd urban-traffic-statistical-analysis
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS / Linux:
source venv/bin/activate
```

### 3. Install required libraries
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application
```bash
streamlit run app.py
```

The app will launch in your browser at `http://localhost:8501`.

---

## ☁️ Deploying on Streamlit Community Cloud (Free)

1. Create a public repository on GitHub (e.g., `urban-traffic-statistical-analysis`).
2. Upload the following 4 files:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `traffic_travel_time_dataset.csv`
3. Visit [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
4. Click **New app** -> Select your repository, branch (`main`), and set the main file path to `app.py`.
5. Click **Deploy!** Your app will be live with a permanent public URL in ~2 minutes.

---

## 📊 Dataset Schema

- `Trip_ID`: Unique trip observation identifier.
- `Distance_KM`: Distance of the trip in kilometers (1.0 - 50.0 km).
- `Travel_Time_Min`: Travel duration in minutes (target continuous variable).
- `Average_Speed_KMH`: Vehicle speed in km/h.
- `Vehicle_Count`: Traffic volume density (vehicles/hour).
- `Peak_Hour`: 'Morning Peak', 'Evening Peak', 'Off-Peak'.
- `Weather_Condition`: 'Clear', 'Rainy', 'Foggy', 'Snowy'.
- `Road_Type`: 'Highway', 'Arterial', 'Urban Local', 'Collector'.
- `Congestion_Level`: 'Low', 'Moderate', 'High', 'Severe'.
- `Accident_Reported`: 'Yes' or 'No'.
- `Day_Of_Week`: Day of observation.
- `City_Zone`: Municipal zone in the metropolitan region.
