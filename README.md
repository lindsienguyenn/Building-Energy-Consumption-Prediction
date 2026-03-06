#  Buildings Energy Consumption Prediction
## Amazon QuickSight Non-Interactive Dashboard: **[View Dashboard](Energy%20Performance%20Dashboard%20(non-interactive.pdf)**


## 1. Introduction to the Dataset

This project explores what constitutes **basic and useful information for a Sustainability Director** managing a multi-family housing portfolio. It uses a large-scale dataset of building energy usage to understand, benchmark, and predict energy performance across properties.

The dataset contains roughly **100,000 building-year observations** collected over **7 years** across multiple U.S. states. Each row represents a **single building observed in a given year**. Energy performance is measured using **Site Energy Usage Intensity (Site EUI)**, which normalizes annual energy consumption by floor area, enabling fair comparison across buildings of different sizes.

The data includes:

* **Building characteristics** (e.g., floor area, year built, facility type, building class)
* **Location information** (anonymized state, elevation)
* **Weather and climate variables** (temperature statistics, heating and cooling degree days, precipitation, snowfall, wind, and extreme temperature days)

Two datasets are used:

* **train.csv**: 75,757 rows × 64 columns, with observed Site EUI values
* **test.csv**: 9,705 rows × 63 columns, with Site EUI withheld

Only three features are categorical (`State_Factor`, `building_class`, `facility_type`); all other variables are numerical.

---

## 2. Dashboard

An interactive dashboard translates raw data into **portfolio-level insights** for sustainability decision-making. The dashboard focuses on high-level KPIs rather than individual building diagnostics.

### Key Views Include:
* Portfolio Overview: Total and average Site EUI, floor area, property count, and Energy Star Rating
* Total energy consumption and average intensity trends by year and state
* State and building class performance comparison (Commercial vs. Residential)
* Top 1,000 properties by Site EUI for targeted efficiency improvements
* Energy intensity by facility type, identifying highest-consuming categories

**Note**: Due to annual data aggregation, month-over-month trend analysis is not available.

---

## 3. Modeling

A supervised machine learning approach is used to predict **Site Energy Usage Intensity (Site EUI)** based on building attributes, climate conditions, and location factors.

* **Objective**: Estimate expected Site EUI to support benchmarking and prioritization
* **Evaluation Metric**: Root Mean Squared Error (RMSE)

The model is designed for **portfolio-level screening**, not billing or regulatory compliance. It helps identify buildings that consume more energy than expected given their characteristics and climate exposure.

Model outputs include:

* Prediction vs. actual Site EUI plots
* RMSE-based performance evaluation
* Correlation analysis between top features and Site EUI
* Ranking of facility types by predicted energy intensity
* For full methodology, preprocessing steps, model configurations, and results, see the [Project Report](https://github.com/lindsienguyenn/Building-Energy-Consumption-Prediction/blob/lindsienguyenn/hifromlindsie/project_report.pdf).


---

## 4. Conclusion

This project demonstrates how historical building, climate, and energy data can be transformed into a **decision-support tool for Sustainability Directors**.

By combining dashboard-driven insights with machine learning predictions, the analysis:

* Identifies high-energy-use buildings for targeted audits or retrofits
* Separates climate-driven energy demand from structural inefficiency
* Supports data-driven prioritization of sustainability investments

Overall, the project provides a scalable framework for understanding and managing energy performance across a multi-family housing portfolio.

www.kaggle.com/competitions/widsdatathon2022/overview/citation

