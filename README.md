![American Mortgage Bank Cover](https://github.com/moseskigo/Phase-4-proj/assets/143738914/05acb060-b616-41fb-9171-3d8fe4a69a04)

# Revolutionizing Real Estate Analysis for Enhanced Mortgage Solutions at American Mortgage Bank

![V G and AMB Logo](https://github.com/moseskigo/Phase-4-proj/assets/143738914/e3b7b6fb-58fa-4707-8689-ff913ff04f8b)

Moringa School Data Science Core Program\
Part-Time 05 Cohort\
Phase 4 Project\
Group 4

The Team
1. [Anne Karuga](https://github.com/Annkaruga)
2. [Rodgers Odhiambo](https://github.com/rodgersbunde)
3. [Linet Wambui](https://github.com/lynwangui)
4. [James Kibunja](https://github.com/Jameskibunja)
5. [Moses Kigo](https://github.com/moseskigo)

## Background
American Mortgage Bank, eEstablished at the turn of the 20th century, stands at the forefront of an ever-evolving real estate landscape, where shifting market dynamics and fluctuating trends pose significant challenges to maintaining competitive lending strategies and effective risk assessments. The bank's commitment to innovation and excellence drives its quest for deeper market insights to refine its mortgage offerings and lending practices.

## Challenge
Navigating the complexities of the real estate market requires more than traditional analysis; it demands a sophisticated, data-driven approach to unravel the intricacies of market behaviour. American Mortgage Bank faces the critical task of decoding these nuances to stay ahead in a competitive environment, enhance its risk management protocols, and meet its clientele's diverse needs with innovative mortgage products.

## Objective
AMB targets to acquire unmatched market insights and predictive analysis tools that will allow the bank to fine-tune lending strategies, elevate risk assessment, and innovate their mortgage products in line with emerging market trends.

## Solution
Vita Group proposes a solution involving advanced analytics and machine learning to conduct a data-driven market analysis for AMB. The plan includes using sophisticated models to analyze time series data, predict future trends, and innovate mortgage products based on these insights.

## Implementation
The implementation strategy consists of four phases: data acquisition and preprocessing; analytical modeling; strategy optimization and product innovation; and continuous learning and adaptation. These steps aim to provide AMB with actionable insights from extensive real estate data.

## Metric of Success
For our time series analysis, we have selected the Mean Absolute Percentage Error (MAPE) as the metric of success to assess the viability of the model. MAPE has been chosen due to its ability to provide weighted error values, where errors are divided by the true values. This characteristic is particularly advantageous in handling outliers effectively. In contrast, Root Mean Square Error (RMSE) solely considers the difference between the real and predicted values, which can be misleading if outliers exist in the data.

By utilizing MAPE, we aim to obtain a comprehensive evaluation of the model's performance that accounts for both the magnitude of errors and the relative proportion they represent. This metric offers a more robust assessment and ensures that outliers do not unduly influence the perception of the model's effectiveness in generating accurate predictions.

Therefore, MAPE has been chosen as the preferred metric to gauge the model's performance in our time series analysis. It provides a reliable indication of the model's ability to forecast accurately, considering the potential presence of outliers in the dataset.


## The Dataset
This dataset, sourced from Zillow's comprehensive housing data, [here](https://www.zillow.com/research/data/), encompasses a detailed time series of the Zillow Home Value Index (ZHVI) across various neighborhoods. It provides a granular view of typical home values, tracking market changes and trends from 2000 to 2024, across different regions and housing types. Designed to offer insights into the evolving real estate landscape, this dataset serves as a foundational tool for analyzing long-term value fluctuations, regional growth patterns, and seasonal market dynamics.

Data Type:   ZHVI All Homes (SFR, Condo/Co-op) Time Series, Smoothed, Seasonally Adjusted\
Geography:  Neighborhood

![Zillow](https://github.com/moseskigo/Phase-4-proj/assets/143738914/6270d9df-d76f-4959-9d6c-cc59e2b63d18)

```python
#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf
import statsmodels.api as sm
from prophet import Prophet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
```

##Market Dynamics
![Phase_Four_Project_164_0](https://github.com/moseskigo/Phase-4-proj/assets/143738914/558c6425-8177-49ff-8dd1-3e360ce6f144)
Comparison: Comparing the regions to the U.S. average, we can infer that some regions have outperformed the national trend, particularly RegionID 112345, which shows a remarkable growth surpassing the national average by a considerable margin in recent years.

Economic Implications: Such disparities in regional housing trends can be due to various factors, including local economic conditions, housing supply constraints, policy changes, and demographic shifts. Regions that outperform the average may be hotspots for investment or undergoing significant development, while those that underperform could be facing economic challenges or slower growth.

##Time Series Modelling
**Time Series Decomposition**
![Phase_Four_Project_94_0](https://github.com/moseskigo/Phase-4-proj/assets/143738914/89b23c9e-904f-4edf-82be-493e697e0bbb)

Breaking the non-stationary time series into its three components—trend, seasonality, and residuals—is indeed a helpful approach for investigating the pattern in the past and aiding in the forecasting of future house values.

## Mode Selection
For modelling, we decided to pick the following to iterate over:

**1. Moving Average (MA) Base Model**
**2. Autoregressive (AR) Model**
**3. AutoRegressive Moving Average (ARMA) Model**
**4. Seasonal AutoRegressive Moving Average (SARIMA) Model**
**5. Facebook Prophet**

**Observation**
Among these models, the SARIMA(tuned) model has the lowest AIC value of 3546.484. A lower AIC value indicates a better fit of the model to the data, considering the trade-off between goodness of fit and model complexity.

Therefore, the SARIMA(tuned) model is considered the best model among the ones compared.

## Model Evaluation

![image](https://github.com/moseskigo/Phase-4-proj/assets/143738914/9eb8f74a-42e1-4689-b314-64bc1a87191e)

There is a general upward trend in home prices throughout the timeframe. This suggests that, on average, home prices in the United States have increased over the past two decades.

The graph hints at potential periods of rapid increase followed by periods of stagnation or slight decrease. This could be indicative of housing bubbles and bursts that have occurred throughout history.

![image](https://github.com/moseskigo/Phase-4-proj/assets/143738914/1ee8bdb6-b26b-4958-b4e9-7673e15b2995)

The graph typically shows the trend of home values over a certain period. An upward trend suggests increasing home values, while a downward trend suggests decreasing home values.

There might be seasonal fluctuations in the data, with home values being higher during certain times of the year.


## Market Drivers Analysis

![image](https://github.com/moseskigo/Phase-4-proj/assets/143738914/54db6466-42e4-467f-9015-5fc795ccfd26)

Periods of increasing unemployment rates often coincide with slower growth or declines in housing values, as seen around the 2008 financial crisis and the 2020 pandemic onset. This suggests that economic downturns, reflected by higher unemployment, can dampen housing market activity and prices.
The relationship between mortgage rates and housing values isn't as direct or immediate but changes in mortgage rates can influence housing market dynamics over time. For instance, periods of lower mortgage rates can support increased housing demand and contribute to rising housing values, assuming other conditions are favorable.
These trends underscore the complex interactions between economic indicators and the housing market. Factors like unemployment and mortgage rates are crucial for understanding housing market dynamics, but they operate within a broader economic and regulatory context that also influences outcomes.

# Coclusions

**Market Dynamics:** The real estate market shows long-term growth trends and volatility, with clear regional variations and potential seasonal patterns.

**Predictive Modeling:** Among the models tested, the SARIMA model provided a nuanced understanding of time series behavior, capturing seasonal trends and potential cyclical patterns in the data.

**Economic Indicators:** Unemployment rates and mortgage interest rates suggest a correlate with real estate values, suggesting their relevance as economic indicators that can impact the market.

# Recommendations

**Innovate Mortgage Products**: Innovate mortgage products based on market insights and customer preferences identified through data analysis. Tailor product offerings to meet the changing needs of customers, such as introducing flexible loan options, competitive interest rates, and value-added services that differentiate the mortgage financier in the market.

**Enhance Risk Management**: Strengthen risk management practices by integrating advanced risk assessment models and predictive tools into the decision-making process. By identifying and mitigating risks effectively, the mortgage financier can optimize its loan portfolio, improve credit quality, and enhance overall financial performance.

**Collaborate with Data Scientists**: Foster collaboration between data scientists, domain experts, and business stakeholders to leverage the full potential of data analytics. By working together to interpret findings, generate actionable insights, and drive strategic initiatives, the mortgage financier can maximize the value derived from data-driven analysis.

**Monitor Market Drivers**: Stay informed about key market drivers, regional trends, and price volatility to anticipate market shifts and capitalize on emerging opportunities. Regularly analyze market indicators, economic factors, and industry developments to make informed decisions that align with the dynamic real estate landscape.

**Continuous Learning and Adaptation**: Establish a culture of continuous learning and adaptation within the organization to stay abreast of evolving market dynamics. Regularly update models, refine strategies, and monitor key performance indicators to ensure agility and responsiveness to market shifts.

