import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Superstore!!!", page_icon=":bar_chart:",layout="wide")

st.title(" :bar_chart: CO2 Emissions ")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)

fl = st.file_uploader(":file_folder: Upload a file",type=(["csv","txt","xlsx","xls"]))
if fl is not None:
    filename = fl.name
    st.write("Using uploaded file: " + filename)
    # If it's a CSV file
    if filename.lower().endswith('.csv'):
        df = pd.read_csv(fl, encoding="ISO-8859-1")
    # Add more conditions here if you want to handle other file types like 'txt', 'xlsx', 'xls'
    
    # Your processing and plotting code here
else:
    # Display a message prompting the user to upload data
    st.warning('Please upload a data file.')
    st.stop()
col1, col2 = st.columns((2))
df['date'] = pd.to_datetime(df['date'], format="%d/%m/%Y")
df['year'] = df['date'].dt.year


# Getting the min and max date 
startDate = pd.to_datetime(df["date"]).min()
endDate = pd.to_datetime(df["date"]).max()

with col1:
    date1 = pd.to_datetime(st.date_input("Start Date", startDate))

with col2:
    date2 = pd.to_datetime(st.date_input("End Date", endDate))

df = df[(df["date"] >= date1) & (df["date"] <= date2)].copy()
st.sidebar.header("Choose your filter: ")
# Create for Sector
region = st.sidebar.multiselect("Pick your Sector", df["sector"].unique())
if not region:
    df2 = df.copy()
else:
    df2 = df[df["sector"].isin(region)]
# Create for State
state = st.sidebar.multiselect("Pick the Country", df2["country"].unique())
if not state:
    df3 = df2.copy()
else:
    df3 = df2[df2["country"].isin(state)]
    
if region and state:
    filtered_df = df[df["sector"].isin(region) & df["country"].isin(state)]
elif region:
    filtered_df = df[df["sector"].isin(region)]
elif state:
    filtered_df = df[df["country"].isin(state)]
else:
    filtered_df = df.copy()
    
# Aggregate total CO2 emissions by year
# Aggregate total CO2 emissions by country
total_emissions_by_country = filtered_df.groupby('country')['value'].sum().sort_values(ascending=False).reset_index()
fig = px.bar(total_emissions_by_country, x='country', y='value', 
             labels={'value': 'Total CO2 Emissions', 'country': 'Country'},
             color='value',  # Color the bars by the value of CO2 emissions
             title='Total CO2 Emissions by Country')
# Plot total CO2 emissions by country
fig.update_layout(width=1000,height=600)
total_emissions_by_country.plot(kind='bar', color='steelblue')
st.plotly_chart(fig,use_container_width=True, height = 200)

total_emissions_by_sector = filtered_df.groupby('sector')['value'].sum().sort_values(ascending=False).reset_index()
fig = px.pie(total_emissions_by_sector, names='sector', values='value', 
             title='Total CO2 Emissions by Sector',
             labels={'value':'Total CO2 Emissions'})
fig.update_layout(width=1000, height=600)
st.plotly_chart(fig, use_container_width=True)
total_emissions_by_year = filtered_df.groupby('year')['value'].sum().reset_index()

# Convert the aggregated data into a Plotly line chart figure
fig = px.line(total_emissions_by_year, x='year', y='value', 
              markers=True,  # This adds markers to the line, similar to 'marker' in matplotlib
              labels={'value': 'Total CO2 Emissions', 'year': 'Year'},
              title='Total CO2 Emissions by Year')

# Update the layout of the figure
fig.update_layout(width=800, height=600, xaxis=dict(tickmode='array', tickvals=total_emissions_by_year['year']),
                  yaxis_title='Total CO2 Emissions',
                  xaxis_title='Year')

# Display the figure in Streamlit
st.plotly_chart(fig, use_container_width=True)
total_emissions_by_year_sector = filtered_df.groupby(['year', 'sector'])['value'].sum().unstack()

# Convert to a "long-form" or "tidy" DataFrame
total_emissions_long = total_emissions_by_year_sector.reset_index().melt(id_vars='year', var_name='sector', value_name='value')

# Create the Plotly figure
fig = px.line(total_emissions_long, x='year', y='value', color='sector',
              labels={'value': 'Total CO2 Emissions', 'year': 'Year', 'sector': 'Sector'},
              title='Total CO2 Emissions by Year and Sector')

# Update the layout of the figure
fig.update_layout(width=800, height=600, xaxis=dict(tickmode='array', tickvals=total_emissions_by_year_sector.index),
                  yaxis_title='Total CO2 Emissions',
                  xaxis_title='Year',
                  legend_title='Sector')

# Add markers to the line
fig.update_traces(mode='lines+markers')

# Display the figure in Streamlit
st.plotly_chart(fig, use_container_width=True)
top_countries = ['China', 'US', 'EU27 & UK']

# Aggregate total CO2 emissions by year and country for the top countries
total_emissions_by_year_country = filtered_df[df['country'].isin(top_countries)].groupby(['year', 'country'])['value'].sum().reset_index()

# Create the Plotly figure
fig = px.line(total_emissions_by_year_country, x='year', y='value', color='country',
              labels={'value': 'Total CO2 Emissions', 'year': 'Year', 'country': 'Country'},
              title='Total CO2 Emissions by Year for Top Emitting Countries')

# Update the layout of the figure
fig.update_layout(width=800, height=600,
                  yaxis_title='Total CO2 Emissions',
                  xaxis_title='Year',
                  legend_title='Country')

# Add markers to the line
fig.update_traces(mode='lines+markers')

# Display the figure in Streamlit
st.plotly_chart(fig, use_container_width=True)
def create_plotly_figure_for_country(country_name):
    # Filter data for the specified country
    country_data = filtered_df[df['country'] == country_name]
    
    # Aggregate total CO2 emissions by year and sector
    total_emissions_by_year_sector = country_data.groupby(['year', 'sector'])['value'].sum().unstack().reset_index()
    
    # Convert to a "long-form" DataFrame
    total_emissions_long = total_emissions_by_year_sector.melt(id_vars='year', var_name='sector', value_name='value')
    
    # Create the Plotly figure
    fig = px.line(total_emissions_long, x='year', y='value', color='sector',
                  labels={'value': 'Total CO2 Emissions', 'year': 'Year', 'sector': 'Sector'},
                  title=f'Total CO2 Emissions by Year and Sector for {country_name}')
    
    # Update the layout of the figure
    fig.update_layout(width=800, height=600,
                      yaxis_title='Total CO2 Emissions',
                      xaxis_title='Year',
                      legend_title='Sector')
    
    # Add markers to the line
    fig.update_traces(mode='lines+markers')
    
    return fig

# Create and display the Plotly figures for China, US, and EU27 & UK
china_fig = create_plotly_figure_for_country('China')
us_fig = create_plotly_figure_for_country('US')
eu27_uk_fig = create_plotly_figure_for_country('EU27 & UK')

# Display the figures in Streamlit
st.plotly_chart(china_fig, use_container_width=True)
st.plotly_chart(us_fig, use_container_width=True)
st.plotly_chart(eu27_uk_fig, use_container_width=True)
def create_monthly_emissions_plot(data, country_name):
    # Filter data for the specified country
    country_data = data[data['country'] == country_name]
    
    # Create a 'month' column for easy monthly aggregation
    country_data['month'] = country_data['date'].dt.month
    
    # Aggregate total CO2 emissions by year and month
    total_emissions_by_year_month = country_data.groupby(['year', 'month'])['value'].sum().unstack()

    # Convert to 'long-form' DataFrame for Plotly
    total_emissions_long = total_emissions_by_year_month.stack().reset_index(name='value')
    total_emissions_long['month'] = total_emissions_long['month'].astype(int)  # Ensure month is int for sorting

    # Create the Plotly figure
    fig = px.line(total_emissions_long, x='month', y='value', color='year', markers=True,
                  labels={'value': 'Total CO2 Emissions', 'month': 'Month', 'year': 'Year'},
                  title=f'Total CO2 Emissions by Month for Each Year ({country_name})')

    # Update the layout of the figure
    fig.update_layout(width=800, height=600,
                      yaxis_title='Total CO2 Emissions',
                      xaxis_title='Month')

    return fig



# Create and display the Plotly figures for 'WORLD', 'China', 'EU27 & UK', and 'US'
world_fig = create_monthly_emissions_plot(df, 'WORLD')
china_fig = create_monthly_emissions_plot(df, 'China')
eu27_uk_fig = create_monthly_emissions_plot(df, 'EU27 & UK')
us_fig = create_monthly_emissions_plot(df, 'US')

# Display the figures in Streamlit
st.plotly_chart(world_fig, use_container_width=True)
st.plotly_chart(china_fig, use_container_width=True)
st.plotly_chart(eu27_uk_fig, use_container_width=True)
st.plotly_chart(us_fig, use_container_width=True)
def create_avg_emissions_plot(data, country, title):
    # Create 'day_of_week' column
    data['day_of_week'] = data['date'].dt.dayofweek
    
    # Aggregate average daily CO2 emissions by day of week and year
    avg_emissions = data[data['country'] == country].groupby(['year', 'day_of_week'])['value'].mean().unstack()
    
    # Convert data to long format for Plotly
    avg_emissions_long = avg_emissions.reset_index().melt(id_vars='year', var_name='day_of_week', value_name='value')
    avg_emissions_long['day_of_week'] = avg_emissions_long['day_of_week'].astype(str)  # Convert to string for Plotly categorical axis
    
    # Create Plotly figure
    fig = px.line(avg_emissions_long, x='day_of_week', y='value', color='year',
                  labels={'day_of_week': 'Day of Week (0 = Monday, 6 = Sunday)',
                          'value': 'Average Daily CO2 Emissions',
                          'year': 'Year'},
                  title=title)
    
    # Update layout if needed
    fig.update_layout(width=800, height=600)
    
    return fig

# Assuming `data` is your DataFrame and 'year', 'day_of_week', 'country', and 'value' are columns in your DataFrame
# Make sure 'date' column is converted to datetime
filtered_df['date'] = pd.to_datetime(df['date'], format="%d/%m/%Y")

# Create and display the Plotly figures
world_fig = create_avg_emissions_plot(df, 'WORLD', 'Average Daily CO2 Emissions by Day of Week for Each Year (WORLD)')
st.plotly_chart(world_fig, use_container_width=True)

# Repeat the process for China, US, EU27 & UK
china_fig = create_avg_emissions_plot(df, 'China', 'Average Daily CO2 Emissions by Day of Week for Each Year (China)')
us_fig = create_avg_emissions_plot(df, 'US', 'Average Daily CO2 Emissions by Day of Week for Each Year (US)')
eu27_uk_fig = create_avg_emissions_plot(df, 'EU27 & UK', 'Average Daily CO2 Emissions by Day of Week for Each Year (EU27 & UK)')

# Display each figure using Streamlit
st.plotly_chart(china_fig, use_container_width=True)
st.plotly_chart(us_fig, use_container_width=True)
st.plotly_chart(eu27_uk_fig, use_container_width=True)
def create_emissions_plot(data, sector, country_name, title):
    # Filter data for the specified sector and country
    sector_country_data = data[(data['sector'] == sector) & (data['country'] == country_name)]
    
    # Aggregate total CO2 emissions by year and month
    emissions_by_year_month = sector_country_data.groupby(['year', 'month'])['value'].sum().unstack()
    
    # Convert to 'long-form' DataFrame for Plotly
    emissions_long = emissions_by_year_month.reset_index().melt(id_vars='year', var_name='month', value_name='value')
    
    # Create Plotly figure
    fig = px.line(emissions_long, x='month', y='value', color='year', markers=True,
                  labels={'value': 'Total CO2 Emissions', 'month': 'Month', 'year': 'Year'},
                  title=title)
    
    # Update layout if needed
    fig.update_layout(width=800, height=600)
    
    return fig


df['month'] = df['date'].dt.month

# Create and display the Plotly figures
for country in ['China', 'US', 'EU27 & UK']:
    fig_power = create_emissions_plot(df, 'Power', country, f'Total CO2 Emissions by Month for Each Year (Power Sector, {country})')
    st.plotly_chart(fig_power, use_container_width=True)

    fig_industry = create_emissions_plot(df, 'Industry', country, f'Total CO2 Emissions by Month for Each Year (Industry Sector, {country})')
    st.plotly_chart(fig_industry, use_container_width=True)
import streamlit as st
import pandas as pd
import plotly.express as px

def plot_emissions(data, sector, country, title):
    # Filter the data for the sector and country
    sector_data = data[(data['sector'] == sector) & (data['country'] == country)]
    # Aggregate total emissions by year and month
    emissions_by_year_month = sector_data.groupby(['year', 'month'])['value'].sum().reset_index()
    
    # Create the Plotly figure
    fig = px.line(emissions_by_year_month, x='month', y='value', color='year', markers=True,
                  title=title,
                  labels={'value': 'Total CO2 Emissions', 'month': 'Month', 'year': 'Year'})
    # Update layout if needed
    fig.update_layout(width=800, height=600)
    
    return fig
for country in ['China', 'US', 'EU27 & UK']:
    fig = plot_emissions(df, 'Ground Transport', country, f'Total CO2 Emissions by Month for Each Year (Ground Transport Sector, {country})')
    st.plotly_chart(fig, use_container_width=True)
    
import plotly.graph_objects as go
# Filter data for the 'Power' sector in 'China'
power_china_data = filtered_df[(df['country'] == 'US') & (df['sector'] == 'Power')]
# Resample the data to get total monthly emissions
monthly_emissions = power_china_data.resample('M', on='date')['value'].sum()
# Fit an ARIMA model
model = ARIMA(monthly_emissions, order=(2, 1, 2))
model_fit = model.fit()
# Make predictions for the next year
predictions = model_fit.predict(start=len(monthly_emissions), end=len(monthly_emissions) + 12, typ='levels')

# Convert ARIMA predictions to a DataFrame for plotting
predictions_df = pd.DataFrame({'date': pd.date_range(start=monthly_emissions.index[-1] + pd.Timedelta(days=1), periods=len(predictions), freq='M'),
                               'Predicted Value': predictions})

# Create the Plotly figure for historical data and predictions
fig = go.Figure()
fig.add_trace(go.Scatter(x=monthly_emissions.index, y=monthly_emissions, mode='lines+markers', name='Historical Data'))
fig.add_trace(go.Scatter(x=predictions_df['date'], y=predictions_df['Predicted Value'], mode='lines+markers', name='Predictions'))

fig.update_layout(title='Total Monthly CO2 Emissions from the Power Sector in US (Historical and Predicted)',
                  xaxis_title='Date',
                  yaxis_title='Total Monthly CO2 Emissions (MtCO2)',
                  width=800, height=600)

st.plotly_chart(fig, use_container_width=True)
filtered_df['date'] = pd.to_datetime(df['date'])







# Prepare the data for plotting
historical_dates = monthly_emissions.index
prediction_dates = pd.date_range(start=historical_dates[-1] + pd.offsets.MonthEnd(1), periods=13, freq='M')[1:]
all_dates = historical_dates.union(prediction_dates)

# Combine historical data and predictions into one DataFrame
combined_data = pd.Series(index=all_dates, dtype='float64')
combined_data[:len(monthly_emissions)] = monthly_emissions
combined_data[-len(predictions):] = predictions

# Create the Plotly figure for historical data and predictions
fig = go.Figure()

# Add historical data trace
fig.add_trace(go.Scatter(x=historical_dates, y=monthly_emissions, mode='lines+markers', name='Historical Data'))

# Add predictions trace
fig.add_trace(go.Scatter(x=prediction_dates, y=predictions, mode='lines+markers', name='Predictions'))

# Update layout
fig.update_layout(
    title='Total Monthly CO2 Emissions from the Power Sector in China (Historical and Predicted)',
    xaxis_title='Date',
    yaxis_title='Total Monthly CO2 Emissions (MtCO2)',
    width=800,
    height=600
)

# Display the figure in Streamlit
st.plotly_chart(fig, use_container_width=True)



# Since we do not have future actual data, we will split the dataset into train and test
# to evaluate the performance of our model on the test set as a proxy for actual data.
split_point = int(len(monthly_emissions) * 0.8)
train, test = monthly_emissions[0:split_point], monthly_emissions[split_point:]

# Define and fit the model on the train set
model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12)
model_fit = model.fit()

# Forecast on the test set
forecast = model_fit.forecast(len(test))

# Calculate error metrics
mae = mean_absolute_error(test, forecast)
rmse = mean_squared_error(test, forecast, squared=False)  # Pass squared=False to get RMSE

# Print the metrics
mae, rmse

