#!/opt/anaconda3/bin/ipython python
# coding: utf-8

# # Run all the below for the dashboard

# In[1]:

import sys
print (sys.version)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math 
import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html 
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut


# # pull the latest john hopkins time series dataset

# In[2]:

my_path = os.path.dirname(os.path.realpath(__file__))

if os.path.isdir('./COVID-19'):
    print("checkout fresh version")
    pull = "git checkout ." 
    path = r"./COVID-19"

    #os.system("sshpass -p your_password ssh user_name@your_localhost")
    os.chdir(path) # Specifying the path where the cloned project needs to be copied
    os.system(pull) # pulling
else:
    print("clone the john hopkins data")
    github_path  = "https://github.com/CSSEGISandData/COVID-19.git" 
    pull = "git clone" 
    os.system(pull + " " + github_path) # pulling
    path = r"./COVID-19"
    os.chdir(path) # Specifying the path where the cloned project needs to be copied


# In[3]:


url = 'https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_confirmed_global.csv&filename=time_series_covid19_confirmed_global.csv'
time_series_data_confirmed = pd.read_csv(url)
url = 'https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_deaths_global.csv&filename=time_series_covid19_deaths_global.csv'
time_series_data_deaths = pd.read_csv(url)
url = 'https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_recovered_global.csv&filename=time_series_covid19_recovered_global.csv'
time_series_data_recovered = pd.read_csv(url)


# In[4]:


time_series_data_confirmed


# In[5]:


time_series_data_confirmed.columns


# In[6]:


def df_for_lineplot_diff(dfs, CaseType):
    '''This is the function for construct df for line plot'''
    
    assert type(CaseType) is str, "CaseType must be one of the following three strings Confirmed/Recovered/Deaths"
    
    
    # Construct confirmed cases dataframe for line plot
    DateList = []
    ChinaList =[]
    OtherList = []

    for key, df in dfs.items():
        dfTpm = df.groupby(['Country/Region'])[CaseType].agg(np.sum)
        dfTpm = pd.DataFrame({'Region':dfTpm.index, CaseType:dfTpm.values})
        #dfTpm = dfTpm.sort_values(by=CaseType, ascending=False).reset_index(drop=True)
        DateList.append(df['Date_last_updated_AEDT'][0])
        #DateList.append(df['Last Update'][0])
        
        ChinaList.append(dfTpm.loc[dfTpm['Region'] == 'China', CaseType].iloc[0])
        OtherList.append(dfTpm.loc[dfTpm['Region'] != 'China', CaseType].sum())

    df = pd.DataFrame({'Date':DateList,
                       'Mainland China':ChinaList,
                       'Other locations':OtherList})
    df['Total']=df['Mainland China']+df['Other locations']

    # Calculate differenec in a 24-hour window
    for index, _ in df.iterrows():
        # Calculate the time differnece in hour
        diff=(df['Date'][0] - df['Date'][index]).total_seconds()/3600
        # find out the latest time after 24-hour
        if diff >= 20:
            break
    plusNum = df['Total'][0] - df['Total'][1]
    plusPercentNum = (df['Total'][0] - df['Total'][1])/df['Total'][1]

    # Select the latest data from a given date
    df['date_day']=[d.date() for d in df['Date']]
    df=df.groupby(by=df['date_day'], sort=False).transform(max).drop_duplicates(['Date'])
    
    df['plusNum'] = plusNum
    df['plusPercentNum'] = plusPercentNum
    
    df=df.reset_index(drop=True)
    
    return df, plusNum, plusPercentNum 


# In[7]:


###get_ipython().run_cell_magic('time', '', "################################################################################\n#### Data processing\n################################################################################\n# Method #1\n# Import csv file and store each csv in to a df list\n\nfilename = os.listdir('./csse_covid_19_data/csse_covid_19_daily_reports/')\nsheet_name = [i.replace('.csv', '') for i in filename if 'data' not in i and i.endswith('.csv')]\nsheet_name.sort(reverse=True)\n\ndfs = {sheet_name: pd.read_csv('./csse_covid_19_data/csse_covid_19_daily_reports/{}.csv'.format(sheet_name))\n          for sheet_name in sheet_name}")
filename= os.listdir('./csse_covid_19_data/csse_covid_19_daily_reports/')
sheet_name = [i.replace('.csv', '') for i in filename if 'data' not in i and i.endswith('.csv')]
sheet_name.sort(reverse=True)
dfs = {sheet_name: pd.read_csv('./csse_covid_19_data/csse_covid_19_daily_reports/{}.csv'.format(sheet_name))
        for sheet_name in sheet_name}


# In[8]:


# Data from each sheet can be accessed via key
keyList = list(dfs.keys())
# Data cleansing
for key, df in dfs.items():
    dfs[key]=dfs[key].rename(columns={'Last_Update': 'Last Update', "Province_State":"Province/State", "Country_Region":"Country/Region", "Lat":"lat", "Long_":"lon", "Latitude":"lat", "Longitude":"lon"})
    dfs[key].loc[:,'Confirmed'].fillna(value=0, inplace=True)
    dfs[key].loc[:,'Deaths'].fillna(value=0, inplace=True)
    dfs[key].loc[:,'Recovered'].fillna(value=0, inplace=True)
    dfs[key]=dfs[key].astype({'Confirmed':'int64', 'Deaths':'int64', 'Recovered':'int64'})
    # Change as China for coordinate search
    dfs[key]=dfs[key].replace({'Country/Region':'Mainland China'}, 'China')
    # Add a zero to the date so can be convert by datetime.strptime as 0-padded date
    dfs[key]['Last Update'] = '0' + dfs[key]['Last Update']
    # Convert time as Australian eastern daylight time
    ### Try to sort out the crazy variety of formatting of dates in the John Hopkins Data
    dates_append = []
    for d in dfs[key]['Last Update']:
        d = str(d)
        d = d.replace("T"," ")
        if "-" in d:
            dates_append.append(datetime.strptime(d[-19:], '%Y-%m-%d %H:%M:%S'))
            #dfs[key]['Date_last_updated_AEDT'] = [datetime.strptime(d, '%Y/%m/%d %H:%M:%S') for d in dfs[key]['Last Update']]
        elif d == 'nan':
            dates_append.append('')
        else:
            try:
                dates_append.append(datetime.strptime(d[-14:], '%m/%d/%Y %H:%M'))
            except:
                try:
                    dates_append.append(datetime.strptime(d[-14:], '%m/%d/%y %H:%M'))
                except:
                    try:
                        dates_append.append(datetime.strptime(d[-13:], '%m/%d/%y %H:%M'))
                    except:
                        try:
                            dates_append.append(datetime.strptime(d, '%m/%d/%Y %H:%M'))
                        except:
                            try:
                                dates_append.append(datetime.strptime(d[-16:], '%m/%d/%Y %H:%M'))
                            except:
                                try:
                                    dates_append.append(datetime.strptime(d[-16:], '%m/%d/%Y %H:%M'))
                                except:
                                    try:
                                        dates_append.append(datetime.strptime(d[-12:], '%m/%d/%y %H:%M'))
                                    except:
                                        print('Error could not parse date: ', d)
                                        dates_append.append('')
                #dfs[key]['Date_last_updated_AEDT'] = [datetime.strptime(d, '%m/%d/%Y %H:%M') for d in dfs[key]['Last Update']]
    dfs[key]['Date_last_updated_AEDT'] = dates_append
    dfs[key]['Date_last_updated_AEDT'] = dfs[key]['Date_last_updated_AEDT'] + timedelta(hours=16)
    dfs[key]['Remaining'] = dfs[key]['Confirmed'] - dfs[key]['Recovered'] - dfs[key]['Deaths']
    dfs[key] = dfs[key].fillna('')
    #print(key)
    #print(df)


# # Add coordinates for each area in the list for the latest table sheet
# # To save time, coordinates calling was done seperately
# # Import the data with coordinates
# data_sheet_df = pd.read_csv(r'../2020-03-24-06-00_data.csv'.format(keyList[0]))
# print(data_sheet_df)
# dfs[keyList[0]]=dfs[keyList[0]].astype({'Date_last_updated_AEDT':'datetime64'})

# In[9]:


#merged_df = left_df.merge(right_df, how='inner', left_on=["A", "B"], right_on=["A2","B2"])
merged_df = dfs[keyList[0]].merge(time_series_data_confirmed[["Province/State", "Country/Region", "3/30/20"]], how='outer', left_on=["Province/State", "Country/Region"], right_on=["Province/State", "Country/Region"])
merged_df['Confirmed'] = merged_df['3/30/20']
del merged_df['3/30/20']
merged_df = merged_df.merge(time_series_data_deaths[["Province/State", "Country/Region", "3/30/20"]], how='outer', left_on=["Province/State", "Country/Region"], right_on=["Province/State", "Country/Region"])
merged_df['Deaths'] = merged_df['3/30/20']
del merged_df['3/30/20']
merged_df = merged_df.merge(time_series_data_recovered[["Province/State", "Country/Region", "3/30/20"]], how='outer', left_on=["Province/State", "Country/Region"], right_on=["Province/State", "Country/Region"])
merged_df['Recovered'] = merged_df['3/30/20']
del merged_df['3/30/20']


# In[10]:


merged_df.columns


# In[11]:


baseURL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"
def loadData(fileName, columnName):
    data = pd.read_csv(baseURL + fileName)              .drop(['Lat', 'Long'], axis=1)              .melt(id_vars=['Province/State', 'Country/Region'], 
                 var_name='date', value_name=columnName) \
             .astype({'date':'datetime64[ns]', columnName:'Int64'}, 
                 errors='ignore')
    data['Province/State'].fillna('<all>', inplace=True)
    data[columnName].fillna(0, inplace=True)
    return data


# In[12]:


def make_country_table(countryName):
    '''This is the function for building df for Province/State of a given country'''
    countryTable = dfs[keyList[0]].loc[dfs[keyList[0]]['Country/Region'] == countryName]
    # Suppress SettingWithCopyWarning
    pd.options.mode.chained_assignment = None
    countryTable['Remaining'] = countryTable['Confirmed'] - countryTable['Recovered'] - countryTable['Deaths']
    countryTable = countryTable[['Province/State','Remaining','Confirmed','Recovered','Deaths','lat','lon']]
    countryTable = countryTable.sort_values(by=['Remaining', 'Confirmed'], ascending=False).reset_index(drop=True)
    # Set row ids pass to selected_row_ids
    countryTable['id'] = countryTable['Province/State']
    countryTable.set_index('id', inplace=True, drop=False)
    # Turn on SettingWithCopyWarning
    pd.options.mode.chained_assignment = 'warn'
    return countryTable


# In[13]:


###get_ipython().run_cell_magic('time', '', "CNTable = make_country_table('China')\nAUSTable = make_country_table('Australia')\nUSTable = make_country_table('US')\nCANTable = make_country_table('Canada')")

CNTable = make_country_table('China')
AUSTable = make_country_table('Australia')
USTable = make_country_table('US')
CANTable = make_country_table('Canada')


# In[14]:


CANTable


# # Work on the UK as a special case where we try to forecast the number of free beds

# # Step 1: Build the UK COVID-19 forecase

# In[15]:


url = "https://www.arcgis.com/sharing/rest/content/items/e5fd11150d274bebaaf8fe2a7a2bda11/data"
df = pd.read_excel(url)
df = df.loc[:,['DateVal','CumCases']]
FMT = '%Y-%m-%d'
date = df['DateVal']
df['data'] = date.map(lambda x : (x - datetime.strptime("2020-01-01", FMT)).days)

#The logistic model
def logistic_model(x,a,b,c):
    return c/(1+np.exp(-(x-b)/a))

#We can use the curve_fit function of scipy library to estimate the parameter values and errors starting from the original data.

x = list(df['data'])
y = list(df['CumCases'])

fit = curve_fit(logistic_model,x,y,p0=[2,100,20000])
a = fit[0][0]
b = fit[0][1]
c = fit[0][2]
errors = [np.sqrt(fit[1][i][i]) for i in [0,1,2]]
sol = int(fsolve(lambda x : logistic_model(x,a,b,c) - int(c),b))

#Exponential model
def exponential_model(x,a,b,c):
    return a*np.exp(b*(x-c))


exp_fit = curve_fit(exponential_model,x,y,p0=[1,1,1],maxfev=2000)
base = datetime.strptime(str(date[-1:].values[0]).replace("T"," ")[:19],"%Y-%m-%d %H:%M:%S")
dates = []

for i_date in date:
    dates.append(datetime.strptime(str(i_date).replace("T"," ")[:19],"%Y-%m-%d %H:%M:%S"))

date_list_pred = [base + timedelta(days=i_x) for i_x in range(1,sol)]
date_list = []
for i_date in dates:
    date_list.append(i_date)

for i_date in date_list_pred:
    date_list.append(i_date)

fig = plt.figure(figsize=(20,10))

pred_x = list(range(max(x),sol))
plt.rcParams['figure.figsize'] = [7, 7]
plt.rc('font', size=14)
# Real data
plt.scatter(dates,y,label="Real data",color="red")
# Predicted logistic curve
plt.plot(date_list, [logistic_model(i,fit[0][0],fit[0][1],fit[0][2]) for i in range(0+min(x),len(date_list)+min(x))], label="Logistic model" )
# Predicted exponential curve
plt.plot(date_list, [exponential_model(i,exp_fit[0][0],exp_fit[0][1],exp_fit[0][2]) for i in range(0+min(x),len(date_list)+min(x))], label="Exponential model" )
plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Total number of infected people")
plt.ylim((min(y)*0.9,c*1.1))
plt.show()


# # Step 2: Download the NHS England free beds hisorical data for a baseline of free beds at different NHS Trusts in England

# In[16]:


lm_predicted = [logistic_model(i,fit[0][0],fit[0][1],fit[0][2]) for i in range(0+min(x),len(date_list)+min(x))]
from datetime import date, datetime

today = date.today()

today_with_time = datetime(
    year=today.year, 
    month=today.month,
    day=today.day,
    hour=0,
    minute=0
)

try:
    location = date_list.index(today_with_time)
    print(today, " was found in the list.") 

except:
    print(today, " was not found in the list.") 

total_cases_predicted_in_a_week = lm_predicted[location+7]
current_cases = lm_predicted[location]
growth_factor_in_a_week = total_cases_predicted_in_a_week / current_cases
print(growth_factor_in_a_week)

#Read in the beds data: Using the beds data for England in Q1 2019 as a baseline (current Quarter in 2020) before the COVID-19 cases

### Using Q1 beds data from 2019 as a baseline
df_england_beds = pd.read_excel('https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2/2020/02/Beds-Timeseries-2010-11-onwards-Q3-2019-20-ADJ-for-missings-j8hyu.xls', header=13)
df_england_beds[['Year','Total ', 'General & Acute']].plot()
df_england_beds[['Year','Total .1', 'General & Acute.1']].plot()
df_england_beds[['Year','Total .2', 'General & Acute.2']].plot()

#read in the free beds via hospital
df_england_beds_region = pd.read_excel("https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2/2019/11/Beds-Open-Overnight-Web_File-Final-Q1-201920.xlsx", header=14)
df_england_beds_region['Free General & Acute'] = df_england_beds_region['General & Acute'] - df_england_beds_region['General & Acute.1']
df_england_beds_region.index = df_england_beds_region['Org Name']
df_england_beds_region['Free General & Acute'][2:].plot(figsize=(50,5), kind='bar', ylim=(0,300))


# # Get the England regions COVID-19 data

# In[17]:


#read in the latest UK COVID-19 data for England regions
# download UK regional cases
url = "https://www.arcgis.com/sharing/rest/content/items/b684319181f94875a6879bbc833ca3a6/data"
df_UK = pd.read_csv(url)
df_UK.head()

df_UK_lat_lon = pd.read_csv('../df_UK_lat_lon.csv')
del df_UK_lat_lon['Unnamed: 0']
df_UK = df_UK.merge(df_UK_lat_lon, how='outer', left_on=["GSS_NM"], right_on=["GSS_NM"])

#from geopy.geocoders import Nominatim
#from geopy.exc import GeocoderTimedOut
#geolocator = Nominatim(user_agent="covid_shahinrostami.com")
#
#for index, row in df_UK.iterrows():
#    location = geolocator.geocode(row.GSS_NM+", UK",timeout=100)
#    df_UK.loc[index,'lat'] = location.latitude 
#    df_UK.loc[index,'lon'] = location.longitude
#
#print("Done!")


# In[18]:



tmp_total_cases = []
# eunsure we have total cases as floats and remove commas to do so
for item in df_UK['TotalCases']:
    if isinstance(item, str):
        tmp_total_cases.append(float(item.replace(',','')))
df_UK['TotalCases'] = tmp_total_cases

df_UK_tmp = df_UK.copy()


# # Match the UK local data to the NHS beds data for England NHS trusts

# In[19]:


import pgeocode
#match the local data to the beds regions
df_nhs_trusts = pd.read_csv('https://nhsenglandfilestore.s3.amazonaws.com/ods/etr.csv', header=None)
matches = []

for org_name in df_england_beds_region['Org Name']: 
    for trust in df_nhs_trusts[1]:
        if trust == org_name:
            print(trust, ' matches ', org_name)
            matches.append(trust + ' matches ' + org_name)

df_england_beds_region.reset_index(drop=True)

merged_df = df_england_beds_region.reset_index(drop=True).merge(df_nhs_trusts[[1,9]], how='outer', left_on=["Org Name"], right_on=[1])

nomi = pgeocode.Nominatim('gb')

lat_list = []
lon_list = []
for item in merged_df[9]:
    item = str(item)
    if item == "NaN" or item == "nan":
        lat_list.append(0)
        lon_list.append(0)
    else:
        lat = nomi.query_postal_code(item).latitude
        lon = nomi.query_postal_code(item).longitude
        lat_list.append(float(lat.item()))
        lon_list.append(float(lon.item()))
merged_df['lat'] = lat_list
merged_df['lon'] = lon_list

#find the closest lat and lon to that trust from the geospatial data and join the datasets to the beds dataset

from math import cos, asin, sqrt

def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
    return 12742 * asin(sqrt(a))

def closest(data, v):
    return min(data, key=lambda p: distance(v['lat'],v['lon'],p['lat'],p['lon']))

hospital_lat = []
hospital_lon = []

for index, row in df_UK.iterrows():
    v = {'lat': row['lat'], 'lon': row['lon']}
    match = closest(list(merged_df[['lat','lon']].T.to_dict().values()),v)
    hospital_lat.append(match['lat'])
    hospital_lon.append(match['lon'])

df_UK['hospital_lat'] = hospital_lat
df_UK['hospital_lon'] = hospital_lon

df_UK_merged = df_UK.merge(merged_df[["Org Name", "Total ", "General & Acute",  "Total .1", "General & Acute.1","lat","lon"]], how='inner', left_on=["hospital_lon", "hospital_lat"], right_on=["lon","lat"])
df_UK_merged['Free Beds Without COVID-19'] = df_UK_merged['General & Acute'] - df_UK_merged['General & Acute.1']

Free_Beds_INCLUDING_COVID = []
counts_list = df_UK_merged.groupby(by=['GSS_NM'])['GSS_NM'].count()

for item in df_UK_merged['GSS_NM']:
    Free_Beds_INCLUDING_COVID.append(counts_list[item])

df_UK_merged['count'] = Free_Beds_INCLUDING_COVID
percentage_cases_open = 1-(277643/1044167)

#8.2% of patients require hospitalisation according to Imperial College study

percentage_hospital = percentage_cases_open * 0.082 
percentage_serious = percentage_cases_open * 0.05 

print(percentage_cases_open, percentage_serious, percentage_hospital)
print("percentage_hospital: ",percentage_hospital)
print("Total Cases: ",df_UK_merged['TotalCases'][0])
print("Ratio: ",df_UK_merged['TotalCases']/df_UK_merged['count'])
    
df_UK_merged['Free Beds INCLUDING COVID-19'] = df_UK_merged['Free Beds Without COVID-19'] - (percentage_hospital * df_UK_merged['TotalCases']/df_UK_merged['count'].astype(float))

df_UK_merged.index = df_UK_merged['Org Name']


# # Print out and plot the expectation for current free beds in the NHS trusts in England

# In[20]:


minimum = min(df_UK_merged['Free Beds INCLUDING COVID-19'][2:].fillna(0))
maximum = max(df_UK_merged['Free Beds INCLUDING COVID-19'][2:].fillna(0))

#Plot a chart representing the current hospital bed vacancy by trust by incrementing over and above 2019 Q1 total free beds with COVID-19 estimated current cases requiring hospitalisation in the UK
df_UK_merged.sort_values(by=['Free Beds INCLUDING COVID-19'])['Free Beds INCLUDING COVID-19'].dropna()[2:].plot(figsize=(100,50), kind='bar', ylim=(minimum,maximum),color=(df_UK_merged.sort_values(by=['Free Beds INCLUDING COVID-19'])['Free Beds INCLUDING COVID-19'][2:].dropna() > 0).map({True: 'g',False: 'r'}),title="Estimated current number of free beds after COVID-19 by hospital trust in the UK",fontsize = 30)
df_UK_merged.sort_values(by=['Free Beds INCLUDING COVID-19'])


# # Print out and plot the forecasted free beds in 1 WEEK for NHS Enland Trusts

# In[21]:


#Plot a chart representing the NEXT WEEKS hospital bed vacancy by trust by incrementing over and above 2019 Q1 total free beds with COVID-19 estimated cases 1 WEEK FROM TODAY requiring hospitalisation in the UK
df_UK_merged['Predicted Free Beds INCLUDING COVID-19 in 1 WEEK'] = df_UK_merged['Free Beds Without COVID-19'] - (percentage_hospital * growth_factor_in_a_week * df_UK_merged['TotalCases']/df_UK_merged['count'])

minimum = min(df_UK_merged['Predicted Free Beds INCLUDING COVID-19 in 1 WEEK'][2:].fillna(0))
maximum = max(df_UK_merged['Predicted Free Beds INCLUDING COVID-19 in 1 WEEK'][2:].fillna(0))

df_UK_merged.sort_values(by=['Predicted Free Beds INCLUDING COVID-19 in 1 WEEK'])['Predicted Free Beds INCLUDING COVID-19 in 1 WEEK'].dropna()[2:].plot(figsize=(100,50), kind='bar', ylim=(minimum,maximum),color=(df_UK_merged.sort_values(by=['Predicted Free Beds INCLUDING COVID-19 in 1 WEEK'])['Predicted Free Beds INCLUDING COVID-19 in 1 WEEK'][2:].dropna() > 0).map({True: 'g',False: 'r'}),title="Estimated current number of free beds after COVID-19 by hospital trust in the UK",fontsize = 30)
df_UK_merged.sort_values(by=['Predicted Free Beds INCLUDING COVID-19 in 1 WEEK']).dropna()


# # Go back to building the UK regions data for the dashboard

# In[22]:


df_UK = df_UK_tmp


# df_UK = pd.read_csv('df_UK.csv')
# del df_UK['Unnamed: 0']

# In[23]:


df_UK


# df_UK.to_csv('df_UK.csv')

# In[24]:


UKTable= pd.DataFrame(columns=CANTable.columns)
UKTable['Province/State'] = df_UK['GSS_NM']
UKTable['Confirmed'] = df_UK['TotalCases']
UKTable['lat'] = df_UK['lat']
UKTable['lon'] = df_UK['lon']
UKTable['id'] = df_UK['GSS_NM']
UKTable.index = UKTable['id']
UKTable


# In[25]:


UKTable_append = UKTable.copy()


# In[26]:


UKTable_append['Country/Region']='UK'
UKTable_append['Last Update'] = '2020-03-24 22:00:00'
UKTable_append['Date_last_updated_AEDT'] = '2020-03-24 22:00:00'
UKTable_append['Recovered'] = 0
UKTable_append['Deaths'] = 0
UKTable_append['Remaining'] = 0
del UKTable_append['id']
UKTable_append


# In[27]:


dfs[keyList[0]].columns


# In[28]:


###get_ipython().run_cell_magic('time', '', "# Save numbers into variables to use in the app\nconfirmedCases=int(dfs[keyList[0]]['Confirmed'].sum())\ndeathsCases=int(dfs[keyList[0]]['Deaths'].sum())\nrecoveredCases=int(dfs[keyList[0]]['Recovered'].sum())\n\n# Construct confirmed cases dataframe for line plot and 24-hour window case difference\ndf_confirmed, plusConfirmedNum, plusPercentNum1 = df_for_lineplot_diff(dfs, 'Confirmed')\n\n\n# Construct recovered cases dataframe for line plot and 24-hour window case difference\ndf_recovered, plusRecoveredNum, plusPercentNum2 = df_for_lineplot_diff(dfs, 'Recovered')\n\n\n# Construct death case dataframe for line plot and 24-hour window case difference\ndf_deaths, plusDeathNum, plusPercentNum3 = df_for_lineplot_diff(dfs, 'Deaths')\n\n# Construct remaining case dataframe for line plot and 24-hour window case difference\ndf_remaining, plusRemainNum, plusRemainNum3 = df_for_lineplot_diff(dfs, 'Remaining')")
confirmedCases=int(dfs[keyList[0]]['Confirmed'].sum())
deathsCases=int(dfs[keyList[0]]['Deaths'].sum())
recoveredCases=int(dfs[keyList[0]]['Recovered'].sum())

# Construct confirmed cases dataframe for line plot and 24-hour window case difference
df_confirmed, plusConfirmedNum, plusPercentNum1 = df_for_lineplot_diff(dfs, 'Confirmed')


# Construct recovered cases dataframe for line plot and 24-hour window case difference
df_recovered, plusRecoveredNum, plusPercentNum2 = df_for_lineplot_diff(dfs, 'Recovered')


# Construct death case dataframe for line plot and 24-hour window case difference
df_deaths, plusDeathNum, plusPercentNum3 = df_for_lineplot_diff(dfs, 'Deaths')

# Construct remaining case dataframe for line plot and 24-hour window case difference
df_remaining, plusRemainNum, plusRemainNum3 = df_for_lineplot_diff(dfs, 'Remaining')


# In[29]:


confirmedCases=dfs[keyList[0]]['Confirmed'].sum()
confirmedCases


# In[30]:


# Create data table to show in app
# Generate sum values for Country/Region level
dfCase = dfs[keyList[0]].groupby(by='Country/Region', sort=False).sum().reset_index()
dfCase = dfCase.sort_values(by=['Confirmed'], ascending=False).reset_index(drop=True)
# As lat and lon also underwent sum(), which is not desired, remove from this table.
#dfCase = dfCase.drop(columns=['lat','lon'])


# In[31]:


dfs[keyList[0]].sort_values('Confirmed',ascending=False).groupby(by=['Country/Region']).first().reset_index()


# In[32]:


# Grep lat and lon by the first instance to represent its Country/Region
#dfGPS = dfs[keyList[0]].groupby(by=['Country/Region'], sort=True).first().reset_index()
dfGPS = dfs[keyList[0]].sort_values('Confirmed',ascending=False).groupby(by=['Country/Region']).first().reset_index()
dfGPS = dfGPS[['Country/Region','lat','lon']]

# Merge two dataframes
dfSum = pd.merge(dfCase, dfGPS, how='inner', on='Country/Region')
dfSum = dfSum.replace({'Country/Region':'China'}, 'Mainland China')
dfSum['Remaining'] = dfSum['Confirmed'] - dfSum['Recovered'] - dfSum['Deaths']
# Rearrange columns to correspond to the number plate order
dfSum = dfSum[['Country/Region','Remaining','Confirmed','Recovered','Deaths','lat','lon']]
# Sort value based on Remaining cases and then Confirmed cases
dfSum = dfSum.sort_values(by=['Remaining', 'Confirmed'], ascending=False).reset_index(drop=True)
# Set row ids pass to selected_row_ids
dfSum['id'] = dfSum['Country/Region']
dfSum.set_index('id', inplace=True, drop=False)

# Save numbers into variables to use in the app
latestDate=datetime.strftime(df_confirmed['Date'][0], '%b %d, %Y %H:%M AEDT')
secondLastDate=datetime.strftime(df_confirmed['Date'][1], '%b %d')
daysOutbreak=(df_confirmed['Date'][0] - datetime.strptime('12/31/2019', '%m/%d/%Y')).days


# In[33]:


#############################################################################################
#### Start to make plots
#############################################################################################
# Line plot for confirmed cases
# Set up tick scale based on confirmed case number
tickList = list(np.arange(0, df_confirmed['Other locations'].max()+1000, 100000))

# Create empty figure canvas
fig_confirmed = go.Figure()
# Add trace to the figure
fig_confirmed.add_trace(go.Scatter(x=df_confirmed['Date'].fillna(''), y=df_confirmed['Mainland China'],
                                   mode='lines+markers',
                                   line_shape='spline',
                                   name='Mainland China',
                                   line=dict(color='#921113', width=4),
                                   marker=dict(size=4, color='#f4f4f2',
                                               line=dict(width=1,color='#921113')),
                                   text=[datetime.strftime(d, '%b %d %Y AEDT') for d in df_confirmed['Date'].fillna(pd.Timestamp('20200101'))],
                                   hovertext=['Mainland China confirmed<br>{:,d} cases<br>'.format(int(i)) for i in df_confirmed['Mainland China'].fillna(0)],
                                   hovertemplate='<b>%{text}</b><br></br>'+
                                                 '%{hovertext}'+
                                                 '<extra></extra>'))
fig_confirmed.add_trace(go.Scatter(x=df_confirmed['Date'].fillna(''), y=df_confirmed['Other locations'],
                                   mode='lines+markers',
                                   line_shape='spline',
                                   name='Other Region',
                                   line=dict(color='#eb5254', width=4),
                                   marker=dict(size=4, color='#f4f4f2',
                                               line=dict(width=1,color='#eb5254')),
                                   text=[datetime.strftime(d, '%b %d %Y AEDT') for d in df_confirmed['Date'].fillna(pd.Timestamp('20200101'))],
                                   hovertext=['Other region confirmed<br>{:,d} cases<br>'.format(int(i)) for i in df_confirmed['Other locations'].fillna(0)],
                                   hovertemplate='<b>%{text}</b><br></br>'+
                                                 '%{hovertext}'+
                                                 '<extra></extra>'))
# Customise layout
fig_confirmed.update_layout(
#    title=dict(
#    text="<b>Confirmed Cases Timeline<b>",
#    y=0.96, x=0.5, xanchor='center', yanchor='top',
#    font=dict(size=20, color="#292929", family="Playfair Display")
#   ),
    margin=go.layout.Margin(
        l=10,
        r=10,
        b=10,
        t=5,
        pad=0
    ),
    yaxis=dict(
        showline=False, linecolor='#272e3e',
        zeroline=False,
        #showgrid=False,
        gridcolor='rgba(203, 210, 211,.3)',
        gridwidth = .1,
        tickmode='array',
        # Set tick range based on the maximum number
        tickvals=tickList,
        # Set tick label accordingly
        ticktext=["{:.0f}k".format(i/1000) for i in tickList]
    ),
#    yaxis_title="Total Confirmed Case Number",
    xaxis=dict(
        showline=False, linecolor='#272e3e',
        showgrid=False,
        gridcolor='rgba(203, 210, 211,.3)',
        gridwidth = .1,
        zeroline=False
    ),
    xaxis_tickformat='%b %d',
    hovermode = 'x',
    legend_orientation="h",
#    legend=dict(x=.35, y=-.05),
    plot_bgcolor='#f4f4f2',
    paper_bgcolor='#cbd2d3',
    font=dict(color='#292929')
)


# In[34]:


dfs[keyList[0]]=dfs[keyList[0]].append(UKTable_append, ignore_index=True)

# In[ ]:


UKTable_append[UKTable_append['Province/State']=='UK']


# In[35]:



# Line plot for combine cases
# Set up tick scale based on confirmed case number
tickList = list(np.arange(0, df_remaining['Total'].max()+2000, 100000))

# Create empty figure canvas
fig_combine = go.Figure()
# Add trace to the figure
fig_combine.add_trace(go.Scatter(x=df_recovered['Date'], y=df_recovered['Total'],
                                   mode='lines+markers',
                                   line_shape='spline',
                                   name='Total Recovered Cases',
                                   line=dict(color='#168038', width=4),
                                   marker=dict(size=4, color='#f4f4f2',
                                               line=dict(width=1,color='#168038')),
                                   text=[datetime.strftime(d, '%b %d %Y AEDT') for d in df_recovered['Date'].fillna(pd.Timestamp('20200101'))],
                                   hovertext=['Total recovered<br>{:,d} cases<br>'.format(int(i)) for i in df_recovered['Total'].fillna(0)],
                                   hovertemplate='<b>%{text}</b><br></br>'+
                                                 '%{hovertext}'+
                                                 '<extra></extra>'))
fig_combine.add_trace(go.Scatter(x=df_deaths['Date'].fillna(pd.Timestamp('20200101')), y=df_deaths['Total'].fillna(0),
                                mode='lines+markers',
                                line_shape='spline',
                                name='Total Death Cases',
                                line=dict(color='#626262', width=4),
                                marker=dict(size=4, color='#f4f4f2',
                                            line=dict(width=1,color='#626262')),
                                text=[datetime.strftime(d, '%b %d %Y AEDT') for d in df_deaths['Date'].fillna(pd.Timestamp('20200101'))],
                                hovertext=['Total death<br>{:,d} cases<br>'.format(int(i)) for i in df_deaths['Total'].fillna(0)],
                                hovertemplate='<b>%{text}</b><br></br>'+
                                              '%{hovertext}'+
                                              '<extra></extra>'))
fig_combine.add_trace(go.Scatter(x=df_remaining['Date'].fillna(pd.Timestamp('20200101')), y=df_remaining['Total'].fillna(0),
                                mode='lines+markers',
                                line_shape='spline',
                                name='Total Remaining Cases',
                                line=dict(color='#e36209', width=4),
                                marker=dict(size=4, color='#f4f4f2',
                                            line=dict(width=1,color='#e36209')),
                                text=[datetime.strftime(d, '%b %d %Y AEDT') for d in df_deaths['Date'].fillna(pd.Timestamp('20200101'))],
                                hovertext=['Total remaining<br>{:,d} cases<br>'.format(int(i)) for i in df_remaining['Total'].fillna(0)],
                                hovertemplate='<b>%{text}</b><br></br>'+
                                              '%{hovertext}'+
                                              '<extra></extra>'))

# Customise layout
fig_combine.update_layout(
#    title=dict(
#    text="<b>Confirmed Cases Timeline<b>",
#    y=0.96, x=0.5, xanchor='center', yanchor='top',
#    font=dict(size=20, color="#292929", family="Playfair Display")
#   ),
    margin=go.layout.Margin(
        l=10,
        r=10,
        b=10,
        t=5,
        pad=0
    ),
    yaxis=dict(
        showline=False, linecolor='#272e3e',
        zeroline=False,
        #showgrid=False,
        gridcolor='rgba(203, 210, 211,.3)',
        gridwidth = .1,
        tickmode='array',
        # Set tick range based on the maximum number
        tickvals=tickList,
        # Set tick label accordingly
        ticktext=["{:.0f}k".format(i/1000) for i in tickList]
    ),
#    yaxis_title="Total Confirmed Case Number",
    xaxis=dict(
        showline=False, linecolor='#272e3e',
        showgrid=False,
        gridcolor='rgba(203, 210, 211,.3)',
        gridwidth = .1,
        zeroline=False
    ),
    xaxis_tickformat='%b %d',
    hovermode = 'x',
    legend_orientation="h",
#    legend=dict(x=.35, y=-.05),
    plot_bgcolor='#f4f4f2',
    paper_bgcolor='#cbd2d3',
    font=dict(color='#292929')
)

# Line plot for death rate cases
# Set up tick scale based on confirmed case number
tickList = list(np.arange(0, (df_deaths['Other locations']/df_confirmed['Other locations']*100).max(), 0.5))

# Create empty figure canvas
fig_rate = go.Figure()
# Add trace to the figure
fig_rate.add_trace(go.Scatter(x=df_deaths['Date'], y=df_deaths['Mainland China']/df_confirmed['Mainland China']*100,
                                mode='lines+markers',
                                line_shape='spline',
                                name='Mainland China',
                                line=dict(color='#626262', width=4),
                                marker=dict(size=4, color='#f4f4f2',
                                            line=dict(width=1,color='#626262')),
                                text=[datetime.strftime(d, '%b %d %Y AEDT') for d in df_deaths['Date'].fillna(pd.Timestamp('20200101'))],
                                hovertext=['Mainland China death rate<br>{:.2f}%'.format(int(i)) for i in (df_deaths['Mainland China']/df_confirmed['Mainland China']).fillna(0)*100],
                                hovertemplate='<b>%{text}</b><br></br>'+
                                              '%{hovertext}'+
                                              '<extra></extra>'))
fig_rate.add_trace(go.Scatter(x=df_deaths['Date'], y=df_deaths['Other locations']/df_confirmed['Other locations']*100,
                                mode='lines+markers',
                                line_shape='spline',
                                name='Other Region',
                                line=dict(color='#a7a7a7', width=4),
                                marker=dict(size=4, color='#f4f4f2',
                                            line=dict(width=1,color='#a7a7a7')),
                                text=[datetime.strftime(d, '%b %d %Y AEDT') for d in df_deaths['Date'].fillna(pd.Timestamp('20200101'))],
                                hovertext=['Other region death rate<br>{:.2f}%'.format(int(i)) for i in (df_deaths['Other locations']/df_confirmed['Other locations']).fillna(0)*100],
                                hovertemplate='<b>%{text}</b><br></br>'+
                                              '%{hovertext}'+
                                              '<extra></extra>'))

# Customise layout
fig_rate.update_layout(
    margin=go.layout.Margin(
        l=10,
        r=10,
        b=10,
        t=5,
        pad=0
    ),
    yaxis=dict(
        showline=False, linecolor='#272e3e',
        zeroline=False,
        #showgrid=False,
        gridcolor='rgba(203, 210, 211,.3)',
        gridwidth = .1,
        tickmode='array',
        # Set tick range based on the maximum number
        tickvals=tickList,
        # Set tick label accordingly
        ticktext=['{:.1f}'.format(i) for i in tickList]
    ),
#    yaxis_title="Total Confirmed Case Number",
    xaxis=dict(
        showline=False, linecolor='#272e3e',
        showgrid=False,
        gridcolor='rgba(203, 210, 211,.3)',
        gridwidth = .1,
        zeroline=False
    ),
    xaxis_tickformat='%b %d',
    hovermode = 'x',
    legend_orientation="h",
#    legend=dict(x=.35, y=-.05),
    plot_bgcolor='#f4f4f2',
    paper_bgcolor='#cbd2d3',
    font=dict(color='#292929')
)


# # Read in the beds data to join into the UK table

# In[36]:


### TODO


# In[37]:


def build_df_region(Region):
    if Region == 'UK':
        Region = 'United Kingdom'
    df_region = pd.DataFrame(time_series_data_confirmed[time_series_data_confirmed['Country/Region'] == Region].transpose()[4:])
    if len(df_region.columns) > 1:
        s = df_region.sum().sort_values(ascending=False, inplace=False)
        df_region = pd.DataFrame(df_region[s[:1].index[0]].astype(int))
        df_region.columns = ['Confirmed']
        df_region_recovered = pd.DataFrame(time_series_data_recovered[time_series_data_confirmed['Country/Region'] == Region].transpose()[4:])
        s = df_region_recovered.sum().sort_values(ascending=False, inplace=False)
        df_region_recovered = pd.DataFrame(df_region_recovered[s[:1].index[0]].astype(int))
        df_region['Recovered'] = df_region_recovered
        df_region_deaths = pd.DataFrame(time_series_data_deaths[time_series_data_confirmed['Country/Region'] == Region].transpose()[4:])
        s = df_region_deaths.sum().sort_values(ascending=False, inplace=False)
        df_region_deaths = pd.DataFrame(df_region_deaths[s[:1].index[0]].astype(int))
        df_region['Deaths'] = df_region_deaths
    else:
        df_region.columns = ['Confirmed']
        df_region['Recovered'] = time_series_data_recovered[time_series_data_confirmed['Country/Region'] == Region].transpose()[4:]
        df_region['Deaths'] = time_series_data_deaths[time_series_data_confirmed['Country/Region'] == Region].transpose()[4:]
    df_region['New'] = df_region['Confirmed'].astype(int).diff().fillna(0)
    list_days = []
    for i in range(1,len(df_region['Confirmed'])+1):
        list_days.append(df_region['Confirmed'][:i].astype(bool).sum(axis=0))
    df_region['DayElapsed'] = list_days
    df_region['date_day'] = df_region.index
    date_list = []
    for date_str in df_region['date_day']:
        date_list.append(datetime.strptime(date_str, '%m/%d/%y'))
    df_region['Date_last_updated_AEDT'] = date_list
    df_region['Date_last_updated_AEDT'] = df_region['Date_last_updated_AEDT'] + timedelta(hours=16)
    df_region=df_region.astype({'Date_last_updated_AEDT':'datetime64', 'date_day':'datetime64'})
    return df_region


# In[38]:


df_UK[:]['TotalCases'].sum()


# In[ ]:


time_series_data_confirmed


# In[ ]:


Region = 'United Kingdom'
df_region = pd.DataFrame(time_series_data_confirmed[time_series_data_confirmed['Country/Region'] == Region].transpose()[4:])


# In[ ]:


def build_df_region_UK(Locality):
    Region = 'United Kingdom'
    #if Locality == 'United Kingdom':
    #    Locality = 'Derby'
    df_region = pd.DataFrame(time_series_data_confirmed[time_series_data_confirmed['Country/Region'] == Region].transpose()[4:])
    if len(df_region.columns) > 1:
        s = df_region.sum().sort_values(ascending=False, inplace=False)
        df_region = pd.DataFrame(df_region[s[:1].index[0]].astype(int))
        df_region.columns = ['Confirmed']
        df_region_recovered = pd.DataFrame(time_series_data_recovered[time_series_data_confirmed['Country/Region'] == Region].transpose()[4:])
        s = df_region_recovered.sum().sort_values(ascending=False, inplace=False)
        df_region_recovered = pd.DataFrame(df_region_recovered[s[:1].index[0]].astype(int))
        df_region['Recovered'] = df_region_recovered
        df_region_deaths = pd.DataFrame(time_series_data_deaths[time_series_data_confirmed['Country/Region'] == Region].transpose()[4:])
        s = df_region_deaths.sum().sort_values(ascending=False, inplace=False)
        df_region_deaths = pd.DataFrame(df_region_deaths[s[:1].index[0]].astype(int))
        df_region['Deaths'] = df_region_deaths
    else:
        df_region.columns = ['Confirmed']
        df_region['Recovered'] = time_series_data_recovered[time_series_data_confirmed['Country/Region'] == Region].transpose()[4:]
        df_region['Deaths'] = time_series_data_deaths[time_series_data_confirmed['Country/Region'] == Region].transpose()[4:]
    df_region['New'] = df_region['Confirmed'].astype(int).diff().fillna(0)
    
    list_days = []
    for i in range(1,len(df_region['Confirmed'])+1):
        list_days.append(df_region['Confirmed'][:i].astype(bool).sum(axis=0))
    df_region['DayElapsed'] = list_days
    df_region['date_day'] = df_region.index
    date_list = []
    for date_str in df_region['date_day']:
        date_list.append(datetime.strptime(date_str, '%m/%d/%y'))
    df_region['Date_last_updated_AEDT'] = date_list
    df_region['Date_last_updated_AEDT'] = df_region['Date_last_updated_AEDT'] + timedelta(hours=16)
    df_region=df_region.astype({'Date_last_updated_AEDT':'datetime64', 'date_day':'datetime64'})
    
    if Locality != 'United Kingdom':
        percentage_locality = df_UK[df_UK['GSS_NM']==Locality]['TotalCases']/df_UK[:]['TotalCases'].sum()
        df_region[['Confirmed','Recovered','Deaths','New']] = (float(percentage_locality) * df_region[['Confirmed','Recovered','Deaths','New']])[['Confirmed','Recovered','Deaths','New']].astype(int)
    # only allow real data values
    count = 0
    for bool_val in df_region['Confirmed']>0:
        if bool_val == True:
            break
        count = count + 1
    local_df = df_region[count:]
    return local_df


# In[ ]:


Locality = 'United Kingdom'
local_df = build_df_region_UK(Locality)


# In[ ]:


local_df


# In[ ]:


today = datetime.today().date()
count = 0
for datestr in df_region.index:
    date = datetime.strptime(datestr,"%m/%d/%y").date()
    print((date - today).days)
    if (date - today).days == 0:
        print(date)
        print(today)
        break
    count = count + 1


# In[ ]:


#local_df.ne(0).idxmax()['Confirmed']
count = 0
for bool_val in local_df.index==local_df.ne(0).idxmax()['Confirmed']:
    if bool_val == True:
        break
    count = count + 1
local_df[count:]


# In[ ]:

Region = 'UK'
if Region == 'UK':
    Region = 'United Kingdom'
CaseType = ['Confirmed', 'Recovered', 'Deaths']

df_region = build_df_region(Region)
df_region


# In[39]:


###UKTable = df_UK_merged[['GSS_NM','Org Name', 'TotalCases', 'Free Beds INCLUDING COVID-19', 'Predicted Free Beds INCLUDING COVID-19 in 1 WEEK']].sort_values(by=['Predicted Free Beds INCLUDING COVID-19 in 1 WEEK']).copy()
###UKTable[['Free Beds INCLUDING COVID-19', 'Predicted Free Beds INCLUDING COVID-19 in 1 WEEK']] = UKTable[['Free Beds INCLUDING COVID-19', 'Predicted Free Beds INCLUDING COVID-19 in 1 WEEK']].fillna(0).astype(int)

Locality = 'Derby'
if Locality != 'United Kingdom':
    percentage_locality = df_UK[df_UK['GSS_NM']==Locality]['TotalCases']/df_UK[:]['TotalCases'].sum()
    print(percentage_locality)
    #print(df_region[['Confirmed','Recovered','Deaths','New']])
    df_region[['Confirmed','Recovered','Deaths','New']] = (float(percentage_locality) * df_region[['Confirmed','Recovered','Deaths','New']])[['Confirmed','Recovered','Deaths','New']].astype(int)

# In[40]:


df_region


# In[41]:


UKTable = df_UK_merged[['GSS_NM','Org Name', 'TotalCases', 'Free Beds INCLUDING COVID-19', 'Predicted Free Beds INCLUDING COVID-19 in 1 WEEK','Free Beds Without COVID-19','count','lat_x', 'lon_x',]].sort_values(by=['Predicted Free Beds INCLUDING COVID-19 in 1 WEEK']).copy()
UKTable[['Free Beds INCLUDING COVID-19', 'Predicted Free Beds INCLUDING COVID-19 in 1 WEEK','Free Beds Without COVID-19']] = UKTable[['Free Beds INCLUDING COVID-19', 'Predicted Free Beds INCLUDING COVID-19 in 1 WEEK','Free Beds Without COVID-19']].fillna(0).astype(int)
UKTable.rename(columns={"Free Beds Without COVID-19" : "Q1 2019 Free Beds","Free Beds INCLUDING COVID-19" : "Q1 2019 Free Bed data with COVID-19 active cases applied at 8.2% Hospitalisation", "Predicted Free Beds INCLUDING COVID-19 in 1 WEEK":"Predicted Free Beds from Q1 2019 data with forecasted COVID-19 in One Week applied at 8.2% Hospitalisation", "lat_x": "lat", "lon_x": "lon", "count": "Number of Trusts in Region","GSS_NM": "Country/Region","Org Name": "Trust Name"}, inplace=True)

df_UK_merged


UKTable

# In[42]:

#### NEW get_ipython here
###get_ipython().run_cell_magic('time', '', '
# Function for generating cumulative line plot for each Country/Region

# Read cumulative data of a given region from ./cumulative_data folder
Region = 'UK'
CaseType = ['Confirmed', 'Recovered', 'Deaths']

df_region = build_df_region(Region)

#df_region = pd.read_csv('./cumulative_data/{}.csv'.format(Region))
#df_region=df_region.astype({'Date_last_updated_AEDT':'datetime64', 'date_day':'datetime64'})

# Line plot for confirmed cases
# Set up tick scale based on confirmed case number
#tickList = list(np.arange(0, df_confirmed['Mainland China'].max()+1000, 10000))

# Create empty figure canvas
fig = make_subplots(specs=[[{"secondary_y": True}]])
#fig = go.Figure()
#fig = make_subplots(specs=[[{"secondary_y": True}]])
# Add trace to the figure
fig.add_trace(go.Scatter(x=df_region['date_day'], 
                         y=df_region['Confirmed'],
                         mode='lines+markers',
                         #line_shape='spline',
                         name='Confirmed case',
                         line=dict(color='#d7191c', width=2),
                         #marker=dict(size=4, color='#f4f4f2',
                         #            line=dict(width=1,color='#921113')),
                         text=[datetime.strftime(d, '%b %d %Y AEDT') for d in df_region['date_day']],
                         hovertext=['{} confirmed<br>{:,d} cases<br>'.format(Region, i) for i in df_region['Confirmed']],
                         hovertemplate='<b>%{text}</b><br></br>'+
                                                 '%{hovertext}'+
                                                 '<extra></extra>'),
                         secondary_y=False,
             )
fig.add_trace(go.Scatter(x=df_region['date_day'], 
                         y=df_region['Recovered'],
                         mode='lines+markers',
                         #line_shape='spline',
                         name='Recovered case',
                         line=dict(color='#1a9622', width=2),                         
                         #marker=dict(size=4, color='#f4f4f2',
                         #            line=dict(width=1,color='#168038')),
                         text=[datetime.strftime(d, '%b %d %Y AEDT') for d in df_region['date_day']],
                         hovertext=['{} Recovered<br>{:,d} cases<br>'.format(Region, i) for i in df_region['Recovered']],
                         hovertemplate='<b>%{text}</b><br></br>'+
                                                 '%{hovertext}'+
                                                 '<extra></extra>'),
                         secondary_y=False,
             )
fig.add_trace(go.Scatter(x=df_region['date_day'], 
                         y=df_region['Deaths'],
                         mode='lines+markers',
                         #line_shape='spline',
                         name='Death case',
                         line=dict(color='#626262', width=2),
                         #marker=dict(size=4, color='#f4f4f2',
                         #            line=dict(width=1,color='#626262')),
                         text=[datetime.strftime(d, '%b %d %Y AEDT') for d in df_region['date_day']],
                         hovertext=['{} Deaths<br>{:,d} cases<br>'.format(Region, i) for i in df_region['Deaths']],
                         hovertemplate='<b>%{text}</b><br></br>'+
                                                 '%{hovertext}'+
                                                 '<extra></extra>'),
                         secondary_y=False,
             )

fig.add_trace(go.Bar(x=df_region['date_day'], 
                     y=df_region['New'],
                         #mode='lines+markers',
                         #line_shape='spline',
                     name='Daily New Cases',
                     marker_color='#626262',
                     opacity = .3,
                         #marker=dict(size=4, color='#f4f4f2',
                         #            line=dict(width=1,color='#626262')),
                     text=[datetime.strftime(d, '%b %d %Y AEDT') for d in df_region['date_day']],
                     hovertext=['{} New<br>{} cases<br>'.format(Region, i) for i in df_region['New']],
                     hovertemplate='<b>%{text}</b><br></br>'+
                                              '%{hovertext}'+
                                              '<extra></extra>'
                    ),
                     secondary_y=True,
             )

# Customise layout
fig.update_layout(
    #title=dict(
    #    text="<b>Confirmed Cases Timeline<b>",
    #    y=0.96, x=0.5, xanchor='center', yanchor='top',
    #    font=dict(size=20, color="#292929", family="Playfair Display")
    #),
    margin=go.layout.Margin(
            l=0,
            r=10,
            b=0,
            t=10,
            pad=0
        ),
    annotations=[
        dict(x=.5,
             y=.4,
             xref="paper",
             yref="paper",
             text=Region,
             opacity=0.5,
             font=dict(family='Arial, sans-serif',
                       size=60,
                       color="grey"),
        )
    ],
    yaxis=dict(showline=False, linecolor='#272e3e',
               zeroline=False,
               rangemode="tozero",
               #automargin=True,
               #showgrid=False,
               gridcolor='rgba(203, 210, 211,.3)',
               gridwidth = .1,
               mirror=True,
               #tickmode='array',
               # Set tick range based on the maximum number
               #tickvals=tickList,
               # Set tick label accordingly
               #ticktext=["{:.0f}k".format(i/1000) for i in tickList]
    ),
    yaxis2=dict(showline=False, linecolor='#272e3e',
                zeroline=False,
                #showgrid=False,
                #automargin=True,
                #range=[-1, max(df_region['New'])+100],
                #fig = make_subplots(specs=[[{"secondary_y": True}]])
                rangemode="tozero",
                mirror=True,
                gridcolor='rgba(203, 210, 211,.3)',
                gridwidth = .1,
                #tickmode='array',
                #anchor="free",
                #overlaying="y",
                #side="right",
                #position=0.15
    ),
    xaxis_title="Cumulative Cases (Select Country/Region From Table)",
    xaxis=dict(showline=False, linecolor='#272e3e',
               showgrid=False,
               gridcolor='rgba(203, 210, 211,.3)',
               gridwidth = .1,
               zeroline=False,
               automargin=True,
    ),
    xaxis_tickformat='%b %d',
    #transition = {'duration':500},
    hovermode = 'x',
    legend_orientation="v",
    legend=dict(x=.02, y=.95, bgcolor="rgba(0,0,0,0)",),
    plot_bgcolor='#f4f4f2',
    paper_bgcolor='#cbd2d3',
    font=dict(color='#292929')
    )

fig.show()

def logistic_model(x,a,b,c):
    return c/(1+np.exp(-(x-b)/a))

def exponential_model(x,a,b,c):
    return a*np.exp(b*(x-c))


# In[43]:


derived_virtual_selected_rows = ['United Kingdom']
selected_row_ids = [6]

dff = dfSum

mapbox_access_token = "pk.eyJ1IjoicGxvdGx5bWFwYm94IiwiYSI6ImNqdnBvNDMyaTAxYzkzeW5ubWdpZ2VjbmMifQ.TXcBE-xg9BFdV2ocecc_7g"

# Generate a list for hover text display
textList=[]
for area, region in zip(dfs[keyList[0]]['Province/State'], dfs[keyList[0]]['Country/Region']):
    region = str(region)
    if type(area) is str:
        if region == "Hong Kong" or region == "Macau" or region == "Taiwan":
            textList.append(area)
        else:
            #print(region)
            textList.append(area+', '+region)
    else:
        textList.append(region)
# Generate a list for color gradient display
colorList=[]

for comfirmed, recovered, deaths in zip(dfs[keyList[0]]['Confirmed'],dfs[keyList[0]]['Recovered'],dfs[keyList[0]]['Deaths']):
    remaining = comfirmed - deaths - recovered
    colorList.append(remaining)


fig2 = go.Figure(go.Scattermapbox(
    lat=dfs[keyList[0]]['lat'],
    lon=dfs[keyList[0]]['lon'],
    mode='markers',
    marker=go.scattermapbox.Marker(
        color=['#d7191c' if i > 0 else '#1a9622' for i in colorList],
        size=[i**(1/3) for i in dfs[keyList[0]]['Confirmed']], 
        sizemin=1,
        sizemode='area',
        sizeref=2.*max([math.sqrt(i) for i in dfs[keyList[0]]['Confirmed']])/(100.**2),
    ),
    text=textList,
    hovertext=['Confirmed: {}<br>Recovered: {}<br>Death: {}'.format(i, j, k) for i, j, k in zip(dfs[keyList[0]]['Confirmed'],
                                                                                                dfs[keyList[0]]['Recovered'],
                                                                                                dfs[keyList[0]]['Deaths'])],
    hovertemplate = "<b>%{text}</b><br><br>" +
                    "%{hovertext}<br>" +
                    "<extra></extra>")
)
fig2.update_layout(
    plot_bgcolor='#151920',
    paper_bgcolor='#cbd2d3',
    margin=go.layout.Margin(l=10,r=10,b=10,t=0,pad=40),
    hovermode='closest',
    transition = {'duration':50},
    annotations=[
    dict(
        x=.5,
        y=-.01,
        align='center',
        showarrow=False,
        text="Points are placed based on data geolocation levels.<br><b>Province/State level<b> - China, Australia, United States, and Canada; <b>Country level<b> - other countries.",
        xref="paper",
        yref="paper",
        font=dict(size=10, color='#292929'),
    )],
    mapbox=go.layout.Mapbox(
        accesstoken=mapbox_access_token,
        style="light",
        # The direction you're facing, measured clockwise as an angle from true north on a compass
        bearing=0,
        center=go.layout.mapbox.Center(
            lat=14.056159 if len(derived_virtual_selected_rows)==0 else dff.iloc[selected_row_ids[0]].lat, 
            lon=22.920039 if len(derived_virtual_selected_rows)==0 else dff.iloc[selected_row_ids[0]].lon
        ),
        pitch=0,
        zoom=1.03 if len(derived_virtual_selected_rows)==0 else 4
    )
)
fig2.show()


# In[48]:


##################################################################################################
#### Start dash app
##################################################################################################

app = dash.Dash('forecasting_dashboard', 
                assets_folder=my_path+'/assets/',
                meta_tags=[
                    {"name": "author", "content": "Jonathan McKinnell and Jun Ye"},
                    {"name": "description", "content": "The coronavirus COVID-19 monitor provides up-to-date data for the global spread of coronavirus."},
                    {"property": "og:title", "content": "Coronavirus COVID-19 Outbreak Global Cases Monitor"},
                    {"name": "viewport", "content": "width=device-width, height=device-height, initial-scale=1.0"}
                ]
      )

app.title = 'Coronavirus COVID-19 Global Monitor'
server = app.server

app.layout = html.Div(style={'backgroundColor':'#f4f4f2'},
    children=[
        html.Div(
            id="header",
            children=[                          
                html.H4(children="Coronavirus (COVID-19) Outbreak Global Cases Monitor"),
                html.P(
                    id="description",
                    children="On Dec 31, 2019, the World Health Organization (WHO) was informed of \
                    an outbreak of “pneumonia of unknown cause” detected in Wuhan City, Hubei Province, China – the \
                    seventh-largest city in China with 11 million residents. As of {}, there are over {:,d} cases \
                    of COVID-19 confirmed globally. This dash board is developed to visualise and track the recent reported \
                    cases on an hourly timescale. New feature added - click on UK tab in Cases by Country and you can see the Free Bed forecasting for NHS Trusts in England. \
                    For the UK free bed forecasting NHS Q1 2019 data was used as a baseline before forecasting the number of COVID-19 cases from the current d ata in the UK \
                    then trying to predict the current number of free beds in England NHS trusts \
                    and then forecast the current number of beds in 1 week based on an Imperial College study of an estimated 8.2% hospitalisation rate \
                    Open Source code available in python: https://github.com/jonathanmckinnell/COVID-19-forecasting-dashboard".format(latestDate, int(confirmedCases)),
                ),
                html.P(
                  id='time-stamp',
                       children="🔴 Last updated on {}. (Sorry, the app server may experince short period of interruption while updating data)".format(latestDate))
                    ]        
                ),
        html.Div(
            id="number-plate",
            style={'marginLeft':'1.5%','marginRight':'1.5%','marginBottom':'.5%'},
                 children=[
                     html.Div(
                         style={'width':'24.4%','backgroundColor':'#cbd2d3','display':'inline-block',
                                'marginRight':'.8%','verticalAlign':'top'},
                              children=[
                                  html.H3(style={'textAlign':'center',
                                                 'fontWeight':'bold','color':'#2674f6'},
                                               children=[
                                                   html.P(style={'color':'#cbd2d3','padding':'.5rem'},
                                                              children='xxxx xx xxx xxxx xxx xxxxx'),
                                                   '{}'.format(daysOutbreak),
                                               ]),
                                  html.H5(style={'textAlign':'center','color':'#2674f6','padding':'.1rem'},
                                               children="Days Since Outbreak")                                        
                                       ]),
                     html.Div(
                         style={'width':'24.4%','backgroundColor':'#cbd2d3','display':'inline-block',
                                'marginRight':'.8%','verticalAlign':'top'},
                              children=[
                                  html.H3(style={'textAlign':'center',
                                                 'fontWeight':'bold','color':'#d7191c'},
                                                children=[
                                                    html.P(style={'padding':'.5rem'},
                                                              children='+ {:,d} in the past 24h ({:.1%})'.format(int(plusConfirmedNum), plusPercentNum1)),
                                                    '{:,d}'.format(int(confirmedCases))
                                                         ]),
                                  html.H5(style={'textAlign':'center','color':'#d7191c','padding':'.1rem'},
                                               children="Confirmed Cases")                                        
                                       ]),
                     html.Div(
                         style={'width':'24.4%','backgroundColor':'#cbd2d3','display':'inline-block',
                                'marginRight':'.8%','verticalAlign':'top'},
                              children=[
                                  html.H3(style={'textAlign':'center',
                                                       'fontWeight':'bold','color':'#1a9622'},
                                               children=[
                                                   html.P(style={'padding':'.5rem'},
                                                              children='+ {:,d} in the past 24h ({:.1%})'.format(plusRecoveredNum, plusPercentNum2)),
                                                   '{:,d}'.format(int(recoveredCases)),
                                               ]),
                                  html.H5(style={'textAlign':'center','color':'#1a9622','padding':'.1rem'},
                                               children="Recovered Cases")                                        
                                       ]),
                     html.Div(
                         style={'width':'24.4%','backgroundColor':'#cbd2d3','display':'inline-block',
                                'verticalAlign':'top'},
                              children=[
                                  html.H3(style={'textAlign':'center',
                                                       'fontWeight':'bold','color':'#6c6c6c'},
                                                children=[
                                                    html.P(style={'padding':'.5rem'},
                                                              children='+ {:,d} in the past 24h ({:.1%})'.format(plusDeathNum, plusPercentNum3)),
                                                    '{:,d}'.format(int(deathsCases))
                                                ]),
                                  html.H5(style={'textAlign':'center','color':'#6c6c6c','padding':'.1rem'},
                                               children="Death Cases")                                        
                                       ])
                          ]),
        html.Div(
            id='dcc-plot',
            style={'marginLeft':'1.5%','marginRight':'1.5%','marginBottom':'.35%','marginTop':'.5%'},
                 children=[
                     html.Div(
                         style={'width':'32.79%','display':'inline-block','marginRight':'.8%','verticalAlign':'top'},
                              children=[
                                  html.H5(style={'textAlign':'center','backgroundColor':'#cbd2d3',
                                                 'color':'#292929','padding':'1rem','marginBottom':'0'},
                                               children='Confirmed Case Timeline'),
                                  dcc.Graph(style={'height':'300px'},figure=fig_confirmed)]),
                     html.Div(
                         style={'width':'32.79%','display':'inline-block','marginRight':'.8%','verticalAlign':'top'},
                              children=[
                                  html.H5(style={'textAlign':'center','backgroundColor':'#cbd2d3',
                                                 'color':'#292929','padding':'1rem','marginBottom':'0'},
                                               children='Recovered/Death Case Timeline'),
                                  dcc.Graph(style={'height':'300px'},figure=fig_combine)]),
                     html.Div(
                         style={'width':'32.79%','display':'inline-block','verticalAlign':'top'},
                              children=[
                                  html.H5(style={'textAlign':'center','backgroundColor':'#cbd2d3',
                                                 'color':'#292929','padding':'1rem','marginBottom':'0'},
                                               children='Death Rate (%) Timeline'),
                                  dcc.Graph(style={'height':'300px'},figure=fig_rate)])]),
        html.Div(
            id='dcc-map',
            style={'marginLeft':'1.5%','marginRight':'1.5%','marginBottom':'.5%'},
                 children=[
                     html.Div(style={'width':'66.41%','marginRight':'.8%','display':'inline-block','verticalAlign':'top'},
                              children=[
                                  html.H5(style={'textAlign':'center','backgroundColor':'#cbd2d3',
                                                 'color':'#292929','padding':'1rem','marginBottom':'0'},
                                               children='Latest Coronavirus Outbreak Map'),
                                  dcc.Graph(
                                      id='datatable-interact-map',
                                      style={'height':'500px'},),
                                  dcc.Graph(
                                      id='datatable-interact-lineplot',
                                      style={'height':'300px'},),
                                  dcc.Graph(
                                      id='datatable-interact-lineplot-uk',
                                      style={'height':'300px'},),
                              ]),
                     html.Div(style={'width':'32.79%','display':'inline-block','verticalAlign':'top'},
                              children=[
                                  html.H5(style={'textAlign':'center','backgroundColor':'#cbd2d3',
                                                 'color':'#292929','padding':'1rem','marginBottom':'0'},
                                               children='Cases by Country/Region'),
                                  dcc.Tabs(
                                        value='tab-1',
                                        parent_className='custom-tabs',
                                        className='custom-tabs-container',
                                        children=[
                                            dcc.Tab(label='The World',
                                                    value='tab-1',
                                                className='custom-tab',
                                                selected_className='custom-tab--selected',
                                                children=[                                                    
                                                    dash_table.DataTable(
                                                        id='datatable-interact-location',
                                                        # Don't show coordinates
                                                        columns=[{"name": i, "id": i} for i in dfSum.columns[0:5]],
                                                        # But still store coordinates in the table for interactivity
                                                        data=dfSum.to_dict("rows"),
                                                        row_selectable="single",
                                                        sort_action="native",
                                                        style_as_list_view=True,
                                                        style_cell={'font_family':'Arial',
                                                                    'font_size':'1.2rem',
                                                                    'padding':'.1rem',
                                                                    'backgroundColor':'#f4f4f2',},
                                                        fixed_rows={'headers':True,'data':0},
                                                        style_table={'minHeight': '1050px', 
                                                                     'height': '1050px', 
                                                                     'maxHeight': '1050px'
                                                                    #'Height':'300px',
                                                                    #'overflowY':'scroll',
                                                                    #'overflowX':'scroll',
                                                                    },
                                                        style_header={'backgroundColor':'#f4f4f2',
                                                                      'fontWeight':'bold'},
                                                        style_cell_conditional=[{'if': {'column_id':'Country/Regions'},'width':'28%'},
                                                                                {'if': {'column_id':'Remaining'},'width':'18%'},
                                                                                {'if': {'column_id':'Confirmed'},'width':'18%'},
                                                                                {'if': {'column_id':'Recovered'},'width':'18%'},
                                                                                {'if': {'column_id':'Deaths'},'width':'18%'},
                                                                                {'if': {'column_id':'Confirmed'},'color':'#d7191c'},
                                                                                {'if': {'column_id':'Recovered'},'color':'#1a9622'},
                                                                                {'if': {'column_id':'Deaths'},'color':'#6c6c6c'},
                                                                                {'textAlign': 'center'}],
                                                    )
                                        ]),
                                        dcc.Tab(label='Australia',
                                                className='custom-tab',
                                                selected_className='custom-tab--selected',
                                                children=[
                                            dash_table.DataTable(
                                                #id='datatable-interact-location-aus',
                                                # Don't show coordinates
                                                columns=[{"name": i, "id": i} for i in AUSTable.columns[0:5]],
                                                # But still store coordinates in the table for interactivity
                                                data=AUSTable.to_dict("rows"),
                                                #row_selectable="single",
                                                sort_action="native",
                                                style_as_list_view=True,
                                                style_cell={'font_family':'Arial',
                                                            'font_size':'1.2rem',
                                                            'padding':'.1rem',
                                                            'backgroundColor':'#f4f4f2',},
                                                fixed_rows={'headers':True,'data':0},
                                                style_table={'minHeight': '1050px', 
                                                             'height': '1050px', 
                                                             'maxHeight': '1050px'
                                                            #'Height':'300px',
                                                            #'overflowY':'scroll',
                                                            #'overflowX':'scroll',
                                                            },
                                                style_header={'backgroundColor':'#f4f4f2',
                                                              'fontWeight':'bold'},
                                                style_cell_conditional=[{'if': {'column_id':'Province/State'},'width':'28%'},
                                                                        {'if': {'column_id':'Remaining'},'width':'18%'},
                                                                        {'if': {'column_id':'Confirmed'},'width':'18%'},
                                                                        {'if': {'column_id':'Recovered'},'width':'18%'},
                                                                        {'if': {'column_id':'Deaths'},'width':'18%'},
                                                                        {'if': {'column_id':'Confirmed'},'color':'#d7191c'},
                                                                        {'if': {'column_id':'Recovered'},'color':'#1a9622'},
                                                                        {'if': {'column_id':'Deaths'},'color':'#6c6c6c'},
                                                                        {'textAlign': 'center'}],
                                            )
                                        ]),
                                        dcc.Tab(label='UK', 
                                                className='custom-tab',
                                                value='tab-2',
                                                selected_className='custom-tab--selected',
                                                children=[
                                            dash_table.DataTable(
                                                id='datatable-interact-location-uk',
                                                # Don't show coordinates
                                                columns=[{"name": i, "id": i} for i in UKTable.columns[:-2]],
                                                # But still store coordinates in the table for interactivity
                                                data=UKTable.to_dict("rows"),
                                                row_selectable="single",
                                                sort_action="native",
                                                style_as_list_view=True,
                                                style_cell={'height': 'auto',
                                                            # all three widths are needed
                                                            'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                                                            'whiteSpace': 'normal',
                                                            'font_family':'Arial',
                                                            'font_size':'1.2rem',
                                                            'padding':'.1rem',
                                                            'backgroundColor':'#f4f4f2',},
                                                fixed_rows={'headers':True,'data':0},
                                                style_table={'minHeight': '1050px', 
                                                             'height': '1050px', 
                                                             'maxHeight': '1050px',
                                                             #'width':'500px',
                                                             #'minWidth': '100px',
                                                            #'overflowY':'scroll',
                                                             'overflowX':'scroll'
                                                            },
                                                style_header={'backgroundColor':'#f4f4f2',
                                                              'fontWeight':'bold'},
                                                style_cell_conditional=[{'if': {'column_id':'Province/State'},'width':'28%'},
                                                                        {'if': {'column_id':'Remaining'},'width':'18%'},
                                                                        {'if': {'column_id':'Confirmed'},'width':'18%'},
                                                                        {'if': {'column_id':'Recovered'},'width':'18%'},
                                                                        {'if': {'column_id':'Deaths'},'width':'18%'},
                                                                        {'if': {'column_id':'Confirmed'},'color':'#d7191c'},
                                                                        {'if': {'column_id':'Recovered'},'color':'#1a9622'},
                                                                        {'if': {'column_id':'Deaths'},'color':'#6c6c6c'},
                                                                        {'textAlign': 'center'}],
                                            )
                                        ]),
                                        dcc.Tab(label='Canada', 
                                                className='custom-tab',
                                                selected_className='custom-tab--selected',
                                                children=[
                                            dash_table.DataTable(
                                                
                                                # Don't show coordinates
                                                columns=[{"name": i, "id": i} for i in CANTable.columns[0:5]],
                                                # But still store coordinates in the table for interactivity
                                                data=CANTable.to_dict("rows"),
                                                #row_selectable="single",
                                                sort_action="native",
                                                style_as_list_view=True,
                                                style_cell={'font_family':'Arial',
                                                            'font_size':'1.2rem',
                                                            'padding':'.1rem',
                                                            'backgroundColor':'#f4f4f2',},
                                                fixed_rows={'headers':True,'data':0},
                                                style_table={'minHeight': '1050px', 
                                                             'height': '1050px', 
                                                             'maxHeight': '1050px'
                                                            #'Height':'300px',
                                                            #'overflowY':'scroll',
                                                            #'overflowX':'scroll',
                                                            },
                                                style_header={'backgroundColor':'#f4f4f2',
                                                              'fontWeight':'bold'},
                                                style_cell_conditional=[{'if': {'column_id':'Province/State'},'width':'28%'},
                                                                        {'if': {'column_id':'Remaining'},'width':'18%'},
                                                                        {'if': {'column_id':'Confirmed'},'width':'18%'},
                                                                        {'if': {'column_id':'Recovered'},'width':'18%'},
                                                                        {'if': {'column_id':'Deaths'},'width':'18%'},
                                                                        {'if': {'column_id':'Confirmed'},'color':'#d7191c'},
                                                                        {'if': {'column_id':'Recovered'},'color':'#1a9622'},
                                                                        {'if': {'column_id':'Deaths'},'color':'#6c6c6c'},
                                                                        {'textAlign': 'center'}],
                                            )
                                        ]),
                                      dcc.Tab(label='Mainland China', 
                                              className='custom-tab',
                                              selected_className='custom-tab--selected',
                                              children=[
                                              dash_table.DataTable(
                                                
                                                # Don't show coordinates
                                                columns=[{"name": i, "id": i} for i in CNTable.columns[0:5]],
                                                # But still store coordinates in the table for interactivity
                                                data=CNTable.to_dict("rows"),
                                                #row_selectable="single",
                                                sort_action="native",
                                                style_as_list_view=True,
                                                style_cell={'font_family':'Arial',
                                                            'font_size':'1.2rem',
                                                            'padding':'.1rem',
                                                            'backgroundColor':'#f4f4f2',},
                                                fixed_rows={'headers':True,'data':0},
                                                style_table={'minHeight': '1050px', 
                                                             'height': '1050px', 
                                                             'maxHeight': '1050px'
                                                            #'Height':'300px',
                                                            #'overflowY':'scroll',
                                                            #'overflowX':'scroll',
                                                            },
                                                style_header={'backgroundColor':'#f4f4f2',
                                                              'fontWeight':'bold'},
                                                style_cell_conditional=[{'if': {'column_id':'Province/State'},'width':'28%'},
                                                                        {'if': {'column_id':'Remaining'},'width':'18%'},
                                                                        {'if': {'column_id':'Confirmed'},'width':'18%'},
                                                                        {'if': {'column_id':'Recovered'},'width':'18%'},
                                                                        {'if': {'column_id':'Deaths'},'width':'18%'},
                                                                        {'if': {'column_id':'Confirmed'},'color':'#d7191c'},
                                                                        {'if': {'column_id':'Recovered'},'color':'#1a9622'},
                                                                        {'if': {'column_id':'Deaths'},'color':'#6c6c6c'},
                                                                        {'textAlign': 'center'}],
                                            )
                                        ]),
                                      dcc.Tab(label='United States', 
                                              className='custom-tab',
                                              selected_className='custom-tab--selected',
                                              children=[
                                              dash_table.DataTable(
                                                
                                                # Don't show coordinates
                                                columns=[{"name": i, "id": i} for i in USTable.columns[0:5]],
                                                # But still store coordinates in the table for interactivity
                                                data=USTable.to_dict("rows"),
                                                #row_selectable="single",
                                                sort_action="native",
                                                style_as_list_view=True,
                                                style_cell={'font_family':'Arial',
                                                            'font_size':'1.2rem',
                                                            'padding':'.1rem',
                                                            'backgroundColor':'#f4f4f2',},
                                                fixed_rows={'headers':True,'data':0},
                                                style_table={'minHeight': '1050px', 
                                                             'height': '1050px', 
                                                             'maxHeight': '1050px'
                                                            #'Height':'300px',
                                                            #'overflowY':'scroll',
                                                            #'overflowX':'scroll',
                                                            },
                                                style_header={'backgroundColor':'#f4f4f2',
                                                              'fontWeight':'bold'},
                                                style_cell_conditional=[{'if': {'column_id':'Province/State'},'width':'28%'},
                                                                        {'if': {'column_id':'Remaining'},'width':'18%'},
                                                                        {'if': {'column_id':'Confirmed'},'width':'18%'},
                                                                        {'if': {'column_id':'Recovered'},'width':'18%'},
                                                                        {'if': {'column_id':'Deaths'},'width':'18%'},
                                                                        {'if': {'column_id':'Confirmed'},'color':'#d7191c'},
                                                                        {'if': {'column_id':'Recovered'},'color':'#1a9622'},
                                                                        {'if': {'column_id':'Deaths'},'color':'#6c6c6c'},
                                                                        {'textAlign': 'center'}],
                                            )
                                        ]),]
                                    )
                              ])
                 ]),
        html.Div(
          id='my-footer',
          style={'marginLeft':'1.5%','marginRight':'1.5%'},
                 children=[
                     html.P(style={'textAlign':'center','margin':'auto'},
                            children=[" Case forecasting by Jonathan McKinnell in the UK |",
                                      " Dashboard developed by ",html.A('Jun', href='https://junye0798.com/')," in Sydney"])]),
        ])

@app.callback(
    Output('datatable-interact-map', 'figure'),
    [Input('datatable-interact-location', 'derived_virtual_selected_rows'),
     Input('datatable-interact-location', 'selected_row_ids')]
)

def update_figures(derived_virtual_selected_rows, selected_row_ids):
    print(selected_row_ids)
    print(derived_virtual_selected_rows)
    # When the table is first rendered, `derived_virtual_data` and
    # `derived_virtual_selected_rows` will be `None`. This is due to an
    # idiosyncracy in Dash (unsupplied properties are always None and Dash
    # calls the dependent callbacks when the component is first rendered).
    # So, if `rows` is `None`, then the component was just rendered
    # and its value will be the same as the component's dataframe.
    # Instead of setting `None` in here, you could also set
    # `derived_virtual_data=df.to_rows('dict')` when you initialize
    # the component.
        
    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows = []
        
    dff = dfSum
        
    mapbox_access_token = "pk.eyJ1IjoicGxvdGx5bWFwYm94IiwiYSI6ImNqdnBvNDMyaTAxYzkzeW5ubWdpZ2VjbmMifQ.TXcBE-xg9BFdV2ocecc_7g"

    # Generate a list for hover text display
    textList=[]
    for area, region in zip(dfs[keyList[0]]['Province/State'], dfs[keyList[0]]['Country/Region']):
        region = str(region)
        if type(area) is str:
            if region == "Hong Kong" or region == "Macau" or region == "Taiwan":
                textList.append(area)
            else:
                #print(region)
                textList.append(area+', '+region)
        else:
            textList.append(region)
            
    # Generate a list for color gradient display
    colorList=[]

    for comfirmed, recovered, deaths in zip(dfs[keyList[0]]['Confirmed'],dfs[keyList[0]]['Recovered'],dfs[keyList[0]]['Deaths']):
        remaining = comfirmed - deaths - recovered
        colorList.append(remaining)

    fig2 = go.Figure(go.Scattermapbox(
        lat=dfs[keyList[0]]['lat'],
        lon=dfs[keyList[0]]['lon'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            color=['#d7191c' if i > 0 else '#1a9622' for i in colorList],
            size=[i**(1/3) for i in dfs[keyList[0]]['Confirmed']], 
            sizemin=1,
            sizemode='area',
            sizeref=2.*max([math.sqrt(i) for i in dfs[keyList[0]]['Confirmed']])/(100.**2),
        ),
        text=textList,
        hovertext=['Confirmed: {}<br>Recovered: {}<br>Death: {}'.format(i, j, k) for i, j, k in zip(dfs[keyList[0]]['Confirmed'],
                                                                                                    dfs[keyList[0]]['Recovered'],
                                                                                                    dfs[keyList[0]]['Deaths'])],
        hovertemplate = "<b>%{text}</b><br><br>" +
                        "%{hovertext}<br>" +
                        "<extra></extra>")
    )
    fig2.update_layout(
        plot_bgcolor='#151920',
        paper_bgcolor='#cbd2d3',
        margin=go.layout.Margin(l=10,r=10,b=10,t=0,pad=40),
        hovermode='closest',
        transition = {'duration':50},
        annotations=[
        dict(
            x=.5,
            y=-.01,
            align='center',
            showarrow=False,
            text="Points are placed based on data geolocation levels.<br><b>Province/State level<b> - China, Australia, United States, and Canada; <b>Country level<b> - other countries.",
            xref="paper",
            yref="paper",
            font=dict(size=10, color='#292929'),
        )],
        mapbox=go.layout.Mapbox(
            accesstoken=mapbox_access_token,
            style="light",
            # The direction you're facing, measured clockwise as an angle from true north on a compass
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=14.056159 if len(derived_virtual_selected_rows)==0 else dff.loc[selected_row_ids[0]].lat, 
                lon=22.920039 if len(derived_virtual_selected_rows)==0 else dff.loc[selected_row_ids[0]].lon
            ),
            pitch=0,
            zoom=1.03 if len(derived_virtual_selected_rows)==0 else 4
        )
    )

    return fig2

@app.callback(
    Output('datatable-interact-lineplot', 'figure'),
    [Input('datatable-interact-location', 'derived_virtual_selected_rows'),
     Input('datatable-interact-location', 'selected_row_ids')]
)

def update_lineplot(derived_virtual_selected_rows, selected_row_ids):
    
    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows = []
        
    dff = dfSum
    
    if selected_row_ids:
        if dff.loc[selected_row_ids[0]]['Country/Region'] == 'Mainland China':
            Region = 'China'
        else:
            Region = dff.loc[selected_row_ids[0]]['Country/Region']
    else:
        Region = 'UK'
        
    if Region == "UK": Region = 'United Kingdom'
    # Read cumulative data of a given region from ./cumulative_data folder

    df_region = build_df_region(Region)

    #df_region = pd.read_csv('C:/Users/jmckinnell/Documents/Personal/COVID-19-research/data_pull/data-visualisation-scripts/dash-2019-coronavirus/cumulative_data/{}.csv'.format(Region))
    #df_region=df_region.astype({'Date_last_updated_AEDT':'datetime64', 'date_day':'datetime64'})

    df = df_region.loc[:,['date_day','Confirmed']]
    df.index = df['date_day']
    df = df.sort_index()
    FMT = '%Y-%m-%d %H:%M:%S'
    date = df['date_day']
    df['data'] = date.map(lambda x : (datetime.strptime(str(x), FMT) - datetime.strptime("2020-01-01 00:00:00", FMT)).days  )
    #We can use the curve_fit function of scipy library to estimate the parameter values and errors starting from the original data.
    x = df['data']
    y = df['Confirmed']
    fit = curve_fit(logistic_model,x,y,p0=[2,100,20000], maxfev = 20000)

    a = fit[0][0]
    b = fit[0][1]
    c = fit[0][2]

    sol = int(fsolve(lambda x : logistic_model(x,a,b,c) - int(c),b))

    dates = []
    for i_date in date:
        dates.append(i_date)

    base = pd.to_datetime(date[-1:].values[0])
    date_list_pred = [base + timedelta(days=i_x) for i_x in range(1,sol)]
    date_list = []
    for i_date in dates:
        date_list.append(i_date)
    for i_date in date_list_pred:
        date_list.append(i_date)
    y_logistic = [logistic_model(i,fit[0][0],fit[0][1],fit[0][2]) for i in range(0+min(x),len(date_list)+min(x))]
    #y_exp = [date_list, [exponential_model(i,exp_fit[0][0],exp_fit[0][1],exp_fit[0][2]) for i in range(0+min(x),len(date_list)+min(x))]]
    y_logistic = list(map(int, y_logistic))
    # Create empty figure canvas
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    #fig3 = go.Figure()

    # Add trace to the figure
    fig3.add_trace(go.Scatter(x=df_region['date_day'], 
                             y=df_region['Confirmed'],
                             mode='lines+markers',
                             #line_shape='spline',
                             name='Confirmed case',
                             line=dict(color='#d7191c', width=4),
                             marker=dict(size=4, color='#d7191c',
                                         line=dict(width=4,color='#d7191c')),
                             text=[datetime.strftime(d, '%b %d %Y AEDT') for d in df_region['date_day']],
                             hovertext=['{} confirmed<br>{:,d} cases<br>'.format(Region, i) for i in df_region['Confirmed']],
                             hovertemplate='<b>%{text}</b><br></br>'+
                                                     '%{hovertext}'+
                                                     '<extra></extra>'),
                             secondary_y=False,
                 )

    # Add forecast trace to the figure
    fig3.add_trace(go.Scatter(x=date_list, 
                             y=y_logistic,
                             mode='lines+markers',
                             #line_shape='spline',
                             name='Predicted Cases',
                             #line=dict(color='#272e3e', width=1),
                             marker=dict(size=2, color='#272e3e',
                                 line=dict(width=1,color='#272e3e')),
                             text=[datetime.strftime(d, '%b %d %Y AEDT') for d in date_list],
                             hovertext=['{} Predicted<br>{:,d} cases<br>'.format(Region, i) for i in y_logistic],
                             hovertemplate='<b>%{text}</b><br></br>'+
                                                     '%{hovertext}'+
                                                     '<extra></extra>'),
                             secondary_y=False,
                 )

    fig3.add_trace(go.Scatter(x=df_region['date_day'], 
                             y=df_region['Recovered'],
                             mode='lines+markers',
                             #line_shape='spline',
                             name='Recovered case',
                             line=dict(color='#1a9622', width=2),                         
                             #marker=dict(size=4, color='#f4f4f2',
                             #            line=dict(width=1,color='#168038')),
                             text=[datetime.strftime(d, '%b %d %Y AEDT') for d in df_region['date_day']],
                             hovertext=['{} Recovered<br>{:,d} cases<br>'.format(Region, i) for i in df_region['Recovered']],
                             hovertemplate='<b>%{text}</b><br></br>'+
                                                     '%{hovertext}'+
                                                     '<extra></extra>'),
                             secondary_y=False,
                 )
    fig3.add_trace(go.Scatter(x=df_region['date_day'], 
                             y=df_region['Deaths'],
                             mode='lines+markers',
                             #line_shape='spline',
                             name='Death case',
                             line=dict(color='#626262', width=2),
                             #marker=dict(size=4, color='#f4f4f2',
                             #            line=dict(width=1,color='#626262')),
                             text=[datetime.strftime(d, '%b %d %Y AEDT') for d in df_region['date_day']],
                             hovertext=['{} Deaths<br>{:,d} cases<br>'.format(Region, i) for i in df_region['Deaths']],
                             hovertemplate='<b>%{text}</b><br></br>'+
                                                     '%{hovertext}'+
                                                     '<extra></extra>'),
                             secondary_y=False,
                 )

    fig3.add_trace(go.Bar(x=df_region['date_day'], 
                         y=df_region['New'],
                             #mode='lines+markers',
                             #line_shape='spline',
                         name='Daily New Cases',
                         marker_color='#626262',
                         opacity = .3,
                             #marker=dict(size=4, color='#f4f4f2',
                             #            line=dict(width=1,color='#626262')),
                         text=[datetime.strftime(d, '%b %d %Y AEDT') for d in df_region['date_day']],
                         hovertext=['{} New<br>{} cases<br>'.format(Region, i) for i in df_region['New']],
                         hovertemplate='<b>%{text}</b><br></br>'+
                                                  '%{hovertext}'+
                                                  '<extra></extra>'
                        ),
                         secondary_y=True,
                 )

    # Customise layout
    fig3.update_layout(
        #title=dict(
        #    text="<b>Confirmed Cases Timeline<b>",
        #    y=0.96, x=0.5, xanchor='center', yanchor='top',
        #    font=dict(size=20, color="#292929", family="Playfair Display")
        #),
        margin=go.layout.Margin(
                l=0,
                r=10,
                b=0,
                t=10,
                pad=0
            ),
        annotations=[
            dict(x=.5,
                 y=.4,
                 xref="paper",
                 yref="paper",
                 text=Region,
                 opacity=0.5,
                 font=dict(family='Arial, sans-serif',
                           size=60,
                           color="grey"),
            )
        ],
        yaxis=dict(showline=False, linecolor='#272e3e',
                   zeroline=False,
                   rangemode="tozero",
                   #automargin=True,
                   #showgrid=False,
                   gridcolor='rgba(203, 210, 211,.3)',
                   gridwidth = .1,
                   mirror=True,
                   #tickmode='array',
                   # Set tick range based on the maximum number
                   #tickvals=tickList,
                   # Set tick label accordingly
                   #ticktext=["{:.0f}k".format(i/1000) for i in tickList]
        ),
        yaxis2=dict(showline=False, linecolor='#272e3e',
                    zeroline=False,
                    showgrid=False,
                    #automargin=True,
                    #range=[-1, max(df_region['New'])+100]
                    rangemode="tozero",
                    #mirror='ticks',
                    #gridcolor='rgba(203, 210, 211,.3)',
                    #gridwidth = .1,
                    #tickmode='array',
                    #anchor="free",
                    #overlaying="y",
                    #side="right",
                    #position=0.15
        ),
        xaxis_title="Cumulative Cases (Select Country/Region From Table)",
        xaxis=dict(showline=False, linecolor='#272e3e',
                   showgrid=False,
                   gridcolor='rgba(203, 210, 211,.3)',
                   gridwidth = .1,
                   zeroline=False,
                   automargin=True,
        ),
        xaxis_tickformat='%b %d',
        #transition = {'duration':500},
        hovermode = 'x',
        legend_orientation="v",
        legend=dict(x=.02, y=.95, bgcolor="rgba(0,0,0,0)",),
        plot_bgcolor='#f4f4f2',
        paper_bgcolor='#cbd2d3',
        font=dict(color='#292929')
        )

    return fig3

@app.callback(
    Output('datatable-interact-lineplot-uk', 'figure'),
    [Input('datatable-interact-location-uk', 'derived_virtual_selected_rows'),
     Input('datatable-interact-location-uk', 'selected_row_ids')]
)

def update_lineplot_uk(derived_virtual_selected_rows, selected_row_ids):
    print('Inside update_lineplot_uk ', derived_virtual_selected_rows, selected_row_ids)
    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows = []
    
    if derived_virtual_selected_rows != None:
        if len(derived_virtual_selected_rows) > 0:
                if derived_virtual_selected_rows[0] != None:
                    Region = UKTable.iloc[derived_virtual_selected_rows[0]]['Country/Region']
                    print(Region)
        else:
            Region = 'UK'
            print('default')
                
    else:
        Region = 'UK'
        print('default')
        
    if Region == "UK": Region = 'United Kingdom'
    # Read cumulative data of a given region from ./cumulative_data folder
    print(Region)
    df_region = build_df_region_UK(Region)

    df = df_region.loc[:,['date_day','Confirmed']]
    df.index = df['date_day']
    df = df.sort_index()
    FMT = '%Y-%m-%d %H:%M:%S'
    date = df['date_day']
    df['data'] = date.map(lambda x : (datetime.strptime(str(x), FMT) - datetime.strptime("2020-01-01 00:00:00", FMT)).days  )
    #We can use the curve_fit function of scipy library to estimate the parameter values and errors starting from the original data.
    x = df['data']
    y = df['Confirmed']
    fit = curve_fit(logistic_model,x,y,p0=[2,100,20000], maxfev = 20000)

    a = fit[0][0]
    b = fit[0][1]
    c = fit[0][2]

    sol = int(fsolve(lambda x : logistic_model(x,a,b,c) - int(c),b))

    dates = []
    for i_date in date:
        dates.append(i_date)

    base = pd.to_datetime(date[-1:].values[0])
    date_list_pred = [base + timedelta(days=i_x) for i_x in range(1,sol)]
    date_list = []
    for i_date in dates:
        date_list.append(i_date)
    for i_date in date_list_pred:
        date_list.append(i_date)
    y_logistic = [logistic_model(i,fit[0][0],fit[0][1],fit[0][2]) for i in range(0+min(x),len(date_list)+min(x))]
    #y_exp = [date_list, [exponential_model(i,exp_fit[0][0],exp_fit[0][1],exp_fit[0][2]) for i in range(0+min(x),len(date_list)+min(x))]]
    y_logistic = list(map(int, y_logistic))
    # Create empty figure canvas
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    #fig3 = go.Figure()

    # Add trace to the figure
    fig3.add_trace(go.Scatter(x=df_region['date_day'], 
                             y=df_region['Confirmed'],
                             mode='lines+markers',
                             #line_shape='spline',
                             name='UK Confirmed cases scaled to region',
                             line=dict(color='#d7191c', width=4),
                             marker=dict(size=4, color='#d7191c',
                                         line=dict(width=4,color='#d7191c')),
                             text=[datetime.strftime(d, '%b %d %Y AEDT') for d in df_region['date_day']],
                             hovertext=['{} confirmed<br>{:,d} cases<br>'.format(Region, i) for i in df_region['Confirmed']],
                             hovertemplate='<b>%{text}</b><br></br>'+
                                                     '%{hovertext}'+
                                                     '<extra></extra>'),
                             secondary_y=False,
                 )

    # Add forecast trace to the figure
    fig3.add_trace(go.Scatter(x=date_list, 
                             y=y_logistic,
                             mode='lines+markers',
                             #line_shape='spline',
                             name='UK Predicted Cases scaled to region',
                             #line=dict(color='#272e3e', width=1),
                             marker=dict(size=2, color='#272e3e',
                                 line=dict(width=1,color='#272e3e')),
                             text=[datetime.strftime(d, '%b %d %Y AEDT') for d in date_list],
                             hovertext=['{} Predicted<br>{:,d} cases<br>'.format(Region, i) for i in y_logistic],
                             hovertemplate='<b>%{text}</b><br></br>'+
                                                     '%{hovertext}'+
                                                     '<extra></extra>'),
                             secondary_y=False,
                 )

    fig3.add_trace(go.Scatter(x=df_region['date_day'], 
                             y=df_region['Recovered'],
                             mode='lines+markers',
                             #line_shape='spline',
                             name='UK Recovered cases scaled to region',
                             line=dict(color='#1a9622', width=2),                         
                             #marker=dict(size=4, color='#f4f4f2',
                             #            line=dict(width=1,color='#168038')),
                             text=[datetime.strftime(d, '%b %d %Y AEDT') for d in df_region['date_day']],
                             hovertext=['{} Recovered<br>{:,d} cases<br>'.format(Region, i) for i in df_region['Recovered']],
                             hovertemplate='<b>%{text}</b><br></br>'+
                                                     '%{hovertext}'+
                                                     '<extra></extra>'),
                             secondary_y=False,
                 )
    fig3.add_trace(go.Scatter(x=df_region['date_day'], 
                             y=df_region['Deaths'],
                             mode='lines+markers',
                             #line_shape='spline',
                             name='UK Death cases scaled to region',
                             line=dict(color='#626262', width=2),
                             #marker=dict(size=4, color='#f4f4f2',
                             #            line=dict(width=1,color='#626262')),
                             text=[datetime.strftime(d, '%b %d %Y AEDT') for d in df_region['date_day']],
                             hovertext=['{} Deaths<br>{:,d} cases<br>'.format(Region, i) for i in df_region['Deaths']],
                             hovertemplate='<b>%{text}</b><br></br>'+
                                                     '%{hovertext}'+
                                                     '<extra></extra>'),
                             secondary_y=False,
                 )

    fig3.add_trace(go.Bar(x=df_region['date_day'], 
                         y=df_region['New'],
                             #mode='lines+markers',
                             #line_shape='spline',
                         name='UK Daily New Cases scaled to region',
                         marker_color='#626262',
                         opacity = .3,
                             #marker=dict(size=4, color='#f4f4f2',
                             #            line=dict(width=1,color='#626262')),
                         text=[datetime.strftime(d, '%b %d %Y AEDT') for d in df_region['date_day']],
                         hovertext=['{} New<br>{} cases<br>'.format(Region, i) for i in df_region['New']],
                         hovertemplate='<b>%{text}</b><br></br>'+
                                                  '%{hovertext}'+
                                                  '<extra></extra>'
                        ),
                         secondary_y=True,
                 )

    # Customise layout
    fig3.update_layout(
        #title=dict(
        #    text="<b>Confirmed Cases Timeline<b>",
        #    y=0.96, x=0.5, xanchor='center', yanchor='top',
        #    font=dict(size=20, color="#292929", family="Playfair Display")
        #),
        margin=go.layout.Margin(
                l=0,
                r=10,
                b=0,
                t=10,
                pad=0
            ),
        annotations=[
            dict(x=.5,
                 y=.4,
                 xref="paper",
                 yref="paper",
                 text=Region,
                 opacity=0.5,
                 font=dict(family='Arial, sans-serif',
                           size=60,
                           color="grey"),
            )
        ],
        yaxis=dict(showline=False, linecolor='#272e3e',
                   zeroline=False,
                   rangemode="tozero",
                   #automargin=True,
                   #showgrid=False,
                   gridcolor='rgba(203, 210, 211,.3)',
                   gridwidth = .1,
                   mirror=True,
                   #tickmode='array',
                   # Set tick range based on the maximum number
                   #tickvals=tickList,
                   # Set tick label accordingly
                   #ticktext=["{:.0f}k".format(i/1000) for i in tickList]
        ),
        yaxis2=dict(showline=False, linecolor='#272e3e',
                    zeroline=False,
                    showgrid=False,
                    #automargin=True,
                    #range=[-1, max(df_region['New'])+100]
                    rangemode="tozero",
                    #mirror='ticks',
                    #gridcolor='rgba(203, 210, 211,.3)',
                    #gridwidth = .1,
                    #tickmode='array',
                    #anchor="free",
                    #overlaying="y",
                    #side="right",
                    #position=0.15
        ),
        xaxis_title="Cumulative Cases (Select Country/Region From Table)",
        xaxis=dict(showline=False, linecolor='#272e3e',
                   showgrid=False,
                   gridcolor='rgba(203, 210, 211,.3)',
                   gridwidth = .1,
                   zeroline=False,
                   automargin=True,
        ),
        xaxis_tickformat='%b %d',
        #transition = {'duration':500},
        hovermode = 'x',
        legend_orientation="v",
        legend=dict(x=.02, y=.95, bgcolor="rgba(0,0,0,0)",),
        plot_bgcolor='#f4f4f2',
        paper_bgcolor='#cbd2d3',
        font=dict(color='#292929')
        )

    return fig3

if __name__ == '__main__':
    app.run_server(debug=1, port=8882)

# In[49]:


print("here")


