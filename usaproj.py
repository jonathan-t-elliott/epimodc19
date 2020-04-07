# main changes from v1.0
# days since 100 cases instead of days from now

import array as arr
import math
import matplotlib.pyplot as plt
import numpy as np
import datetime
import json
import requests
import io
import pandas as pd
from scipy.signal import savgol_filter
from scipy.ndimage.filters import uniform_filter1d

# GET CURRENT DATA FROM CSSEGIS GitHub Source
baseURL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
baseURL2 = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
deaths_= requests.get(baseURL).content
cases_ = requests.get(baseURL2).content

df_deaths=pd.read_csv(io.StringIO(deaths_.decode('utf-8')))
df_cases = pd.read_csv(io.StringIO(cases_.decode('utf-8')))

usa = df_deaths.iloc[225,:] #find the US-specific historical deaths
usa_cases = df_cases.iloc[225,:]

x = 0;
#days = arr.array('f',[0])
#acc_deaths = arr.array('f',[usa[len(usa)-1]])
days = arr.array('f',[0-len(usa)+5])

acc_deaths = arr.array('f',[usa[4]])
acc_cases = arr.array('f', [usa_cases[4]])
# extract the (cummulative) number of deaths for each day
for i in range (5, usa.size):
    days.append(i-usa.size+1)
    acc_deaths.append(usa[i])
    acc_cases.append(usa_cases[i])

# calculate new deaths each day
acc_ndeaths = arr.array('f',[0])
for i in range (1, len(acc_deaths)):
    acc_ndeaths.append(acc_deaths[i]-acc_deaths[i-1])

result = np.where(np.array(acc_cases) > 100)   
day_100 = result[0][0]

latest_deaths = acc_deaths[len(acc_deaths)-1]; #most recent death statistic
latest_cases = acc_cases[len(acc_cases)-1];

today_100 = len(usa)-4-day_100; # how many days between the 100 cases mark and today.
days_adj = (np.array(days)+(today_100)).tolist()

print(today_100)

# Maybe do some basic fitting here for the deaths?

plt.rcParams['figure.figsize'] = [15, 7]
fig, (ax10) = plt.subplots(1,1)

# Subplot 1 is number of cases in SIR model.
ax10.plot(days_adj, np.array(acc_deaths)*200, label='Deaths')
ax10.plot(days_adj, acc_cases, label='Cases')
ax10.legend()

print(acc_cases[len(acc_cases)-1])
print(acc_deaths[len(acc_cases)-1]*50)


# SET UP MY SIR MODEL
# This is a basic epidemiological compartment model that doesn't include demographic weightings (will soon)
# but does include the following features:
#  - estimate of infected people based on death rate extrapolation (important when cases are underreported due
#    to testing strategy only for symptomatic individuals or close contacts
#  - time-varrying suppression/mitigation of R (reproductive value) as ratio of R0, which reflects current and
#    projected likely policy shifts (based on measures required in other countries to reduce R)
#  - DYNAMIC Case Fatality Rate - CFR changes as a function of surge capacity of hospital, based on current numbers.
#    provides upper limit on predicted deaths.
#  - sorts infected populatino into asymptomatic, mild/moderate, and severe. Assumes severe need hospitilization.

pop = 330414717
sus = 0.70

# THIS IS A VERY IMPORTANT PARAMETER - HOW MANY ACTUAL CASES ARE THERE FOR EVERYONE CURRENTLY DYING.
# will differ from actual cases reported, if there is exponential growth and/or if testing is inefficient.
test_eff = 0.35

# initialize population stocks
N = pop*sus
I = arr.array('f',[latest_cases/test_eff])
R = arr.array('f', [latest_cases/test_eff/20])
D = arr.array('f',[latest_deaths])

S = arr.array('f',[N-I[0]-D[0]-R[0]])
D_std = arr.array('f',[ D[0] ])

time = arr.array('f',[days_adj[len(days_adj)-1]])
no_days = 300

I1 = arr.array('f', [I[0] * 0.20])
I2 = arr.array('f', [I[0] * 0.75])
I3 = arr.array('f', [I[0] * 0.05])

# epidemilogical constants
R0 = 2.3
gamma = 1 / 14
beta = R0 * gamma

CMR = 0.005
CMR_max = 0.05
rec_CMR = arr.array('f', [CMR])

h_beds = 100000
beds_left = arr.array('f',[1])
beds_left[0] = h_beds

# suppression efficacy
lv1 = 0
lv2 = 0
lv3 = 4

r1 = 0.70
r2 = 0.30
r3 = 0.20

# initialize pi to the current suppression/mitigation level
if (lv1 == 0):
    if(lv2 == 0):
        if(lv3 == 0): 
            pi = arr.array('f',[r3])
        else:
            pi = arr.array('f',[r2])
    else:
        pi = arr.array('f', [r1])
else:
    pi = arr.array('f',[1])
    
# determine the pi vector (mitigation/suppression over time)
for i in range (1, no_days):
    if (i > lv3 ):
        pi.append( r3 )
    elif (i > lv2 ):
        pi.append( r2 )
    elif (i > lv1):
        pi.append( r1 )
    else:
        pi.append( 1.0 ) 

# smooth since it requires some time
#pi_sm = savgol_filter(pi, 7, 2)
pi_sm = uniform_filter1d(pi, size=5)

# Run the simulation
for i in range (1, no_days):
    time.append(time[0]+i) # time vector
    
    # S and I compartments for the SIR model
    S.append(S[i-1] - (pi_sm[i] * beta * S[i-1] * I[i-1] / N))
    I.append(I[i-1] + (pi_sm[i] * beta * S[i-1] * I[i-1] / N) - (1 - CMR) * gamma * I[i-1])

    # Separate into 
    I1.append( I[i] * 0.20 )
    I2.append( I[i] * 0.75 ) 
    I3.append( I[i] * 0.05 ) #ICU

    beds_left.append( h_beds - I3[i])

    if (beds_left[i] < 0):
        CMR_d = CMR_max
    else:
        #CMR_d = (( (CMR_max - CMR) * (1 - beds_left[i] / h_beds) ) ** 2) / 0.025 + CMR;
        CMR_d = (CMR_max - CMR) / (1 + math.exp(-5 * (0.5-beds_left[i]/h_beds))) + CMR;
        
        #=1/(1+EXP(-5*(I2-0)))*($E$3-$E$2)+$E$2
    rec_CMR.append(CMR_d)

    R.append( R[i - 1] + (1 - CMR_d) * gamma * I[i - 1] )
    D.append( D[i - 1] + (CMR_d) * gamma * I[i - 1] )
    
    D_std.append( D_std[i - 1] + CMR * gamma * I[i - 1] )

# calculate the simulated number of new deaths each day.
nDeaths = arr.array('f',[D[0] - acc_deaths[len(acc_deaths)-1]] )
nDeaths_std = arr.array('f',[D_std[0] - acc_deaths[len(acc_deaths)-1]] )

for i in range (0, no_days-1):
    nDeaths.append(D[i+1]-D[i])
    nDeaths_std.append(D_std[i+1]-D_std[i])
    
# Some key facts:
print("Latest Est. Active Cases: " + str(I[0]))
print("Latest Death Count: " + str(latest_deaths))
print("New Deaths Today: " + str(acc_ndeaths[len(acc_ndeaths)-1]))
    
    
## PLOT RESULTS OF SIMULATION
plt.rcParams['figure.figsize'] = [15, 7]
fig, (ax1, ax2, ax3) = plt.subplots(1,3)

# Subplot 1 is number of cases in SIR model.
ax1.plot(time, I, label='Infected')
  #ax1.plot(time, S, label='Suseptible')
ax1.plot(time, R, label='Recovered')
ax1.plot(time, D, label='Deceased')
ax1.set_xlabel('Days from Today')
ax1.set_ylabel('No. of Cases')
ax1.legend()
#ax.set_xlabel()

# Subplot 2 is number of cases of different severity
ax2.semilogy(time, I1, label='Asymptomatic')
ax2.semilogy(time, I2, label='Mild/Moderate')
ax2.semilogy(time, I3, label='Severe')
ax2.semilogy([time[0], time[no_days-1]],[h_beds, h_beds])
ax2.set_xlabel('Days from Today')
ax2.set_ylabel('No. of Cases')
ax2.legend()

#semilogy(time,I1,time,I2,time,I3,[time(1) time(end)],[h_beds h_beds],'k--');
#xlabel('Days from Today')
#ylabel('No. of Cases');
#legend('Asymptom.','mild/mod','severe')

ax3.plot(time,D,label='Fatalities with variable CMR')
ax3.plot(time,D_std,label='Fatalities with constant CMR')
ax3.plot(time, np.divide(rec_CMR, float(5*10.0**(-8))), label='Variable CMR')
ax3.set_xlabel('Days from Today')
ax3.set_ylabel('No. of Fatalities')
ax3.legend()

#limits 0 4*10**5

def count2cmr(d):
    return d * 5*10**(-8)

def cmr2count(d):
    return d / (5*10**(-8))

ax3b = ax3.secondary_yaxis('right', functions=(count2cmr,cmr2count))
ax3b.set_xlabel('CMR (%)')


# MORE PLOTTING
plt.rcParams['figure.figsize'] = [10, 15]

fig, ((ax2,ax4,ax1)) = plt.subplots(3,1)
#ax1.scatter(days_adj, acc_deaths,label='Actual Data')
#ax1.plot(time, D, label='Projected')
#ax1.set_xlim(today_100-21,today_100+21)
#ax1.set_ylim(0, max(D[0:60]))
#ax1.set_xlabel('Days since 100 cases')
#ax1.set_ylabel('Cumulative Deaths')
#ax1.legend()

ax2.bar(days_adj, acc_ndeaths, color='k', label='Actual Historical Data')
ax2.bar(time[1:],((np.array(nDeaths[1:]) + np.array(nDeaths_std[1:]))/2.0),color='r',alpha=0.5, label='Projected Data')
ax2.fill_between(time[2:], nDeaths_std[2:], nDeaths[2:], alpha=0.2,facecolor='red')
ax2.set_xlim(today_100-21,today_100+21)
ax2.set_ylim(0,max(nDeaths[0:30])*1.1)
ax2.set_xlabel('Days from Today')
ax2.set_ylabel('New Deaths per Day')
ax2.grid()
ax2.legend()

ax3.plot(time, np.array(pi_sm) * R0)
ax3.set_xlim(0,30)
ax3.set_xlabel('Days since 100 cases')
ax3.set_ylim(0, 2.2)
ax3.set_ylabel('Reproductive Number (R) [R0 = 2.2]')

ax4.plot(days_adj, acc_deaths,label='Total Deaths', color='orange')
ax4.plot(time,((np.array(D) + np.array(D_std))/2.0), linestyle='--', color='orange', label='Total Deaths (projected)')
#ax4.plot(time,D,label='Fatalities with constant CMR')
#ax4.plot(time,D_std,label='Fatalities with constant CMR')
ax4.fill_between(time, D_std, D, alpha=0.2,facecolor='gold')
#ax4.plot(time, np.divide(rec_CMR, float(5*10.0**(-8))), label='Variable CMR')
ax4.set_xlim(0,150)
ax4.set_xlabel('Days since 100 cases')
ax4.set_ylabel('Predicted Deaths (Cumulative)')

ax1.scatter(days_adj, acc_deaths,label='Total Deaths', color='orange')
ax1.semilogy(time[1:21],((np.array(D[1:21]) + np.array(D_std[1:21]))/2.0), linestyle='--', color='orange', label='Total Deaths (projected)')
#ax4.plot(time,D,label='Fatalities with constant CMR')
#ax4.plot(time,D_std,label='Fatalities with constant CMR')
ax1.fill_between(time[1:21], D_std[1:21], D[1:21], alpha=0.2,facecolor='gold')
#ax4.plot(time, np.divide(rec_CMR, float(5*10.0**(-8))), label='Variable CMR')
ax1.set_xlim(today_100-21,today_100+21)
ax1.set_ylim(1, max(D[today_100-21:today_100+21]))
ax1.set_xlabel('Days since 100 cases')
ax1.set_ylabel('Predicted Deaths (Cumulative)')



