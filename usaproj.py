import array as arr
import matplotlib.pyplot as plt
import numpy as np
import datetime
import json
import requests
import io
import pandas as pd
from scipy.signal import savgol_filter

# ADD CURRENT DATA

#baseURL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/" 
#baseURL = "https://github.com/Omaroid/Covid-19-API/blob/master/data.json"
baseURL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
deaths_= requests.get(baseURL).content
df_deaths=pd.read_csv(io.StringIO(deaths_.decode('utf-8')))
usa = df_deaths.iloc[225,:] #find the US-specific historical deaths

print(len(usa))

x = 0;
days = arr.array('f',[0])
acc_deaths = arr.array('f',[usa[len(usa)-1]])

for i in range (4, usa.size):
    days.append(i-usa.size+1)
    acc_deaths.append(usa[i])

acc_ndeaths = arr.array('f',[0])
for i in range (1, len(acc_deaths)):
    acc_ndeaths.append(acc_deaths[i]-acc_deaths[i-1])
    
latest_deaths = acc_deaths[len(acc_deaths)-1];




# SET UP MY SIR MODEL
pop = 330414717
sus = 0.20

# THIS IS A VERY IMPORTANT PARAMETER - HOW MANY ACTUAL CASES ARE THERE FOR EVERYONE CURRENTLY DYING.
# will differ from actual cases reported, if there is exponential growth and/or if testing is inefficient.
test_eff = 200

N = pop*sus

I = arr.array('f',[latest_deaths*test_eff])
R = arr.array('f', [latest_deaths*test_eff/2])
D = arr.array('f',[latest_deaths])

S = arr.array('f',[N-I[0]-D[0]-R[0]])

D_std = arr.array('f',[ D[0] ])
time = arr.array('f',[0])

no_days = 300

I1 = arr.array('f', [I[0] * 0.20])
I2 = arr.array('f', [I[0] * 0.75])
I3 = arr.array('f', [I[0] * 0.05])

# epidemilogical constants
R0 = 2.2
gamma = 1 / 12
beta = R0 * gamma

CMR = 0.01
CMR_max = 0.05
rec_CMR = arr.array('f', [CMR])

h_beds = 100000
beds_left = arr.array('f',[1])
beds_left[0] = h_beds

# suppression efficacy
lv1 = 0
lv2 = 7
lv3 = 14

r1 = 0.85
r2 = 0.60
r3 = 0.35

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
pi_sm = savgol_filter(pi, 101, 2)

# Run the simulation
for i in range (1, no_days):
    time.append(i) # time vector
    
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
        CMR_d = (( (CMR_max - CMR) * (1 - beds_left[i] / h_beds) ) ** 2) / 0.025 + CMR;

    rec_CMR.append(CMR_d)

    R.append( R[i - 1] + (1 - CMR_d) * gamma * I[i - 1] )
    D.append( D[i - 1] + (CMR_d) * gamma * I[i - 1] )
    D_std.append( D_std[i - 1] + CMR * gamma * I[i - 1] )

# calculate the simulated number of new deaths each day.
nDeaths = arr.array('f',[D[0] - acc_deaths[len(acc_ndeaths)-1]] )
for i in range (1, no_days):
    nDeaths.append(D[i]-D[i-1])
    


# Some key facts:
print("Latest Est. Active Cases: " + str(I[0]))
print("Latest Death Count: " + str(latest_deaths))
print("New Deaths Today: " + str(acc_ndeaths[len(acc_ndeaths)-1]))
    
    
## PLOT RESULTS OF SIMULATION
plt.rcParams['figure.figsize'] = [15, 5]
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
fig, (ax1,ax2,ax3) = plt.subplots(1,3)
ax1.scatter(days, acc_deaths,label='Actual Data')
ax1.plot(time, D, label='Projected')
ax1.set_xlim(-10, 10)
ax1.set_ylim(0, max(D[0:10]))
ax1.set_xlabel('Days from Today')
ax1.set_ylabel('Cumulative Deaths')
ax1.legend()

ax2.bar(days, acc_ndeaths)
ax2.bar(time, nDeaths)
ax2.set_xlim(-30,30)
ax2.set_ylim(0,max(nDeaths[0:30])*1.1)
ax2.set_xlabel('Days from Today')
ax2.set_ylabel('New Deaths per Day')

ax3.plot(time, pi)
ax3.set_xlim(0,30)
ax3.set_xlabel('Days from Today')
ax3.set_ylabel('Reproductive Number (R)')
