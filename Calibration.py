import pandas as pd
import numpy as np
import scipy.stats as si
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from scipy.optimize import minimize

def Call(f,k,sigma,r,t):
    d1=(np.log(f/k)+(0.5*sigma**2)*t)/(sigma*np.sqrt(t))
    d2=(np.log(f/k)-(0.5*sigma**2)*t)/(sigma*np.sqrt(t))
    call=np.exp(-r*t)*(f*si.norm.cdf(d1,0.0,1.0)-k*si.norm.cdf(d2,0.0,1.0))
    return call


def Put(f,k,sigma,r,t):
    d1=(np.log(f/k)+(0.5*sigma**2)*t)/(sigma*np.sqrt(t))
    d2=(np.log(f/k)-(0.5*sigma**2)*t)/(sigma*np.sqrt(t))
    put=np.exp(-r*t)*(k*si.norm.cdf(-d2,0.0,1.0)-f*si.norm.cdf(-d1,0.0,1.0))
    return put

def ClmVol(f,k,r,t,c):
    left=0
    right=5
    while (right-left)>0.0001:
        mid=0.5*(left+right)
        if Call(f,k,mid,r,t) > c:
            right=mid
        else:
            left=mid
    return 0.5*(left+right)

def PlmVol(f,k,r,t,p):
    left=0
    right=5
    while (right-left)>0.0001:
        mid=0.5*(left+right)
        if Put(f,k,mid,r,t) > p:
            right=mid
        else:
            left=mid
    return 0.5*(left+right)

def fwd(s0,r,t,div,div_t):
    fwd = s0*np.exp(r*t)-div*np.exp(r*div_t)
    return fwd


df3 = pd.read_csv('InputData3.csv')
s0 = 306.165
r = 0.0159
div = 1.39

# Calculate forward price
df3['fwd'] = df3.apply(lambda row:
     fwd(s0,r,row['t'],div,row['t']-0.1342),
     axis=1)

# Using put and call price together to calculate Implied Volatility
df3['MKTVol'] = df3.apply(lambda row:
     PlmVol(row['fwd'],row['Strike'],r,row['t'],row['Put'])
     if row['Strike'] <= 305
     else ClmVol(row['fwd'],row['Strike'],r,row['t'],row['Call']),
     axis=1)

#print(df2)

def SABRVol(alpha,beta,rho,nu,f,k,t):
    M = (f*k)**((1-beta)/2.)
    A = 1+((((1-beta)*alpha)**2)/(24.*(M**2))+(alpha*beta*nu*rho)/(4.*M)+((nu**2)*(2-3*(rho**2))/24.))*t
    B = 1+(1/24.)*(((1-beta)*np.log(f/k))**2)+(1/1920.)*(((1-beta)*np.log(f/k))**4)
    Z = (nu/alpha)*M*np.log(f/k)
    X = np.log((np.sqrt(1-2*rho*Z+Z**2)+Z-rho)/(1-rho))
    if k == f:
        SABRVol = (alpha/M)*A
    else:
        SABRVol = (nu*np.log(f/k)*A)/(X*B)
    return SABRVol

def SSE(x):
    alpha = x[0]
    beta = x[1]
    rho = x[2]
    nu = x[3]
    sum_sq_err = 0
    for i in range(len(df3['MKTVol'])):
        sum_sq_err += (SABRVol(alpha,beta,rho,nu,df3['fwd'][i],df3['Strike'][i],df3['t'][i]) - df3['MKTVol'][i])**2
    return sum_sq_err

starting_guess = np.array([0.001,0.5,0,0.001])
bnds = ((0.001, None), (0, 1), (-0.999, 0.999), (0.001, None))
sol = minimize(SSE, starting_guess, bounds=bnds, method='SLSQP') # for a constrained minimization of multivariate scalar functions
print(sol)

df3['SABRVol'] = df3.apply(lambda row:
     SABRVol(sol.x[0],sol.x[1],sol.x[2],sol.x[3], row['fwd'], row['Strike'], row['t']),
     axis=1)

print(df3)

fig = plt.figure()
ax = plt.axes(projection='3d')
xdata = df3['Strike']
ydata = df3['t']
zdata = df3['MKTVol']
ax.scatter3D(xdata, ydata, zdata, c='r',  marker='o')
xtry = np.linspace(250, 360, 100)
ytry = np.linspace(0.1, 0.4, 100)
X, Y = np.meshgrid(xtry, ytry)
Z = np.array([[0.0]*len(xtry)]*len(ytry))
for i in range(len(ytry)):
    for j in range(len(xtry)):
        Z[i,j]=(SABRVol(sol.x[0],sol.x[1],sol.x[2],sol.x[3],fwd(s0,r,ytry[i],div,ytry[i]-ytry[0]),xtry[j],ytry[i]))
ax.plot_surface(X, Y, Z, alpha = 0.6)
ax.set_xlabel('Strike')
ax.set_ylabel('Time to maturity')
ax.set_zlabel('Implied Volatility')
plt.show()


plt.scatter(df3['t'], df3['MKTVol'], label='MKT')
plt.xlabel('Time to maturity')
plt.ylabel('Implied Volatility')
plt.legend(loc=0)
plt.show()
