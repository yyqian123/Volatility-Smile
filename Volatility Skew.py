import numpy as np
import scipy.stats as si
import pandas as pd
from matplotlib import pyplot as plt


def BSC(s0,k,sigma,r,t):
    d1=(np.log(s0/k)+(r+0.5*sigma**2)*t)/(sigma*np.sqrt(t))
    d2=(np.log(s0/k)+(r-0.5*sigma**2)*t)/(sigma*np.sqrt(t))
    bs_call=s0*si.norm.cdf(d1,0.0,1.0)-k*np.exp(-r*t)*si.norm.cdf(d2,0.0,1.0)
    return bs_call


def BSP(s0,k,sigma,r,t):
    d1=(np.log(s0/k)+(r+0.5*sigma**2)*t)/(sigma*np.sqrt(t))
    d2=(np.log(s0/k)+(r-0.5*sigma**2)*t)/(sigma*np.sqrt(t))
    bs_put=k*np.exp(-r*t)*si.norm.cdf(-d2,0.0,1.0)-s0*si.norm.cdf(-d1,0.0,1.0)
    return bs_put

def BSClmVol(s0,k,r,t,c):
    left=0
    right=5
    while (right-left)>0.0001:
        mid=0.5*(left+right)
        if BSC(s0,k,mid,r,t) > c:
            right=mid
        else:
            left=mid
    return 0.5*(left+right)

def BSPlmVol(s0,k,r,t,p):
    left=0
    right=5
    while (right-left)>0.0001:
        mid=0.5*(left+right)
        if BSP(s0,k,mid,r,t) > p:
            right=mid
        else:
            left=mid
    return 0.5*(left+right)

def St(s0, sigma0, pho, volvol):
    x1 = np.random.normal(0, 1, 365)
    x2 = np.random.normal(0, 1, 365)
    z1 = x1
    z2 = pho * x1 + np.sqrt((1-pho*pho)) * x2
    s = [0.0] * 365
    sigma = [0.0] * 365
    s[0] = s0
    sigma[0] = sigma0
    for i in range(364):
        s[i+1]= s[i] + s[i]*sigma[i] * np.sqrt(1.0/365) * z1[i]
        sigma[i + 1] = sigma[i] + sigma[i] * volvol * np.sqrt(1.0/365) * z2[i]
    return s[-1]



def option_price(s0, sigma0, pho, volvol,strike_c,strike_p):
    p_value = [0.0] * len(strike_p)
    c_value=[0.0]*len(strike_c)
    s = St(s0, sigma0, pho, volvol)
    for i in range(len(strike_p)):
        p_value[i] = max(0, strike_p[i] - s)
    for j in range(len(strike_c)):
        c_value[j] = max(0, s - strike_c[j])
    #print(s)
    return p_value+c_value

#print(option_price(s0, sigma0, alpha, volvol,strike_c,strike_p))

def simulation(s0, sigma0, pho, volvol,strike_c,strike_p,n):
    l = len(strike_p)+len(strike_c)
    op_simu = np.array([[0.0]*l]*n)
    for i in range(n):
        op_simu[i,:] = option_price(s0, sigma0, pho, volvol, strike_c, strike_p)
    return np.mean(op_simu, axis=0)

#print(simulation(s0, sigma0, alpha, volvol,strike_c,strike_p,n))

def implied_vol(s0, sigma0, pho, volvol, strike_c, strike_p, n):
    optionprice = simulation(s0, sigma0, pho, volvol, strike_c, strike_p, n)
    implied_vol_p = [0.0] * len(strike_p)
    implied_vol_c = [0.0] * len(strike_c)
    for i in range(len(strike_p)):
        implied_vol_p[i] = BSPlmVol(s0, strike_p[i], r, t, optionprice[i])
    for j in range(len(strike_c)):
        implied_vol_c[j] = BSClmVol(s0,strike_c[j], r, t, optionprice[j+len(strike_p)])
    #print(optionprice)
    return implied_vol_p + implied_vol_c


n=10000
s0 = 45
sigma0 = 0.16
r=0.00
t=1
strike_p = [30, 33, 36, 38, 40, 42, 43, 44, 45]
strike_c = [46, 47, 48, 50, 52, 55, 58, 61, 65]

x=strike_p+strike_c

'''y1=implied_vol(s0, sigma0, -5, 0.25,strike_c,strike_p,n)
y2=implied_vol(s0, sigma0, -5, 0,strike_c,strike_p,n)
y3=implied_vol(s0, sigma0, -5, 1,strike_c,strike_p,n)
plt.plot(x,y1,'b')
plt.plot(x,y2,'r')
plt.plot(x,y3,'g')
plt.show()'''

y1=implied_vol(s0, sigma0, -0.3, 0.8, strike_c, strike_p, n)
#print(y1)
y2=implied_vol(s0, sigma0, -0.6, 0.8, strike_c, strike_p, n)
y3=implied_vol(s0, sigma0, 0, 0.8, strike_c, strike_p, n)
y4=implied_vol(s0, sigma0, 0, 1.6, strike_c, strike_p, n)
plt.plot(x,y1,'b',label='cor=-0.3,vov=1.2')
plt.plot(x,y2,'r',label='cor=-0.6,vov=1.2')
plt.plot(x,y3,'g',label='cor=0,vov=1.2')
#plt.plot(x,y4,'y',label='cor=0,vov=1.3')
plt.xlabel("Strike")
plt.ylabel("Implied Vol")
plt.legend()
plt.show()
