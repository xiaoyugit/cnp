# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 18:44:52 2012

@author: Eric
"""
from pylab import *

def k(u):
    '''kernel function'''
    return exp(-u**2/2)/sqrt(2*pi)

def zheng(x,res,h):
    den=0.
    num=0.
    n=len(x)
    for i in xrange(n):
        density=k((x[i]-x)/h)
        tmp=density*res*res[i]
        tmp[i]=0
        num+=tmp.sum()
        tmp=2*density**2*res**2*res[i]**2
        tmp[i]=0
        den+=tmp.sum()
    return num/sqrt(den)
