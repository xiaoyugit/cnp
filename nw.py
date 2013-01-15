# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 17:31:36 2012

@author: Eric
"""

from pylab import *
from numpy import fill_diagonal
from scipy.stats import norm,lognorm
from openopt import QP
from scipy.optimize import fmin
from scipy.optimize import fmin_bfgs
from scipy.integrate import trapz

class nw:
    def __init__(self,ydat,xdat,x,h=nan,kf='gaussian'):
        self.ndim=len(ydat)        
        self.ydat=ydat
        self.xdat=xdat
        self.x=x
        self.kf=kf
        if isnan(h):
            self.h=bws(ydat,xdat,0.1).h
            print "optimal bandwidth is %s" %self.h
        else: self.h=h
    def k(self):
        self.u=array((mat(self.x).T-mat(self.xdat))/self.h)
        if self.kf=="gaussian":
            return exp(-self.u*self.u/2)/sqrt(2*pi)
    def m(self):
        '''nw estimator'''
        self.den=dot(self.k(),ones(self.ndim))        
        return dot(self.k(),self.ydat)/self.den
    def kernelweight(self):
        '''H'''
        self.den=dot(self.k(),ones(self.ndim))        
        return self.k()/self.den
    def m_fd(self):
        '''jacob matrix of m'''
        self.den=dot(self.k(),ones(self.ndim))        
        left=(-self.u/self.h)*self.k()*array(mat(self.den).T*ones(self.ndim))
        right=-self.k()*dot((-self.u/self.h)*self.k(),ones((self.ndim,self.ndim)))
        return (left+right)*self.ydat/array(mat(self.den**2).T*ones(self.ndim))
    def pointwisec(self,alpha=0.05):
        "pontwise confidence interval"
        critical_v=norm.ppf(1-alpha/2.)
        m_var=(1-self.m())*self.m()
        density=dot(self.k(),ones(self.ndim))/self.ndim/self.h
        width=sqrt(m_var/(2*pi**0.5)/(density*self.ndim*self.h))
        return [self.m()-critical_v*width,self.m()+critical_v*width]
    def uniformc(self,alpha=0.05):
        '''uniform confidence band'''
        m_var=(1-self.m())*self.m()
        density=dot(self.k(),ones(self.ndim))/self.ndim/self.h
        width=sqrt(m_var/(2*pi**0.5)/(density*self.ndim*self.h))
        t1=-log(-log(1-alpha)/2.)/sqrt(2*log(1./self.h))
        t2=sqrt(2*log(1./self.h))
        t3=log(1./(8*pi**2))/sqrt(8*log(1./self.h))
        critical_v=(t1+t2+t3)
        return [self.m()-critical_v*width,self.m()+critical_v*width]
    def mean(self):
        return trapz(dot(self.m_fd(),ones(self.ndim)),exp(self.x))

class cnw(nw):
    '''constrained nw'''
    def tilt(self):
        ######construct constraint weight matrix and optimize p############
        ndim=self.ndim
        wmatrix=self.ndim*self.m_fd()
        wmatrix_2=self.ndim*self.k()*self.ydat
        p0=ones(ndim)/ndim
        lb=zeros(ndim)
        ub=ones(ndim)
        Aeq=ones(ndim)
        beq=1. 
        #A=vstack((-wmatrix,-wmatrix_2,-wmatrix_2))
        #b=hstack((zeros(len(wmatrix)),ones(len(wmatrix_2)),zeros(len(wmatrix_2))))
        A=vstack((-wmatrix,-wmatrix_2))
        b=hstack((zeros(len(wmatrix)),ones(len(wmatrix_2))))
        p=QP(diag(ones(ndim)),2*p0*ones(ndim),lb=lb,ub=ub,Aeq=Aeq,beq=beq,A=A,b=b,name="find optimal p that satisfies m'>0 and m<1")
        r=p._solve('cvxopt_qp')
        solution=r.xf
        self.solution=solution
        self.ydat=ndim*solution*self.ydat

class bws(nw):
    def __init__(self,ydat,xdat,h=0.15,kf='gaussian'):
        self.ndim=len(ydat)        
        self.ydat=ydat
#        self.ydat=mat(ones(self.ndim)).T*mat(ydat)
#        fill_diagonal(self.ydat,0)
        self.xdat=mat(ones(self.ndim)).T*mat(xdat)
#        fill_diagonal(self.xdat,0)
        self.x=xdat
        self.kf=kf
        self.h=h
        self.h=fmin(self.mse,self.h)[0]#optimal h       
    def k(self):
        self.u=array((mat(self.x).T-mat(self.xdat))/self.h)
        if self.kf=="gaussian":
            weight=exp(-self.u*self.u/2)/sqrt(2*pi)
            fill_diagonal(weight,0)
        return weight
    def mse(self,h):
        '''least square cross-validation bandwidth selection'''
        self.h=h    
        mse=sum((self.ydat-self.m())**2)
        print "current function value: %s" % (mse/self.ndim)
        return mse/self.ndim
    def aic(self,h):
        '''aic'''
        self.h=h    
        left=log(sum((self.ydat-self.m())**2)/self.ndim)
        #left=log(mat(self.ydat)*mat(mat(ones([self.ndim,self.ndim]))-self.kernelweight())*mat(mat(ones([self.ndim,self.ndim]))-self.kernelweight()).T*mat(self.ydat).T/self.ndim)
        right=(1+trace(self.kernelweight())/self.ndim)/(1-(trace(self.kernelweight())+2)/self.ndim)
        aic=left+right
        print "current function value: %s" % aic
        return aic
