 # -*- coding: utf-8 -*-
"""
Created on Thu May 16 11:29:45 2019

@author: Abhinav_Kulkarni
MIMO uplink detection
DSGO <=> DN 
"""
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as pl

import time as t
import decimal as dc
import mpmath as mt
          

#function declarations
#QPSK Modulation
def qamod(sym,constel):  
    c=np.sqrt(constel)
    re=-2*np.mod(sym,c)+c-1
    im=2*np.floor(sym/c)-c+1
    return re+1j*im

def qademod(symb,constel):  #One symbol at a time
    QAM_map=qamod(np.arange(0,constel),constel)
    symbol_distance=np.square(np.abs(symb-QAM_map))
    return np.argmin(symbol_distance)

def calculate_evm(ex_est,ex):     
    evm_real=np.sum(np.square(ex_est.real-ex.real))
    evm_imag=np.sum(np.square(ex_est.imag-ex.imag))
    return np.sqrt((evm_real+evm_imag)/ex.size)
        
def calculate_symbol_error_rate(x_est,data): #requires a flattened array
    error_rate=0
    for loop in range(data.size):
        if qademod(x_est[loop],M_QAM)!=data[loop]:
            error_rate+=1
    return (error_rate)
        
def calculate_SINR_AVG(x_est_l,x_l):  #do not put x_est_frame = x_frame, else division by zero error 
    power_transmit=np.square(np.linalg.norm(x_l,axis=0))
    power_receive=np.square(np.linalg.norm(x_est_l,axis=0))
    return np.average(10*np.log10(np.divide(power_receive,np.abs((power_transmit-power_receive)))))
#    return np.average(np.abs(power_transmit/power_transmit-power_receive))

def calculate_SigmaN2(SNR):
    return np.float_power(10,-1*SNR/10.0) #evaluates to float64

def float2fix(a,bit_precision):
    a_int_frac=np.modf(a)
    scaled_frac=np.multiply(a_int_frac[0],np.power(2,bit_precision))
    int_part=np.rint(scaled_frac)
    n_int_part=np.divide(int_part,np.power(2,bit_precision))
    return_value=a_int_frac[1]+n_int_part
    return return_value

def float2fix_complex(a_cmplx,bit_precision):
    a_cmplx_l=np.copy(a_cmplx)
    a_cmplx_l.real=float2fix(a_cmplx_l.real,bit_precision)
    a_cmplx_l.imag=float2fix(a_cmplx_l.imag,bit_precision)
    return a_cmplx_l


#variable declarations
#MKratio=32   #MK ratio
set_iteration=4
K=np.uint16(8)    #number of user t  erminals
cluster_number=np.uint16(2) #number of clusters  
MperC=32
M=np.uint16(cluster_number*MperC)   #number of antennas 
bit_precision=8
bit_precision_mp=50
multiple_frame=100
M_QAM=np.uint16(16)
Nframes=1


mmse_roh=0.01


QAM_map=qamod(np.arange(0,M_QAM),M_QAM)     #write a qammod function
data= np.random.randint(0,M_QAM,size=(K,Nframes))
QAM_mod=np.array(QAM_map[data]) #symbol mapped tx signal
QAM_var=np.sqrt(np.var(QAM_map))
x=QAM_mod
x_d=x

def generate_data(M_QAM,K,Nframes):
    QAM_map= qamod(np.arange(0,M_QAM),M_QAM)     #write a qammod function
    data= np.random.randint(0,M_QAM,size=(K,Nframes))
    QAM_mod=np.array(QAM_map[data]) #symbol mapped tx signal
    QAM_var=np.sqrt(np.var(QAM_map))
    x=QAM_mod/QAM_var
    #x=QAM_mod
    return x,data,QAM_var

def simulate_receive_signal(x,SNR,scale,H):
    var=np.sqrt(calculate_SigmaN2(SNR)/2)
    noise_sys=var*np.random.normal(loc=0, scale=1, size=(M,Nframes*2)).view(np.complex128)
    y=np.zeros((M,Nframes),dtype=x.dtype) #array holding y(number of antennas)
    for f in range(0,Nframes):
        y[:,f]= (H @ x[:,f]) + noise_sys[:,f]
    return y        
     
x_est=np.zeros(x.shape,dtype=x.dtype)  

i_gauss_ADMM=np.uint16(6) #number of iterations

#ZF method
def zf_method(y,x,A):
    x_est_zf=np.zeros(x.shape,dtype=x.dtype) 
    invA=np.zeros(A.shape,dtype=A.dtype)
    yMF=np.zeros((K,Nframes),dtype=x.dtype)
    invA=np.linalg.inv(A)    
    for f in range(0,Nframes):
        yMF[:,f]=np.conjugate(H.T) @ y[:,f]   #read about the MRC ratio combining
        x_est_zf[:,f]=invA @ yMF[:,f]
    return x_est_zf

def mmse_method(y,x,A,sigmaN2):
    x_est_mmse=np.zeros(x.shape,dtype=x.dtype) 
    invA=np.zeros(A.shape,dtype=A.dtype)
    yMF=np.zeros((K,Nframes),dtype=x.dtype)
    sigmaN2_QAM=np.square(QAM_var)
    A_mmse=A+((sigmaN2/sigmaN2_QAM)*np.identity(K,dtype=np.complex128))
    invA=np.linalg.inv(A_mmse)    
    for f in range(0,Nframes):
        yMF[:,f]=np.conjugate(H.T) @ y[:,f]   #read about the MRC ratio combining
        x_est_mmse[:,f]=invA @ yMF[:,f]
    return x_est_mmse

def admm_method(y,x,M,K,i_gauss_ADMM,Max_C,roh,alpha,betta,SigmaN2):
    x_est_admm=np.zeros(x.shape,dtype=x.dtype) 
    MperC=np.uint16(M/Max_C) #number of antennas per cluster
    w=np.zeros((K,1),dtype=A.dtype)
    zc=np.zeros((K,Max_C),dtype=A.dtype) 
    vc=np.zeros((K,Max_C),dtype=A.dtype)
    Lambdac=np.zeros((K,Max_C),dtype=A.dtype)
    H_temp=np.zeros((MperC,H.shape[1]),dtype=H.dtype) 
    HH2D=np.zeros((K,K),dtype=H.dtype)
    E=np.zeros((K,K),dtype=H.dtype)
    ID=np.zeros((K,K),dtype=H.dtype)
    InvDE=np.zeros((K,K),dtype=H.dtype)
    for f in range(Nframes):
        Lambdac[:]=0
        zc[:]=0
        vc[:]=0
        HH2D[:]=0
        E[:]=0
        ID[:]=0
        InvDE[:]=0
        for clusterc in range(Max_C):
            H_temp=H[MperC*clusterc:MperC*(clusterc+1),:].copy();
            HH2D=np.conjugate(H_temp.T) @ H_temp
            E=np.tril(HH2D,-1)
            np.fill_diagonal(ID,1/HH2D.diagonal(0))
            InvDE=np.linalg.inv(np.tril(HH2D,0))     #create a function for using previous value of r
            zc[:,clusterc]=InvDE @ (np.conjugate(H_temp.T) @ y[MperC*clusterc:MperC*(clusterc+1),f])
            w[:,0]+=zc[:,clusterc]
        #cluster loop     
        x_est_admm[:,f]=w[:,0]/Max_C;    
        for i in range(i_gauss_ADMM):
            for clusterc in range(Max_C):
                H_temp=H[MperC*clusterc:MperC*(clusterc+1),:];
                HH2D=np.conjugate(H_temp.T) @ H_temp + roh * np.identity(K,dtype=H.dtype)
                E=np.tril(HH2D,-1)
                np.fill_diagonal(ID,1/HH2D.diagonal(0))
                InvDE=np.linalg.inv(np.tril(HH2D,0))     #create a function for using previous value of r        
                Lambdac[:,clusterc]+=alpha*(zc[:,clusterc]-x_est_admm[:,f])  
                zc[:,clusterc]=InvDE @ ((np.conjugate(H_temp.T) @ y[MperC*clusterc:MperC*(clusterc+1),f])+betta*(x_est_admm[:,f]+Lambdac[:,clusterc])-(np.triu(HH2D,1)@x_est_admm[:,f]))
                vc[:,clusterc]=zc[:,clusterc]+Lambdac[:,clusterc]   #compute bc
                w[:,0]+=vc[:,clusterc] #accumulate bc =for s
            x_est_admm[:,f]=w[:,0]/Max_C
            w[:,0]=0
    return x_est_admm

####################################################
   
#Decentralized Coordinate-Descent Data Detection
#and Precoding for Massive MU-MIMO
#Kaipeng Li1, Oscar Casta˜neda2, Charles Jeon3, Joseph R. Cavallaro1, and Christoph Studer2
        
#calculating for single frame
#Note M=B; K=U
def cd_method(y,x,M,K,iterations,clustern,SigmaN2):
    x_est_cd=np.zeros(x.shape,dtype=x.dtype)
    muc=np.zeros((K,clustern),dtype=A.dtype)    #these are the constants
    nuc=np.zeros((K,clustern),dtype=A.dtype)
    rc=np.zeros((M//clustern,1),dtype=A.dtype) 
    deltauc=np.zeros((K,clustern),dtype=A.dtype)
    a_per_clust=M//clustern
    hctemp=np.zeros((a_per_clust,K),dtype=A.dtype)
    x_cluster=np.zeros((K,clustern),dtype=A.dtype)
 
    Ex=1                    #per user transmit energy from user terminal
    N0=SigmaN2#noise variance
    
    #performance parameters
    
    #frame loop goes here
    
#       distributed processing   
    for cluster in range(clustern):
        hctemp=H[a_per_clust*cluster:a_per_clust*(cluster+1),:]
        muc[:,cluster]=(1/(np.power(np.linalg.norm(hctemp,axis=0),2)+(N0/Ex)))
        nuc[:,cluster]=muc[:,cluster]*(np.power(np.linalg.norm(hctemp,axis=0),2))

    for f in range(Nframes):          
        for cluster in range(clustern):
            rc=y[a_per_clust*cluster:a_per_clust*(cluster+1),f].copy() #use copy(), else value of global valuye of y changes
            x_cluster[:,cluster]=0
            hctemp=H[a_per_clust*cluster:a_per_clust*(cluster+1),:]
            for itern in range(iterations):
                for u in range(K):
                    deltauc[u,cluster]=muc[u,cluster]*(np.conjugate(hctemp[:,u].T) @ rc) + ((nuc[u,cluster]-1) * x_cluster[u,cluster])
                    x_cluster[u,cluster]=muc[u,cluster]*(np.conjugate(hctemp[:,u].T) @ rc) + (nuc[u,cluster] * x_cluster[u,cluster])
                    rc-=(deltauc[u,cluster] * hctemp[:,u])
        #centralized processing  
        x_est_cd[:,f]=(1/clustern)*np.sum(x_cluster,axis=1)       #Assumed the variance of each cluster is same, so factor (1/clustern)
    return x_est_cd        
#######################################################

       
def sgd_method(y,x,M,K,SNR):   
    x_est_sgd=np.zeros(x.shape,dtype=x.dtype) 
    A=np.identity(K,dtype=H.dtype)
    W=np.zeros((M,K),dtype=H.dtype)
    um=np.float64(0)
    x_est_partial=np.zeros((x_est_sgd.shape[0],M),dtype=x_est_sgd.dtype)
    #cluster processing at each m corresponds to a single antenna
    #1 antenna = 1 cluster
    u=0.5*K*np.log(4.0*M*np.power(10.0,SNR/10.0))/M 
    for f in range(Nframes):
        A=np.identity(K,dtype=H.dtype)
        for cluster in range(M):   
            um=u/np.power(np.linalg.norm(H[cluster,:]),2)
            W[cluster,:]=um*(A @ H[cluster,:].T)
            A=A-(W[cluster,:].reshape(W.shape[1],1) @ np.conjugate(H[cluster,:].reshape(1,H.shape[1])))    #this has an issue
            x_est_partial[:,cluster]=np.conjugate(W[cluster,:]) * y[cluster,f]
        #central processing; accumulate partial x_est values and combine them into x_ext 
        x_est_sgd[:,f]=np.sum(x_est_partial,axis=1)
        x_est_partial[:]=0
    return x_est_sgd
##################################################################################



def f_gradient_local(Hc,x,yc): #computes gradient based on the current estimate of x
    x_gradient=np.zeros((x.shape[0],1),dtype=np.complex128)
    p1=np.conjugate(Hc.T) @ Hc
    p2=np.conjugate(Hc.T) @ yc.reshape(yc.shape[0],1)
    x_gradient=((p1 @ x)-p2)
    return x_gradient

def f_gradient_local_fixed(Hc,x,yc): #computes gradient based on the current estimate of x
    x_gradient=np.zeros((x.shape[0],1),dtype=np.complex128)
    p1=float2fix_complex(np.conjugate(Hc.T) @ Hc,bit_precision)
    p2=float2fix_complex(np.conjugate(Hc.T) @ yc.reshape(yc.shape[0],1),bit_precision)
    x_gradient=(float2fix_complex((p1 @ x),bit_precision)-p2)
    return x_gradient

def f2_gradient_local(Hc):
    p1=np.around(np.conjugate(Hc.T) @ Hc,decimals=8)
    x2_gradient=np.abs(np.diag(p1))
    return x2_gradient[:,np.newaxis]
    

def dsgo_method(y,x,M,K,H,iteration,clusters):
    gradient_l=np.zeros((K,clusters),dtype=np.complex128)
    gradient2_l=np.zeros((K,clusters),dtype=np.complex128)
    x_est_sgd_final=np.zeros(x.shape,dtype=x.dtype)
    x_est_sgd_l=np.zeros((K,1),dtype=np.complex128)
#    x_init=np.zeros((K,1),dtype=np.complex128)
    MperC=M//clusters
    for f in range(Nframes):
        for i in range(iteration): #iterations
            for cl in range(clusters):  #cluster loop
                 Hc=H[MperC*cl:MperC*(cl+1),:].copy();
                 yc=y[MperC*cl:MperC*(cl+1),[f]].copy();
                 if(i==0):
                     x_est_sgd_l=(np.conjugate(Hc.T) @ yc)/(np.power(np.linalg.norm(Hc,axis=0),2)).reshape(K,1)
                     gradient2_l[:,cl]=f2_gradient_local(Hc)
                 gradient_l[:,cl]=f_gradient_local(Hc,x_est_sgd_l,yc)
#                 gradient2_l[:,cl]=f2_gradient_local(Hc)
                 if(cl==(clusters-1)):
                     if(i==0):
                         g2=np.average(gradient2_l,axis=1).reshape(gradient2_l.shape[0],1)
                     g1=np.average(gradient_l,axis=1).reshape(gradient_l.shape[0],1)
                     x_est_sgd_l=x_est_sgd_l-(g1/g2)
        x_est_sgd_final[:,f]=x_est_sgd_l.reshape(K)
    return x_est_sgd_final

def dsgo_method_star(y,x,M,K,H,iteration,clusters):
    data_path1=np.zeros((K,clusters),dtype=np.complex128)
    data_path2=np.zeros((K,clusters),dtype=np.complex128)
    x_est_sgd_final=np.zeros(x.shape,dtype=x.dtype)
    x_est_sgd_l=np.zeros((K,1),dtype=np.complex128)
    MperC=M//clusters
    for f in range(Nframes):
        for i in range(iteration): #iterations
            for cl in range(clusters):  #cluster loop
                 Hc=H[MperC*cl:MperC*(cl+1),:].copy();
                 yc=y[MperC*cl:MperC*(cl+1),[f]].copy();
                 if(i==0):
                     x_est_sgd_l=(np.conjugate(Hc.T) @ yc)/(np.power(np.linalg.norm(Hc,axis=0),2)).reshape(K,1)
                     data_path1[:,[cl]]=f2_gradient_local(Hc)
                     data_path2[:,[cl]]=f_gradient_local(Hc,x_est_sgd_l,yc)
                 else:    
                     data_path2[:,[cl]]=f_gradient_local(Hc,data_path1[:,[cl]],yc)
                 if(cl==(clusters-1)): #last cluster
                     if(i==0):
                         g2=np.average(data_path1,axis=1).reshape(data_path1.shape[0],1)
                     g1=np.average(data_path2,axis=1).reshape(data_path2.shape[0],1)
                     x_est_sgd_l=x_est_sgd_l-(g1/g2)
                     for bcast in range(clusters):
                         data_path1[:,[bcast]]=x_est_sgd_l
        x_est_sgd_final[:,f]=x_est_sgd_l.reshape(K)
    return x_est_sgd_final


def dsgo_method_ring(y,x,M,K,H,iteration,clusters):
    data_path1=np.zeros((K,1),dtype=np.complex128)
    data_path2=np.zeros((K,1),dtype=np.complex128)
    x_est_sgd_final=np.zeros(x.shape,dtype=x.dtype)
    x_est_sgd_l=np.zeros((K,1),dtype=np.complex128)
    MperC=M//clusters
    for f in range(Nframes):
        for i in range(iteration): #iterations
            for cl in range(clusters):  #cluster loop
                 Hc=H[MperC*cl:MperC*(cl+1),:].copy();
                 yc=y[MperC*cl:MperC*(cl+1),[f]].copy();
                 if(i==0):
                     x_est_sgd_l=(np.conjugate(Hc.T) @ yc)/(np.power(np.linalg.norm(Hc,axis=0),2)).reshape(K,1)
                     data_path1+=f2_gradient_local(Hc)
                     data_path2+=f_gradient_local(Hc,x_est_sgd_l,yc)
                 else:    
                     data_path2+=f_gradient_local(Hc,data_path1,yc)
                 if(cl==(clusters-1)): #last cluster
                     if(i==0):
                         g2=data_path1/clusters
                     g1=data_path2/clusters
                     x_est_sgd_l=x_est_sgd_l-(g1/g2)
                     data_path1=x_est_sgd_l
                     data_path2[:]=0
        x_est_sgd_final[:,f]=x_est_sgd_l.reshape(K)
    return x_est_sgd_final

def dsgo_method_ring_fixed(y,x,M,K,H,iteration,clusters):
    y_l=y.copy()
    x_l=x.copy()
    H_l=H.copy()
    
    y_l=float2fix_complex(y_l,bit_precision)
    x_l=float2fix_complex(x_l,bit_precision)
    H_l=float2fix_complex(H_l,bit_precision)
    
    
    
    data_path1=np.zeros((K,1),dtype=np.complex128)
    data_path2=np.zeros((K,1),dtype=np.complex128)
    x_est_sgd_final=np.zeros(x_l.shape,dtype=x_l.dtype)
    x_est_sgd_l=np.zeros((K,1),dtype=np.complex128)
    MperC=M//clusters
    for f in range(Nframes):
        for i in range(iteration): #iterations
            for cl in range(clusters):  #cluster loop
                 Hc=H_l[MperC*cl:MperC*(cl+1),:];
                 yc=y_l[MperC*cl:MperC*(cl+1),[f]];
                 if(i==0):
                     x_est_sgd_l=float2fix_complex((np.conjugate(Hc.T) @ yc),bit_precision)/(np.power(np.linalg.norm(Hc,axis=0),2)).reshape(K,1)
                     data_path1+=f2_gradient_local(Hc)
                     data_path2+=f_gradient_local_fixed(Hc,x_est_sgd_l,yc)
                 else:    
                     data_path2+=f_gradient_local_fixed(Hc,data_path1,yc)
                 if(cl==(clusters-1)): #last cluster
                     if(i==0):
                         g2=float2fix_complex(data_path1/clusters,bit_precision)
                     g1=data_path2/clusters
                     x_est_sgd_l=x_est_sgd_l-float2fix_complex((g1/g2),bit_precision)
                     data_path1=x_est_sgd_l
                     data_path2[:]=0
        x_est_sgd_final[:,f]=x_est_sgd_l.reshape(K)
   

    x_est_sgd_final_o=x_est_sgd_final.copy() #required or no?
    x_est_sgd_final_o=float2fix_complex(x_est_sgd_final_o,bit_precision)
    
    
    return x_est_sgd_final_o



p_a=1/M_QAM


def new_exp(val):
    res=np.longdouble(0)
    for it in range(10):
        res=res+np.power(val,it)/np.math.factorial(it)
    return res


def wa_mp(sl,t,qam_sym):
    #convert numpy -> mpmath
    mt.dps = bit_precision_mp
    mt.pretty=True
    sl_m=mt.mpc(sl)
    t_m=mt.mpc(t)
    qam_sym_m=mt.matrix(qam_sym)
    e_diff=mt.matrix(qam_sym_m.rows,qam_sym_m.cols)
    nom=mt.matrix(qam_sym_m.rows,qam_sym_m.cols)
    res=mt.matrix(qam_sym_m.rows,qam_sym_m.cols)
    denom=mt.mpf(0)
    for i in range(qam_sym_m.rows):
        e_diff[i,0]=-1*mt.fdiv((mt.fabs(sl_m-qam_sym_m[i,0]))**2,t_m)
        nom[i,0]=mt.exp(e_diff[i,0])
        denom=mt.fadd(denom,nom[i,0])
    for i in range(qam_sym_m.rows):
        res[i,0]=mt.fdiv(nom[i,0],denom)
         #convert mpmath -> numpy
    return np.complex128(res)


def wa(sl,t,qam_sym):
    e_diff=-1*np.square(np.abs(sl-qam_sym))
    e_diff_t=e_diff/t
    nom=np.exp(e_diff_t)
    denom=np.sum(nom)
    res=nom/denom
    return res

def F(sl,t,qam_sym):
    su=qam_sym * wa_mp(sl,t,qam_sym)
    return np.sum(su)



def FG_mp(sl,t,qam_sym):
    #convert numpy -> mpmath
    mt.dps = bit_precision_mp
    mt.pretty=True
    sl_m=mt.mpc(sl)
    t_m=mt.mpc(t)
    f_v=mt.mpc(0)
    g_v=mt.mpc(0)
    qam_sym_m=mt.matrix(qam_sym)
    e_diff=mt.matrix(qam_sym_m.rows,qam_sym_m.cols)
    nom=mt.matrix(qam_sym_m.rows,qam_sym_m.cols)
    wa=mt.matrix(qam_sym_m.rows,qam_sym_m.cols)
    denom=mt.mpf(0)
    for i in range(qam_sym_m.rows):
        e_diff[i,0]=-1*mt.fdiv((mt.fabs(sl_m-qam_sym_m[i,0]))**2,t_m)
        nom[i,0]=mt.exp(e_diff[i,0])
        denom=mt.fadd(denom,nom[i,0])
    for i in range(qam_sym_m.rows):
        wa[i,0]=mt.fdiv(nom[i,0],denom)
        f_v=f_v+mt.fmul(qam_sym_m[i,0],wa[i,0])
    #G begins here
    diffe=mt.matrix(qam_sym_m.rows,qam_sym_m.cols) 
    mule=mt.matrix(qam_sym_m.rows,qam_sym_m.cols) 
    for i in range(qam_sym_m.rows):
        diffe[i,0]=(mt.fabs(qam_sym_m[i,0]-f_v))**2
        mule[i,0]=mt.fmul(diffe[i,0],wa[i,0])
        g_v=g_v+mule[i,0]
       #convert mpmath -> numpy
    return np.complex128(f_v),np.complex128(g_v)

def FG_mp_vector(sl_vec,t,qam_sym):
    F_val=np.zeros(sl_vec.size,dtype=sl_vec.dtype)
    G_val=np.zeros(sl_vec.size,dtype=sl_vec.dtype)
    for it in range(sl_vec.size):
        val=FG_mp(sl_vec[it],t,qam_sym)
        F_val[it]=val[0]
        G_val[it]=val[1]
    return F_val,G_val



def G(sl,t,qam_sym):
    t1=np.square(np.abs(qam_sym-F(sl,t,qam_sym)))
    t2=wa_mp(sl,t,qam_sym) * t1
    return np.sum(t2)
   
def F_vector(sl_vec,t,qam_sym):
    val=np.zeros(sl_vec.size,dtype=sl_vec.dtype)
    for it in range(sl_vec.size):
        val[it]=F(sl_vec[it],t,qam_sym)
    return val

def G_vector(sl_vec,t,qam_sym):
    val=np.zeros(sl_vec.size,dtype=sl_vec.dtype)
    for it in range(sl_vec.size):
        val[it]=G(sl_vec[it],t,qam_sym)
    return val
        
def normal_prob_function(x,u,sigma):
    denom=np.sqrt(2*np.pi*np.square(sigma))
    nom_1=-1*np.square(np.abs(x-u))/2*np.square(sigma)
    nom_2=np.exp(nom_1)
    return nom_2/denom

def normal_prob_function_complex(x_real,x_imag,u,sigma):
    return normal_prob_function(x_real,u,sigma)*normal_prob_function(x_imag,u,sigma)
            
def variance_cluster(z,s0,ps0): #z is a symbol s0 distributed equally
    e=np.zeros((z.shape))
    for it in range(z.size):
        e[it]=(1/ps0)*np.sum(np.square(np.abs(z[it]-s0)))
    return np.float64(e)

def factor_calculation(beta,phi1,phi2,N0): #z is a symbol s0 distributed equally
    mt.dps = bit_precision_mp
    mt.pretty=True
    beta_m=mt.mpc(beta)
    phi1_m=mt.mpc(phi1)
    phi2_m=mt.mpc(phi2)
    N0_m=mt.mpc(N0)
    nom=mt.mpc(0)
    denom=mt.mpc(0)
    res=mt.mpc(0)
    nom=mt.fmul(beta_m,phi2_m)
    denom=mt.fadd(N0_m,mt.fmul(beta_m,phi1_m))
    res=mt.fdiv(nom,denom)
    return np.complex128(res)



def amp_method_fd(y,x,M,K,iterations,clustern,SigmaN2):
                    #per user transmit energy from user terminal
    N0=SigmaN2  #noise variance
    Ex=QAM_var**2
    z_cluster=np.zeros((K,clustern),dtype=A.dtype)
    sigma_cluster=np.zeros((K,clustern),dtype=A.dtype)
    s=np.full(K,Ex,dtype=A.dtype)
    vc=np.zeros(K,dtype=A.dtype) 
    phi_1=np.complex(0)  
    phi_2=np.complex(0)     
#       distributed processing
    x_est_lo=np.zeros(K,dtype=A.dtype)
    a_per_clust=M//clustern
    QAM_new=QAM_map/QAM_var
    x_est_amp_fd=np.zeros(x.shape,dtype=x.dtype)
    beta_c=beta*clustern
    for f in range(Nframes): 
        for cluster in range(clustern):
            H_c=(H[a_per_clust*cluster:a_per_clust*(cluster+1),:].copy())
            y_c=(y[a_per_clust*cluster:a_per_clust*(cluster+1),f].copy())
            ymr_c=np.conjugate(H_c.T) @ y_c
            gmr_c=np.conjugate(H_c.T) @ H_c
            phi_1=0;
            phi_2=0
            s[:]=0
            vc[:]=0
            for itern in range(iterations):
                z_cluster[:,cluster]=ymr_c+(np.identity(K,dtype=np.complex128)-gmr_c) @ s + vc
                mt.dps=bit_precision_mp
                N0_m=mt.mpf(N0)
                beta_m=mt.mpc(beta_c)
                phi_1_m=mt.mpc(phi_1)
                t1=mt.fmul(beta_m,phi_1_m)
                t2=mt.fadd(N0_m,t1)
                fg=FG_mp_vector(z_cluster[:,cluster],np.complex128(t2),QAM_new)
                #if(itern!=iterations-1):
                mean_vec=fg[1]
                phi_2=np.mean(mean_vec)
                vc=factor_calculation(beta_c, phi_1,phi_2,N0)*(z_cluster[:,cluster]-s)
                s=fg[0]
                phi_1=phi_2.copy()
            #sigma_cluster[:,cluster]=clustern*N0+beta_c*variance_cluster(z_cluster[:,cluster],QAM_new,M_QAM)
            sigma_cluster[:,cluster]=clustern*N0+beta_c*mean_vec
       ##mpmath
        mt.dps=bit_precision_mp
        sigma_val_m=mt.matrix(sigma_cluster)
        v_val_m=mt.matrix(sigma_cluster)
        sigma_c_m=mt.matrix(sigma_val_m.rows,1)
        z_clus_m=mt.matrix(z_cluster)
        x_est_m=mt.matrix(z_clus_m.rows,1)
        for i in range(sigma_val_m.rows):
            for j in range(sigma_val_m.cols):
                sigma_val_m[i,j]=mt.fdiv(1.0,sigma_val_m[i,j])
                sigma_c_m[i,0]=mt.fadd(sigma_c_m[i,0],sigma_val_m[i,j])
            sigma_c_m[i,0]=mt.fdiv(1.0,sigma_c_m[i,0])    
           
        for i in range(sigma_val_m.rows):
           for j in range(sigma_val_m.cols):
               v_val_m[i,j]=mt.fdiv(sigma_val_m[i,j],sigma_c_m[i,0])
               t1=mt.fmul(z_clus_m[i,j],v_val_m[i,j])
               x_est_m[i,0]=mt.fadd(x_est_m[i,0],t1)
        x_est_np=np.complex128(x_est_m)
        var_np=np.complex128(sigma_c_m)
        for itern in range(K):
            val=FG_mp(x_est_np[itern],clustern*N0+beta_c*var_np[itern],QAM_new)
            x_est_lo[itern]=val[0]
 
        x_est_amp_fd[:,f]= x_est_lo      #Assumed the variance of each cluster is same, so factor (1/clustern)
    return x_est_amp_fd        


def amp_method_fd_work(y,x,M,K,iterations,clustern,SigmaN2):
                    #per user transmit energy from user terminal
    N0=SigmaN2  #noise variance
    N0_c=clustern*N0
    Ex=QAM_var**2
    z_cluster=np.zeros((K,clustern),dtype=A.dtype)
    sigma_cluster=np.zeros((K,clustern),dtype=A.dtype)
    s=np.full(K,Ex,dtype=A.dtype)
    vc=np.zeros(K,dtype=A.dtype) 
    phi_1=np.complex(0)  
    phi_2=np.complex(0)     
#       distributed processing
    x_est_lo=np.zeros(K,dtype=A.dtype)
    a_per_clust=M//clustern
    QAM_new=QAM_map/QAM_var
    x_est_amp_fd=np.zeros(x.shape,dtype=x.dtype)
    beta_c=beta*clustern
    for f in range(Nframes): 
        for cluster in range(clustern):
            H_c=(H[a_per_clust*cluster:a_per_clust*(cluster+1),:].copy())
            y_c=(y[a_per_clust*cluster:a_per_clust*(cluster+1),f].copy())
            ymr_c=np.conjugate(H_c.T) @ y_c
            gmr_c=np.conjugate(H_c.T) @ H_c
            phi_1=0;
            phi_2=0
            s[:]=0
            vc[:]=0
            for itern in range(iterations):
                z_cluster[:,cluster]=ymr_c+(np.identity(K,dtype=np.complex128)-gmr_c) @ s + vc
                mt.dps=bit_precision_mp
                N0_m=mt.mpf(N0_c)
                beta_m=mt.mpc(beta_c)
                phi_1_m=mt.mpc(phi_1)
                t1=mt.fmul(beta_m,phi_1_m)
                t2=mt.fadd(N0_m,t1)
                fg=FG_mp_vector(z_cluster[:,cluster],np.complex128(t2),QAM_new)
                #if(itern!=iterations-1):
                mean_vec=fg[1]
                phi_2=np.mean(mean_vec)
                vc=factor_calculation(beta_c, phi_1,phi_2,N0_c)*(z_cluster[:,cluster]-s)
                s=fg[0]
                phi_1=phi_2.copy()
            sigma_cluster[:,cluster]=N0_c+beta_c*mean_vec
#            sigma_cluster[:,cluster]=variance_cluster(z_cluster[:,cluster],QAM_new,M_QAM)
       ##mpmath
        mt.dps=bit_precision_mp
        sigma_val_m=mt.matrix(sigma_cluster)
        v_val_m=mt.matrix(sigma_cluster)
        sigma_c_m=mt.matrix(sigma_val_m.rows,1)
        z_clus_m=mt.matrix(z_cluster)
        x_est_m=mt.matrix(z_clus_m.rows,1)
        for i in range(sigma_val_m.rows):
            for j in range(sigma_val_m.cols):
                sigma_val_m[i,j]=mt.fdiv(1.0,sigma_val_m[i,j])
                sigma_c_m[i,0]=mt.fadd(sigma_c_m[i,0],sigma_val_m[i,j])
            sigma_c_m[i,0]=mt.fdiv(1.0,sigma_c_m[i,0])    
           
        for i in range(sigma_val_m.rows):
           for j in range(sigma_val_m.cols):
               v_val_m[i,j]=mt.fdiv(sigma_val_m[i,j],sigma_c_m[i,0])
               t1=mt.fmul(z_clus_m[i,j],v_val_m[i,j])
               x_est_m[i,0]=mt.fadd(x_est_m[i,0],t1)
        x_est_np=np.complex128(x_est_m)
        var_np=np.complex128(sigma_c_m)
        for itern in range(K):
            val=FG_mp(x_est_np[itern],N0_c+beta*var_np[itern],QAM_new)
            x_est_lo[itern]=val[0]
 
        x_est_amp_fd[:,f]= x_est_lo      #Assumed the variance of each cluster is same, so factor (1/clustern)
    return x_est_amp_fd        






def amp_method_pd(y,x,M,K,iterations,clustern,SigmaN2):
                    #per user transmit energy from user terminal
    Ex=(1/M_QAM)*np.sum(np.square(np.abs(QAM_map)))
    SNR_amp=1/np.float64(SigmaN2)  #noise variance
   # N0=(beta*Ex)/SNR_amp
    N0=SigmaN2
    #np.float64(SigmaN2)
    
    
    #N0=SigmaN2  
    x_est_amp=np.zeros(x.shape,dtype=x.dtype)
    s=np.full((K,1),np.mean(QAM_map),dtype=A.dtype) #
    phi_1=np.complex128(0)
    phi_2=np.complex128(0)
    y_c=np.zeros((M//clustern,1),dtype=A.dtype) 
    vc=np.full((K,1),QAM_var,dtype=A.dtype)
    a_per_clust=M//clustern
    H_c=np.zeros((a_per_clust,K),dtype=A.dtype)
    y_mrc_1=np.zeros((K),dtype=A.dtype)
    g_mrc_1=np.zeros((K,K),dtype=A.dtype)
    z_cluster=np.zeros((K,1),dtype=A.dtype)
    x_est_lo=np.zeros(K,dtype=A.dtype)
    mean_vec=np.zeros(K,dtype=A.dtype)
    mean_vec_p=np.zeros(K,dtype=A.dtype)
    #lambda_val=np.ones((clustern),dtype=A.dtype)
   
    QAM_new=QAM_map/QAM_var
   # beta=K/M
    
    #performance parameters
    
    #frame loop goes here
    
#       distributed processing
    for f in range(Nframes): 
        y_mrc_1[:]=0
        g_mrc_1[:]=0
        z_cluster[:]=0
        s[:]=0
        vc[:]=0
        phi_2=0
        for cluster in range(clustern):
            H_c=H[a_per_clust*cluster:a_per_clust*(cluster+1),:].copy()
            y_c=y[a_per_clust*cluster:a_per_clust*(cluster+1),f].copy()
            y_mrc_1=y_mrc_1+(np.conjugate(H_c.T) @ y_c)
            g_mrc_1=g_mrc_1+(np.conjugate(H_c.T) @ H_c)
            #x_cluster[:,[cluster]]=np.linalg.inv((np.conjugate(H_c.T) @ H_c)+(N0/Ex)*np.identity(K)) @ y_mrc[:,[cluster]]
            
        phi_1=QAM_var    
        for itern in range(iterations):
            z_cluster[:,0]=y_mrc_1+(np.identity(K,dtype=np.complex128)-g_mrc_1) @ s[:,0] + vc[:,0]
            mt.dps=bit_precision_mp
            N0_m=mt.mpf(N0)
            beta_m=mt.mpc(beta)
            phi_1_m=mt.mpc(phi_1)
            t1=mt.fmul(beta_m,phi_1_m)
            t2=mt.fadd(N0_m,t1)
            fg=FG_mp_vector(z_cluster[:,0],np.complex128(t2),QAM_new)
            #if(itern!=iterations-1):
            mean_vec=fg[1]
            phi_2=np.mean(mean_vec)
            vc[:,0]=factor_calculation(beta, phi_1,phi_2,N0)*(z_cluster[:,0]-s[:,0])
            s[:,0]=fg[0]
            phi_1=phi_2.copy()
            
        
        
        ##calculation of log likelihood
        for itern in range(K):
            val=FG_mp(z_cluster[itern,0],N0+beta*mean_vec[itern],QAM_new)
            x_est_lo[itern]=val[0]

        #centralized processing
           
        x_est_amp[:,f]=x_est_lo   #Assumed the variance of each cluster is same, so factor (1/clustern)
    
    return x_est_amp        


   # xhat=sign(real(xhat))+sqrt(-1)*sign(imag(xhat));

def ep_method(y,x,M,K,iterations,clustern,SigmaN2):
    episilon=np.nextafter(np.float64(0),np.float64(1))
    Ex=1                #per user transmit energy from user terminal
    N0=SigmaN2  #noise variance
      #noise variance
    #N0=SigmaN2  
    x_est_cd=np.zeros(x.shape,dtype=x.dtype)
    s=np.full((K,clustern),np.mean(QAM_map),dtype=A.dtype) #
    eta_c=np.zeros(clustern,dtype=A.dtype)
    omega_0=1/Ex
    y_c=np.zeros((M//clustern,1),dtype=A.dtype) 
    vc=np.full((K,clustern),QAM_var,dtype=A.dtype)
    a_per_clust=M//clustern
    H_c=np.zeros((a_per_clust,K),dtype=A.dtype)
    x_c=np.zeros((K,clustern),dtype=A.dtype)
    x_0=np.zeros(K,dtype=A.dtype)
    gamac=np.zeros(K,dtype=A.dtype)
    
    #lambda_val=np.ones((clustern),dtype=A.dtype)
   
    QAM_new=QAM_map/QAM_var
   # beta=K/M
    
    #performance parameters
    
    #frame loop goes here
    
#       distributed processing
    for f in range(Nframes): 
        for itern in range(iterations):
            for cluster in range(clustern):
                H_c=H[a_per_clust*cluster:a_per_clust*(cluster+1),:].copy()
                y_c=y[a_per_clust*cluster:a_per_clust*(cluster+1),f].copy()
                tc=omega_0-eta_c[cluster]
                gamac=(omega_0*x_0-eta_c[cluster]*x_c[:,cluster])/(tc)
                
                Sigmac_0=((np.conjugate(H_c.T) @ H_c)/SigmaN2)+tc*np.identity(K)
                Sigmac=np.linalg.inv(Sigmac_0)
                
                x_c_0=Sigmac @ (((np.conjugate(H_c.T) @ y_c)/SigmaN2)+tc*gamac)
                
                omega_c=K/(np.trace(Sigmac))
                
                eta_c[cluster]=omega_c-tc
                x_c[:,cluster]=(omega_c*x_c_0-tc*gamac)/(eta_c[cluster])
          #centralized processing
            t0=np.sum(eta_c[:])
            gama0=(x_c @ eta_c)/(t0+episilon)
            fg=FG_mp_vector(gama0,1/t0,QAM_new)
            x_0=fg[0]
            v_0=fg[1] 
            omega_0=K/(np.sum(v_0)+episilon)           
        x_est_cd[:,f]=x_0       #Assumed the variance of each cluster is same, so factor (1/clustern)
    return x_est_cd        


   # xhat=sign(real(xhat))+sqrt(-1)*sign(imag(xhat));

def ep_method1(y,x,M,K,iterations,clustern,SigmaN2):
    episilon=np.nextafter(np.float64(0),np.float64(1))
    Ex=1                #per user transmit energy from user terminal
    N0=SigmaN2  #noise variance
      #noise variance
    #N0=SigmaN2  
    x_est_cd=np.zeros(x.shape,dtype=x.dtype)
    s=np.full((K,clustern),np.mean(QAM_map),dtype=A.dtype) #
    eta_c=np.zeros(clustern,dtype=np.float64)
    omega_0=1/Ex
    y_c=np.zeros((M//clustern,1),dtype=A.dtype) 
    vc=np.full((K,clustern),QAM_var,dtype=A.dtype)
    a_per_clust=M//clustern
    H_c=np.zeros((a_per_clust,K),dtype=A.dtype)
    x_c=np.zeros((K,clustern),dtype=A.dtype)
    x_0=np.zeros(K,dtype=A.dtype)
    gamac=np.zeros(K,dtype=A.dtype)
    
    #lambda_val=np.ones((clustern),dtype=A.dtype)
   
    QAM_new=QAM_map/QAM_var
   # beta=K/M
    
    #performance parameters
    
    #frame loop goes here
    
#       distributed processing
    
    tc=0
    mt.dps = bit_precision_mp
    for f in range(Nframes): 
        for itern in range(iterations):
            eta_c_sum=mt.mpc(0)
            for cluster in range(clustern):
                H_c=H[a_per_clust*cluster:a_per_clust*(cluster+1),:].copy()
                y_c=y[a_per_clust*cluster:a_per_clust*(cluster+1),f].copy()
                
                tc_m=mt.mpc(0)
                
                omega_0_m=mt.mpc(omega_0)
                eta_c_m=mt.mpc(eta_c[cluster])
                tc_m=mt.fsub(omega_0_m,eta_c_m)
                
                x_0_m=mt.matrix(x_0)
                x_c_m=mt.matrix(x_c[:,cluster])
                tcgc_m=mt.matrix(x_0)
                gamac_m=mt.matrix(x_c_m.rows,1)
                for i in range(x_c_m.rows):
                    gamac_m[i,0]=mt.fsub(mt.fmul(omega_0_m,x_0_m[i,0]),mt.fmul(eta_c_m,x_c_m[i,0]))
                    gamac_m[i,0]=mt.fdiv(gamac_m[i,0],tc_m)
                    tcgc_m[i,0]=mt.fmul(tc_m,gamac_m[i,0])
                
                tc=np.complex128(tc_m)
                gamac=np.complex128(gamac_m) 
                tcgc=np.complex128(tcgc_m)
                
                Sigmac_0=((np.conjugate(H_c.T) @ H_c)/SigmaN2)+tc*np.identity(K)
                Sigmac=np.linalg.inv(Sigmac_0)
                x_c_0=Sigmac @ (((np.conjugate(H_c.T) @ y_c)/SigmaN2)+tcgc)
                
                tr_m=mt.mpc(np.trace(Sigmac))
                omega_c_m=mt.mpc(0)
                omega_c_m=mt.fdiv(K,tr_m)
                eta_c_m=mt.fsub(omega_c_m,tc_m)
                episilon=mt.mpf('1e-384')
                eta_c_m=mt.fadd(eta_c_m,episilon)
                
                x_c_0_m=mt.matrix(x_c_0)
                for i in range(x_c_0_m.rows):
                    x_c_m[i,0]=mt.fsub(mt.fmul(omega_c_m,x_c_0_m[i,0]),tcgc[i])
                    x_c_m[i,0]=mt.fdiv(x_c_m[i,0],eta_c_m)
                eta_c[cluster]=np.complex128(eta_c_m)    
                eta_c_sum=mt.fadd(eta_c_sum,eta_c_m)
                x_c[:,cluster]=np.complex128(x_c_m)
                
          #centralized processing
            t0_m=mt.mpc(0)
            t0_m=mt.fdiv(1.0,eta_c_sum)
            t0=np.complex128(t0_m)
            gama0=(x_c @ eta_c)
            gama0=gama0*t0
            
            fg=FG_mp_vector(gama0,t0,QAM_new)
            
            x_0=fg[0]
            v_0=fg[1] 
            omega_0_m1=mt.mpc(0)
            v_0_sum_m=mt.mpc(0)
            
            for i in range(K):
                v_0_m=mt.mpc(v_0[i])
                v_0_sum_m=v_0_sum_m+v_0_m
            omega_0_m1=mt.fdiv(K,v_0_sum_m)  
            omega_0=np.complex128(omega_0_m1)
            
        x_est_cd[:,f]=x_0       #Assumed the variance of each cluster is same, so factor (1/clustern)
    return x_est_cd        




def sgd_method_course(y,x,M,K,H,iteration,clusters,alpha_sgd):
    data_path1=np.zeros((K,clusters),dtype=np.complex128)
    data_path2=np.zeros((K,clusters),dtype=np.complex128)
    x_est_sgd_final=np.zeros(x.shape,dtype=x.dtype)
    x_est_sgd_l=np.zeros((K,1),dtype=np.complex128)
    MperC=M//clusters
    for f in range(Nframes):
        for i in range(iteration): #iterations
            for cl in range(clusters):  #cluster loop
                 Hc=H[MperC*cl:MperC*(cl+1),:].copy();
                 yc=y[MperC*cl:MperC*(cl+1),[f]].copy();
                 if(i==0):
                     x_est_sgd_l=(np.conjugate(Hc.T) @ yc)/(np.power(np.linalg.norm(Hc,axis=0),2)).reshape(K,1)
                     data_path2[:,[cl]]=f_gradient_local(Hc,x_est_sgd_l,yc)
                 else:    
                     data_path2[:,[cl]]=f_gradient_local(Hc,data_path1[:,[cl]],yc)
                 if(cl==(clusters-1)): #last cluster
                     g1=np.average(data_path2,axis=1).reshape(data_path2.shape[0],1)
                     x_est_sgd_l=x_est_sgd_l-alpha_sgd*g1
                     for bcast in range(clusters):
                         data_path1[:,[bcast]]=x_est_sgd_l
        x_est_sgd_final[:,f]=x_est_sgd_l.reshape(K)
    return x_est_sgd_final






#sqn definitions
    

def f2_gradient_local_sqn(g1,g1_temp):
    return g1-g1_temp



def dsgo_method_star_sqn(y,x,M,K,H,iteration,clusters):
    data_path1=np.zeros((K,clusters),dtype=np.complex128)
    data_path2=np.zeros((K,clusters),dtype=np.complex128)
    g2_temp=np.zeros((K,clusters),dtype=np.complex128)
    x_est_sgd_final=np.zeros(x.shape,dtype=x.dtype)
    x_est_sgd_l=np.zeros((K,1),dtype=np.complex128)
    MperC=M//clusters
    for f in range(Nframes):
        for i in range(iteration): #iterations
            for cl in range(clusters):  #cluster loop
                 Hc=H[MperC*cl:MperC*(cl+1),:].copy();
                 yc=y[MperC*cl:MperC*(cl+1),[f]].copy();
                 if(i==0):
                     x_est_sgd_l=(np.conjugate(Hc.T) @ yc)/(np.power(np.linalg.norm(Hc,axis=0),2)).reshape(K,1)
                     data_path2[:,[cl]]=f_gradient_local(Hc,x_est_sgd_l,yc)
                     data_path1[:,[cl]]=f2_gradient_local_sqn(Hc,data_path2[:,[cl]],g2_temp[:,[cl]])
                 else:    
                     data_path2[:,[cl]]=f_gradient_local(Hc,data_path1[:,[cl]],yc)
                 if(cl==(clusters-1)): #last cluster
                     if(i==0):
                         g2=np.average(data_path1,axis=1).reshape(data_path1.shape[0],1)
                     g1=np.average(data_path2,axis=1).reshape(data_path2.shape[0],1)
                     x_est_sgd_l=x_est_sgd_l-(g1/g2)
                     for bcast in range(clusters):
                         data_path1[:,[bcast]]=x_est_sgd_l
        x_est_sgd_final[:,f]=x_est_sgd_l.reshape(K)
    return x_est_sgd_final


snr_iter=np.arange(-5,30,2)

time_admm_r1=np.zeros(snr_iter.size)
time_admm_r2=np.zeros(snr_iter.size)
time_admm_r3=np.zeros(snr_iter.size)
symerr_admm_r1=np.zeros(snr_iter.size) 
symerr_admm_r2=np.zeros(snr_iter.size) 

symerr_admm_r3=np.zeros(snr_iter.size)  
time_cd=np.zeros(snr_iter.size)
symerr_cd=np.zeros(snr_iter.size) 
time_sgd=np.zeros(snr_iter.size)
symerr_sgd=np.zeros(snr_iter.size) 
time_zf=np.zeros(snr_iter.size)
symerr_zf=np.zeros(snr_iter.size) 

symerr_bsgd=np.zeros(snr_iter.size) 
symerr_bsgd1=np.zeros(snr_iter.size) 
symerr_bsgd2=np.zeros(snr_iter.size)
symerr_amp_fd=np.zeros(snr_iter.size) 
symerr_amp_pd=np.zeros(snr_iter.size)
symerr_ep=np.zeros(snr_iter.size)  
symerr_mmse=np.zeros(snr_iter.size) 

symerr_all=np.zeros((5,10,snr_iter.size))

scale=np.sqrt(cluster_number)

for K in range(8,10,2):
    H=np.random.normal(loc=0, scale=1, size=(M,K*2)).view(np.complex128)/np.sqrt(2*M)
    A=np.transpose(np.conjugate(H)) @ H   #zero forcing
    mmse_roh=0.01
    beta=np.float64(K/M)
    A_mmse=A+mmse_roh*np.identity(K,dtype=np.complex128)
    d_size=K*Nframes*multiple_frame
    for i in range(snr_iter.size):
       # print(snr_iter[i])
        for fr in range(multiple_frame):
           # print(fr)
            print('U={} SNR={} FRAME={}'.format(K,snr_iter[i],fr))
            x=generate_data(M_QAM,K,Nframes)  
            y=simulate_receive_signal(x[0],snr_iter[i],scale,H)
            start_time=t.time()
            roh=np.float32(calculate_SigmaN2(snr_iter[i])/10.0)
            alpha=np.float32(1/1)
            betta=np.float32(1*roh/10.0)
            x_est=admm_method(y,x[0],M,K,set_iteration,cluster_number,roh,alpha,betta,calculate_SigmaN2(snr_iter[i]))
            time_admm_r3[i]=(t.time()-start_time)/Nframes
            symerr_admm_r3[i]+=calculate_symbol_error_rate(x[2]*x_est.flatten(),x[1].flatten())
            start_time=t.time()
            x_est=cd_method(y,x[0],M,K,set_iteration,cluster_number,calculate_SigmaN2(snr_iter[i]))
            time_cd[i]=(t.time()-start_time)/Nframes
            symerr_cd[i]+=calculate_symbol_error_rate(x[2]*x_est.flatten(),x[1].flatten())
            start_time=t.time()
            x_est=sgd_method(y,x[0],M,K,snr_iter[i])
            time_sgd[i]=(t.time()-start_time)/Nframes
            symerr_sgd[i]+=calculate_symbol_error_rate(x[2]*x_est.flatten(),x[1].flatten())
            x_est=dsgo_method_ring(y,x[0],M,K,H,set_iteration,cluster_number)
            symerr_bsgd[i]+=calculate_symbol_error_rate(x[2]*x_est.flatten(),x[1].flatten())
            x_est=dsgo_method_ring_fixed(y,x[0],M,K,H,set_iteration,cluster_number)
            symerr_bsgd1[i]+=calculate_symbol_error_rate(x[2]*x_est.flatten(),x[1].flatten())
            x_est=amp_method_fd_work(y,x[0],M,K,set_iteration,cluster_number,calculate_SigmaN2(snr_iter[i]))
            symerr_amp_fd[i]+=calculate_symbol_error_rate(x[2]*x_est.flatten(),x[1].flatten())
            x_est=amp_method_pd(y,x[0],M,K,set_iteration,cluster_number,calculate_SigmaN2(snr_iter[i]))
            symerr_amp_pd[i]+=calculate_symbol_error_rate(x[2]*x_est.flatten(),x[1].flatten())
            x_est=ep_method(y,x[0],M,K,set_iteration,cluster_number,calculate_SigmaN2(snr_iter[i]))
            symerr_ep[i]+=calculate_symbol_error_rate(x[2]*x_est.flatten(),x[1].flatten())
            x_est=zf_method(y,x[0],A)
            symerr_zf[i]+=calculate_symbol_error_rate(x[2]*x_est.flatten(),x[1].flatten())
        symerr_admm_r3[i]=symerr_admm_r3[i]/d_size  
        symerr_cd[i]= symerr_cd[i]/d_size 
        symerr_sgd[i]=symerr_sgd[i]/d_size 
        symerr_zf[i]=symerr_zf[i]/d_size 
        symerr_bsgd[i]=symerr_bsgd[i]/d_size 
        symerr_bsgd1[i]=symerr_bsgd1[i]/d_size 
        symerr_amp_fd[i]=symerr_amp_fd[i]/d_size 
        symerr_amp_pd[i]=symerr_amp_pd[i]/d_size
        symerr_ep[i]=symerr_ep[i]/d_size  
        symerr_mmse[i]=symerr_mmse[i]/d_size
    symerr_all[K//2,:,:]=np.array([snr_iter,symerr_admm_r3,symerr_cd,symerr_sgd,symerr_bsgd,symerr_bsgd1,symerr_amp_fd,symerr_amp_pd,symerr_ep,symerr_zf])
     
     
    




pl.semilogy(snr_iter,symerr_admm_r3,'r-x',label='ADMM Iter={}'.format(set_iteration))
pl.semilogy(snr_iter,symerr_cd,'b-s',label='CD Iter={}'.format(set_iteration))
pl.semilogy(snr_iter,symerr_sgd,'m-^',label='SGD')
pl.semilogy(snr_iter,symerr_bsgd,'g-o',label='DSGO Iter={}'.format(set_iteration))
pl.semilogy(snr_iter,symerr_bsgd1,'y--p',label='DSGO fp Iter={}'.format(set_iteration))
pl.semilogy(snr_iter,symerr_amp_fd,'c--p',label='LAMA-FD'.format(set_iteration))
pl.semilogy(snr_iter,symerr_amp_pd,'c--^',label='LAMA-PD'.format(set_iteration))
pl.semilogy(snr_iter,symerr_ep,'y--o',label='EP'.format(set_iteration))
pl.semilogy(snr_iter,symerr_zf,'k-*',label='ZF',alpha=0.7)



pl.title('Symbol error rate: U={} B={} C={} QAM={}'.format(K,M,cluster_number,M_QAM))  
pl.xlabel('SNR')
pl.ylabel('Symbol error rate')
pl.legend()



pl.grid(True)








