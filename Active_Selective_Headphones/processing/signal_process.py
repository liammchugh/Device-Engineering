import time
import machine
import numpy as np
from scipy import signal
from machine import Pin, Timer,ADC, SoftI2C, UART
from ulab import numpy as np

uart = UART(1,baudrate=115200, tx=1, rx=3)

activated=0
timerFreq= 48000
FFTFreq = 1
i2cFreq=480000
windowsize=256
threshsize=5
ADCsize = 4096
DACsize=2^16
ADCoffset = ADCsize/2
mu=0.01

FFL_PIN =32
FBL_PIN = 33
FFR_PIN = 34
FBR_PIN =35
FIVE_PIN =36
SCL_PIN = 22
SDA_PIN = 21
OUTA =0
OUTB = 1
OUTC = 2
OUTD = 3
DAC_ADDR=0x55
cmd=24

weightsL= np.zeros(windowsize)
weightsR= np.zeros(windowsize)
FFLB= np.zeros(windowsize)
FBLB= np.zeros(windowsize)
FFRB= np.zeros(windowsize)
FBRB= np.zeros(windowsize)
FIVEB= np.zeros(windowsize)
FREQS=np.zeros(windowsize)
THRESHS=np.zeros(threshsize)

i2c=SoftI2C(Pin(SCL_PIN),Pin(SDA_PIN),freq=i2cFreq)

FFL=ADC(Pin(FFL_PIN))
FFL.atten(ADC.ATTN_11DB)
FBL=ADC(Pin(FBL_PIN))
FBL.atten(ADC.ATTN_11DB)
FFR=ADC(Pin(FFR_PIN))
FFR.atten(ADC.ATTN_11DB)
FBR=ADC(Pin(FBR_PIN))
FBR.atten(ADC.ATTN_11DB)
FIVE=ADC(Pin(FIVE_PIN))
FIVE.atten(ADC.ATTN_11DB)


def center(X):
    mean = np.mean(X, axis=1, keepdims=True)
    return X - mean, mean

def whiten(X):
    cov = np.cov(X)
    U, S, V = np.linalg.svd(cov)
    d = np.diag(1.0 / np.sqrt(S))
    whiteM = U @ d @ U.T
    return whiteM @ X, whiteM

def fast_ica(X, alpha=1.0, thresh=1e-8, max_iter=5000):
    n_components = X.shape[0]
    W = np.random.rand(n_components, n_components)
    
    for i in range(n_components):
        w = W[i, :].copy()
        
        for _ in range(max_iter):
            w_new = (X @ np.tanh(alpha * (w @ X.T)).T - np.mean(1 - np.tanh(alpha * (w @ X.T))**2) * w)
            w_new -= np.sum([np.dot(w_new, W[k]) * W[k] for k in range(i)], axis=0)
            w_new /= np.linalg.norm(w_new)
            
            if np.abs(np.abs(np.dot(w_new, w)) - 1) < thresh:
                break
            w = w_new
        
        W[i, :] = w

    return W @ X, W

def ICA_process(X):
    X_centered, mean = center(X)
    X_white, _ = whiten(X_centered)
    S, _ = fast_ica(X_white)
    return (S.T + mean.flatten()).T


def signalProcess(d, weights, buffer):
    y=np.dot(weights,buffer)
    out=(y/ADCsize)*DACsize-1
    norm_buffer=np.linalg.norm(buffer)+1e-8
    weights += mu*(d-y)*buffer/norm_buffer
    return y, weights
    
def FFTprocess(timer):
    global FREQS
    global THRESHS
    real,imag=np.fft.fft(FFLB)
    for i in range(windowsize):
        FREQS[i]=np.sqrt(real[i]**2+imag[i]**2)
    real,imag=np.fft.fft(FFRB)
    for i in range(windowsize):
        FREQS[i]=(np.sqrt(real[i]**2+imag[i]**2)+FREQS[i])/2
    THRESHS=np.sort(FREQS)[windowsize-threshsize:windowsize]
    

def process(timer):
    global weightsL
    global weightsR
    global FFLB
    global FBLB
    global FFRB
    global FBRB
    global FIVEB
    FFLB= np.roll(FFLB,1)
    FBLB= np.roll(FBLB,1)
    FFRB= np.roll(FFRB,1)
    FBRB= np.roll(FBRB,1)
    FIVEB= np.roll(FIVEB,1)
    FFLB[0]=FFL.read() - ADCoffset
    FBLB[0]=FBL.read() - ADCoffset
    FFRB[0]=FFR.read() - ADCoffset
    FBRB[0]=FBR.read() - ADCoffset
    FIVEB[0]=FIVE.read() - ADCoffset
    if(not activated):
        dL=FFLB[0]
        dR=FFRB[0]
    NLMSL, weightsL = signalProcess(dL,weightsL,FBLB)
    NLMSR,weightsR=signalProcess(dR,weightsR, FBRB)
    signals = np.vstack((FFLB, FFRB, FBLB, FBRB, FIVEB))
    separated_signals = ICA_process(signals)
    FFLB, FFRB, FBLB, FBRB, FIVEB = separated_signals
    # i2c.writeto(DAC_ADDR,(int(cmd+OUTA).to_bytes(1,"big")+int(NLMSL).to_bytes(2,"big")))
    # i2c.writeto(DAC_ADDR,(int(cmd+OUTB).to_bytes(1,"big")+int(dL).to_bytes(2,"big")))
    # i2c.writeto(DAC_ADDR,(int(cmd+OUTC).to_bytes(1,"big")+int(NLMSR).to_bytes(2,"big")))
    # i2c.writeto(DAC_ADDR,(int(cmd+OUTD).to_bytes(1,"big")+int(dR).to_bytes(2,"big")))


