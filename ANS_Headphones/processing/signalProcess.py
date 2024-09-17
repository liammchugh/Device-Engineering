
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
    # i2c.writeto(DAC_ADDR,(int(cmd+OUTA).to_bytes(1,"big")+int(NLMSL).to_bytes(2,"big")))
    # i2c.writeto(DAC_ADDR,(int(cmd+OUTB).to_bytes(1,"big")+int(dL).to_bytes(2,"big")))
    # i2c.writeto(DAC_ADDR,(int(cmd+OUTC).to_bytes(1,"big")+int(NLMSR).to_bytes(2,"big")))
    # i2c.writeto(DAC_ADDR,(int(cmd+OUTD).to_bytes(1,"big")+int(dR).to_bytes(2,"big")))