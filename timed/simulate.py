import numpy as np
def simulate_lcamp(snrs,amps=np.ones(4),N=1000,debias=True,snr_calc_N=-1):
    snrs=np.asarray(snrs)
    amps=np.asarray(amps)
    sigmas = amps/snrs
    
    x0 = amps[0] + sigmas[0]*np.random.randn(N) + 1j*sigmas[0]*np.random.randn(N)
    x1 = amps[1] + sigmas[1]*np.random.randn(N) + 1j*sigmas[1]*np.random.randn(N)
    x2 = amps[2] + sigmas[2]*np.random.randn(N) + 1j*sigmas[2]*np.random.randn(N)
    x3 = amps[3] + sigmas[3]*np.random.randn(N) + 1j*sigmas[3]*np.random.randn(N)
    
    if snr_calc_N > 0 :
        x0s = amps[0] + sigmas[0]*np.random.randn(snr_calc_N) + 1j*sigmas[0]*np.random.randn(snr_calc_N)
        x1s = amps[1] + sigmas[1]*np.random.randn(snr_calc_N) + 1j*sigmas[1]*np.random.randn(snr_calc_N)
        x2s = amps[2] + sigmas[2]*np.random.randn(snr_calc_N) + 1j*sigmas[2]*np.random.randn(snr_calc_N)
        x3s = amps[3] + sigmas[3]*np.random.randn(snr_calc_N) + 1j*sigmas[3]*np.random.randn(snr_calc_N)
        snr0 = np.mean(x0s/sigmas[0])
        snr1 = np.mean(x1s/sigmas[0])
        snr2 = np.mean(x2s/sigmas[0])
        snr3 = np.mean(x3s/sigmas[0])
        snrs=np.asarray([snr0,snr1,snr2,snr3])
    
    A0 = np.abs(x0)
    A1 = np.abs(x1)
    A2 = np.abs(x2)
    A3 = np.abs(x3)
    
    if debias:
        A0 = deb_sample(A0,sigmas[0])
        A1 = deb_sample(A1,sigmas[0])
        A2 = deb_sample(A2,sigmas[0])
        A3 = deb_sample(A3,sigmas[0])
    
    mask = (A0>0)&(A1>0)&(A2>0)&(A3>0)
    
    lA0 = np.log(A0[mask])
    lA1 = np.log(A1[mask])
    lA2 = np.log(A2[mask])
    lA3 = np.log(A3[mask])
    
    lcamp = lA0 + lA1 - lA2 - lA3
    #sigma = np.std(lcamp)
    sigma = np.sqrt(np.sum(1./snrs**2))
    return lcamp, sigma


def simulate_camp(snrs,amps=np.ones(4),N=1000,debias=True,snr_calc_N=-1):
    snrs=np.asarray(snrs)
    amps=np.asarray(amps)
    sigmas = amps/snrs
    
    x0 = amps[0] + sigmas[0]*np.random.randn(N) + 1j*sigmas[0]*np.random.randn(N)
    x1 = amps[1] + sigmas[1]*np.random.randn(N) + 1j*sigmas[1]*np.random.randn(N)
    x2 = amps[2] + sigmas[2]*np.random.randn(N) + 1j*sigmas[2]*np.random.randn(N)
    x3 = amps[3] + sigmas[3]*np.random.randn(N) + 1j*sigmas[3]*np.random.randn(N)
    
    if snr_calc_N > 0 :
        x0s = amps[0] + sigmas[0]*np.random.randn(snr_calc_N) + 1j*sigmas[0]*np.random.randn(snr_calc_N)
        x1s = amps[1] + sigmas[1]*np.random.randn(snr_calc_N) + 1j*sigmas[1]*np.random.randn(snr_calc_N)
        x2s = amps[2] + sigmas[2]*np.random.randn(snr_calc_N) + 1j*sigmas[2]*np.random.randn(snr_calc_N)
        x3s = amps[3] + sigmas[3]*np.random.randn(snr_calc_N) + 1j*sigmas[3]*np.random.randn(snr_calc_N)
        snr0 = np.mean(x0s/sigmas[0])
        snr1 = np.mean(x1s/sigmas[0])
        snr2 = np.mean(x2s/sigmas[0])
        snr3 = np.mean(x3s/sigmas[0])
        snrs=np.asarray([snr0,snr1,snr2,snr3])
    
    A0 = np.abs(x0)
    A1 = np.abs(x1)
    A2 = np.abs(x2)
    A3 = np.abs(x3)
    
    if debias:
        A0 = deb_sample(A0,sigmas[0])
        A1 = deb_sample(A1,sigmas[0])
        A2 = deb_sample(A2,sigmas[0])
        A3 = deb_sample(A3,sigmas[0])
    
    mask = (A0>0)&(A1>0)&(A2>0)&(A3>0)
    
    
    camp = A0*A1/A2/A3
    #sigma = np.std(lcamp)
    sigma = camp*np.sqrt(np.sum(1./snrs**2))
    return camp, sigma


def simulate_cphase(snrs,amps=np.ones(3),N=1000,snr_calc_N=-1):
    snrs=np.asarray(snrs)
    amps=np.asarray(amps)
    sigmas = amps/snrs
    
    x0 = amps[0] + sigmas[0]*np.random.randn(N) + 1j*sigmas[0]*np.random.randn(N)
    x1 = amps[1] + sigmas[1]*np.random.randn(N) + 1j*sigmas[1]*np.random.randn(N)
    x2 = amps[2] + sigmas[2]*np.random.randn(N) + 1j*sigmas[2]*np.random.randn(N)
    
    if snr_calc_N > 0 :
        x0s = amps[0] + sigmas[0]*np.random.randn(snr_calc_N) + 1j*sigmas[0]*np.random.randn(snr_calc_N)
        x1s = amps[1] + sigmas[1]*np.random.randn(snr_calc_N) + 1j*sigmas[1]*np.random.randn(snr_calc_N)
        x2s = amps[2] + sigmas[2]*np.random.randn(snr_calc_N) + 1j*sigmas[2]*np.random.randn(snr_calc_N)
        snr0 = np.mean(x0s/sigmas[0])
        snr1 = np.mean(x1s/sigmas[0])
        snr2 = np.mean(x2s/sigmas[0])
        snrs=np.asarray([snr0,snr1,snr2])
    
    B=x0*x1*x2
    cp=np.angle(B)
      
    sigma = np.sqrt(np.sum(1./snrs**2))
    return cp, sigma

def deb_sample(amps,sigma):
    amps = np.asarray(amps)
    amps_deb = (amps**2 - sigma**2)
    amps_deb = np.sqrt(np.maximum(0,amps_deb))
    return amps_deb