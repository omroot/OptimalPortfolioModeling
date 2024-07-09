#---------------------------------------------------------------
def plot2D(x,y,xLabel='',yLabel='',title='',pathChart=None):
    import matplotlib.pyplot as mpl
    fig=mpl.figure()
    ax=fig.add_subplot(1,1,1) #one row, one column, first plot
    ax.plot(x,y,color='blue')
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel,rotation=90)
    mpl.xticks(rotation='vertical')
    mpl.title(title)
    if pathChart==None:
        mpl.show()
    else:
        mpl.savefig(pathChart)
    mpl.clf() # reset pylab
    return
#---------------------------------------------------------------
def main():
    import numpy as np
    import CLA
    #1) Path
    path='H:/PROJECTS/Data/CLA_Data.csv'
    #2) Load data, set seed
    headers=open(path,'r').readline().split(',')[:-1]
    data=np.genfromtxt(path,delimiter=',',skip_header=1) # load as numpy array
    mean=np.array(data[:1]).T
    lB=np.array(data[1:2]).T
    uB=np.array(data[2:3]).T
    covar=np.array(data[3:])
    #3) Invoke object
    cla=CLA.CLA(mean,covar,lB,uB)
    cla.solve()
    print cla.w # print all turning points
    #4) Plot frontier
    mu,sigma,weights=cla.efFrontier(100)
    plot2D(sigma,mu,'Risk','Expected Excess Return','CLA-derived Efficient Frontier')
    #5) Get Maximum Sharpe ratio portfolio
    sr,w_sr=cla.getMaxSR()
    print np.dot(np.dot(w_sr.T,cla.covar),w_sr)[0,0]**.5,sr
    print w_sr
    #6) Get Minimum Variance portfolio
    mv,w_mv=cla.getMinVar()
    print mv
    print w_mv
    return    
    



    x,y,z,w_=[],[],[],[]
    for i in range(len(cla.w)-1):
        w0=np.copy(cla.w[i])
        w1=np.copy(cla.w[i+1])
        for a in np.linspace(1,0,10000):
            w=a*w0+(1-a)*w1
            w_.append(w)
            x.append(np.dot(np.dot(w.T,cla.covar),w)[0,0]**.5)
            y.append(np.dot(w.T,cla.mean)[0,0])
            z.append(cla.evalSR(a,w0,w1))
    print max(y),w_[z.index(max(z))]
    plot2D(x,y,'Risk','Expected Excess Return','CLA-derived Efficient Frontier', \
        'H:/PROJECTS/TUDOR/Outside Presentations/Papers/CLA/Figures/Figure1.png')
    plot2D(x,z,'Risk','Sharpe ratio','CLA-derived Sharpe Ratio function', \
        'H:/PROJECTS/TUDOR/Outside Presentations/Papers/CLA/Figures/Figure2.png')
#---------------------------------------------------------------
# Boilerplate
if __name__=='__main__':main()