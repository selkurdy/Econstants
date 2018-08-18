import sys, os.path
import argparse
import numpy as np
import numpy.ma as ma
import  math as m
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import leastsq
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
"""
16/Apr/2014 7:46 Computation of elastic constants from las files
    needed Vp, Vs, density (rho)
sample input
   9669.50000       8.50000       7.00000       8.48822      49.31210      49.53606      89.88329      88.17348       7.74350    9780.73438       2.95125    -999.25000
   9670.00000       8.50000       7.00000       8.49448      49.26857      49.49063      89.67466      87.96484       7.49536    9781.23438       2.95132    -999.25000
   9670.50000       8.50000       7.00000       8.49077      49.04301      49.25521      89.46601      87.75620       8.54917    9781.73438       2.94848    -999.25000
   9671.00000       8.50000       7.00000       8.48380      48.78158      48.98235      89.25738      87.54756      12.48408    9782.23438       2.94232    -999.25000
   9671.50000       8.50000       7.00000       8.47181      48.79290      48.99416      89.04873      87.33892      17.32290    9782.73438       2.93823    -999.25000


sample output
    9669.50       20279       11341         3.0       379.6        0.27         966         454         708
   9670.00       20297       11368         3.0       381.4        0.27         970         453         707
   9670.50       20390       11395         2.9       382.9        0.27         975         460         715
   9671.00       20500       11422         2.9       383.9        0.27         979         469         725
   9671.50       20495       11450         2.9       385.2        0.27         981         464         721
        ^               ^               ^               ^           ^              ^              ^             ^            ^
    depth          Vp             Vs            RhoB    Mu           PR           E          Lambda   Bulk K





0. Convert velocity to cm/sec.
       Vp = 14000 * 0.3048 * 100 = 426720 cm/sec

1. Invert the velocity equation and solve for Kc (sometimes referred to as the space modulus M) for 100% water-bearing condition.
      Kc = (Vp ^ 2) * DENS = 426750 * 426720 * 2.44 = 44.5*10^10 dynes/cm2 = 44.5 GPa

+to read in a file using -f specifying --delimiter comma. This does not use Pandas data frames
C:\\Users\20504\sami\SEKDATA\wdata>python econstants.py --datafilename nf90.csv -d 5 6 7 4 -n -999.25 --delimiter , --tompsi  >nf9_econst.lst

+To read in a csv file using --csvfilename and ouotputting a default dataout.csv file.
    This uses Pandas dataframe of specific format
C:\\Users\20504\sami\SEKDATA\wdata>python econstants.py --csvfilename nf90.csv  --tompsi
MD(ft),TWTP(s),AIP(fsgc),INTRHO(g/cc),RHOB(g/cc),TVDSS(ft),DTP(?s/ft),DTS(?s/ft),VPVS,PR,YME(Mpsi),BMK(Mpsi),SMG(Mpsi)
106,0.00008,5228.21777,0,1.03,0.19995,199,-999.25,-999.25000,-999.25000,-999.25000,-999.25000,-999.25000
106.5,0.00028,5228.21777,0.00515,1.03,0.69995,199,-999.25,-999.25000,-999.25000,-999.25000,-999.25000,-999.25000
107,0.00048,5228.21777,0.0103,1.03,1.19995,199,-999.25,-999.25000,-999.25000,-999.25000,-999.25000,-999.25000
107.5,0.00068,5228.21777,0.01545,1.03,1.69995,199,-999.25,-999.25000,-999.25000,-999.25000,-999.25000,-999.25000
108,0.00088,5228.21777,0.0206,1.03,2.19995,199,-999.25,-999.25000,-999.25000,-999.25000,-999.25000,-999.25000
11070,1.65539,75718.91406,273.43024,2.9789,10963.15234,39.55607,78.3415,1.98052,0.32891,17.38237,16.93304,6.54008
11070.5,1.65544,74506.40625,273.44507,2.96786,10963.65234,40.10707,79.20362,1.97480,0.32758,16.92599,16.36096,6.37477
11071,1.65549,72929.48438,273.45984,2.95332,10964.15234,40.8481,80.35526,1.96717,0.32577,16.34146,15.63207,6.16301
11071.5,1.6555,71016.50781,273.47452,2.93531,10964.65234,41.78531,81.79908,1.95760,0.32346,15.64620,14.77112,5.91110
11072,1.6555,68879.125,273.48911,2.91466,10965.15234,42.87943,83.46696,1.94655,0.32073,14.89063,13.84361,5.63728

+pandas commands to edit csv file in ipython to remove unnecessary columns then save to csv again
        df= pd.read_csv(cmdl.csvfilename,header=0,\
        names=['MD','TWT','AIP','INTRHOB','RHOB','TVDSS','DTP','DTS','VPVS','PR','YME','BMK','SMG'])<---------
        #print df.head(10)

Data columns (total 13 columns):
MD(ft)          21933  non-null values
TWTP(s)         21933  non-null values
AIP(fsgc)       21933  non-null values
INTRHO(g/cc)    21933  non-null values
RHOB(g/cc)      21933  non-null values
TVDSS(ft)       21933  non-null values
DTP(?s/ft)      21933  non-null values
DTS(?s/ft)      21933  non-null values
VPVS            21933  non-null values
PR              21933  non-null values
YME(Mpsi)       21933  non-null values
BMK(Mpsi)       21933  non-null values
SMG(Mpsi)       21933  non-null values
dtypes: float64(13)

In [16]: df1.columns =['MD','TWTP','AIP','INTRHO','RHOB','TVDSS','DTP','DTS','VP
VS','PR','YME','BMK','SMG']   <---------

In [18]: df2 = df1[['TVDSS','DTP','DTS','RHOB']].dropna()<---------

Int64Index: 21933 entries, 0 to 21932
Data columns (total 4 columns):
TVDSS    21933  non-null values
DTP      21933  non-null values
DTS      21933  non-null values
RHOB     21933  non-null values
dtypes: float64(4)


In [20]: df3 = df2[df2['DTS'] != -999.2500] <---------

In [21]: df3
Out[21]:
<class 'pandas.core.frame.DataFrame'>
Int64Index: 21603 entries, 330 to 21932
Data columns (total 4 columns):
TVDSS    21603  non-null values
DTP      21603  non-null values
DTS      21603  non-null values
RHOB     21603  non-null values
dtypes: float64(4)

In [22]: df3.head(10)  <---------
Out[22]:
         TVDSS        DTP        DTS     RHOB
330  165.19914  150.00000  550.00000  1.80000
331  165.69914  150.01727  548.36652  1.80083
332  166.19914  150.03455  546.73303  1.80166
333  166.69914  150.05180  545.09955  1.80249
334  167.19914  150.06908  543.46606  1.80332
335  167.69911  150.08635  541.83258  1.80415
336  168.19911  150.10362  540.19910  1.80498
337  168.69911  150.12090  538.56561  1.80581
338  169.19911  150.13815  536.93207  1.80663
339  169.69911  150.15543  535.29858  1.80746

In [23]: df3.to_csv('wellxx.csv',index=False)  <---------

+ 'wellxx.csv' is the file to read into  this program using --csvfilename
C:\\Users\20504\sami\SEKDATA\wdata>python econstants.py  -f wellxx.csv -d 0 1 2 3 -r 1 --delimiter ,

+ Added xplots of attributes vs tvdss and attributes  color coded by tvdss

+ April 0215 added TVDSS range selection and plotting with color legend
C:\\Users\20504\sami\SEKDATA\wdata>python econstants.py  --csvfilename wellxx.csv --plotrange 5800 7500

C:\\Users\20504\sami\SEKDATA\wdata>python econstants.py  --csvfilename wellxx.csv --plotrange 5800 7500 --lstsqrfit  --fitdegree 2

+ added extra 4 plots and degree fit option. Romoved equation annotation: cannot locate xy from df??

+Apr0515 final version: added list comprehension to add coefficients to lists thn convert to numpy array then to data frame
    Computer polynomial coefficients are saved to a csv file
    output csv files include input file name plus identifier.csv
C:\\Users\20504\sami\SEKDATA\wdata>python econstants.py  --csvfilename wellxx.csv --plotrange 8000 10900 --fitdegree 1  --leastsqrfit

C:\\Users\20504\sami\SEKDATA\wdata>python econstants.py  --csvfilename wellxx.csv  --fitdegree 1  --leastsqrfit

python econstants.py --datafilename BAH010.las --headerlines 43 --datacols 0 3 4 1 --velocity
"""

def gmcc2lbcft(gmcc):
    return gmcc  * 62.427961

def lbcft2gmcc(lbcft):
    return lbcft / 62.427961

def m2ft(m):
    return m * 3.2808

def ft2m(ft):
    return ft /3.2808

def comp_bulkmod(lame,mu):
    bulk= lame + (2.0 /3.0) *mu
    return bulk


def comp_dfbulkmod(df):
    bulk= df['LAME'] + (2.0 /3.0) *df['MU']
    return bulk


#frm web site efunda.com works OK
def comp_lame1(mu,pr):
    lame= (2.0 * mu * pr) /(1.0 - 2.0 * pr)
    return lame

def comp_dflame1(df):
    lame= (2.0 * df['MU'] * df['PR']) /(1.0 - 2.0 * df['PR'])
    return lame


#from SEG Multicomponent  book does not work
def comp_lame0(vp,vs,rho):
    lame=rho *( vp * vp - 2.0 * vs *vs)
    return lame




def comp_pr(vp,vs):
    vpvsrsqr= (vp/vs) * (vp/vs)
    pr=((vpvsrsqr -2.0)  / (vpvsrsqr -1.0) )/2.0
    return pr

def comp_dfpr(df):
    vpvsrsqr= (df['VP']/df['VS']) ** 2
    pr=((vpvsrsqr -2.0)  / (vpvsrsqr -1.0) )/2.0
    return pr

#mixed unit shear velocity in ft/s, densith in gm/cc
def comp_mu(vs,rho,f2m=True):
    if f2m:
        vs_cmsec= vs /3.2808 * 100.0 # convert ft/s to cm/sec
    else:
        vs_cmsec= vs * 100.0 # convert ft/s to cm/sec
    
    mu_dynescmsqr =rho * vs_cmsec * vs_cmsec
    mu_gpa = mu_dynescmsqr / m.pow(10,10)
    return mu_gpa


def comp_dfmu(df):

    vs_cmsec= df['VS'] /3.2808 * 100.0 # convert ft/s to cm/sec
    mu_dynescmsqr = df['RHOB'] * vs_cmsec * vs_cmsec
    mu_gpa = mu_dynescmsqr / m.pow(10,10)


    return mu_gpa



def comp_young(mu,pr):
    y=2.0 * mu *(1.0 + pr)
    return y


def comp_dfyoung(df):
    y=2.0 * df['MU'] *(1.0 + df['PR'])
    return y


def dtp2vp(df):
    return 1000000.0/df['DTP']

def dts2vs(df):
    return 1000000.0/df['DTS']


def vsvpsqr(vp,vs):
    return (vs/vp) ** 2


def vsvpsqrdf(df):
    return (df['VS']/df['VP']) ** 2

def comp_lambdarho(lame,rho):
    return lame* rho


def comp_dflambdarho(df):
    return df['LAME'] * df['RHOB']

def comp_murho(mu,rho):
    return mu* rho


def comp_dfmurho(df):
    return df['MU'] * df['RHOB']

    
def comp_ai(vp,rho):
        
    return vp * rho
    
    
def comp_relfectivity(vp,rho,metric=False):
    """
    In terms of well log measurements, for near vertical incidence :
    1: Zp1 = KD4 * DENS1 / (DTC1 * KS3)
    2: Zp2 = KD4 * DENS2 / (DTC2 * KS3)
    3: Refl = (Zp2 - Zp1) / (Zp2 + Zp1)
    4: Atten = Prod (1 - Refl ^ 2)

    Where: 
    KD4 = 1000 for Metric units     (DENS in Kg/m3, DTC in usec/m)
    KD4 = 10^6 for English units   (DENS in g/cc, DTC in usec/ft) 
    KS3 = 1.00 for Metric units
    KS3 = 3.281 for English units
    """

    if metric:
        kd4 = 1000.0
        ks3 = 1.0
    else:
        kd4 = 10**6
        ks3 = 3.2808
    zp1 = kd4 * rho[0:-1] * (vp[0:-1]/ks3)
    zp2 = kd4 * rho[1:] * (vp[1:]/ks3)
    rs = (zp2 - zp1) / (zp2 + zp1)  
    atten = np.prod(1.0 - rs **2)
    return rs, atten


    
    
#list data in
def list0(z,vp,vs,rho):
    for i in range(vp.size):
        print("%10.2f  %10.0f  %10.0f  %10.1f" %(z[i],vp[i],vs[i],rho[i]))


#list mu
def list1(z,vp,vs,rho,mu):
    for i in range(vp.size):
        print("%10.2f  %10.0f  %10.0f  %10.1f  %10.0f" %(z[i],vp[i],vs[i],rho[i],mu[i]))


#list poisson ratio
def list2(z,vp,vs,rho,mu,pr):
    for i in range(vp.size):
        print("%10.2f  %10.0f  %10.0f  %10.1f  %10.0f  %10.2f" %(z[i],vp[i],vs[i],rho[i],mu[i],pr[i]))


#list young's modulus
def list3(z,vp,vs,rho,mu,pr,y):
    for i in range(vp.size):
        print("%10.2f  %10.0f  %10.0f  %10.1f  %10.0f  %10.2f  %10.0f" \
        %(z[i],vp[i],vs[i],rho[i],mu[i],pr[i],y[i]))


#list Lame constant
def list4(z,vp,vs,rho,mu,pr,y,lame):
    for i in range(vp.size):
        print("%10.2f  %10.0f  %10.0f  %10.1f  %10.0f  %10.2f  %10.0f  %10.0f" \
        %(z[i],vp[i],vs[i],rho[i],mu[i],pr[i],y[i],lame[i]))




#list Bulk Modulus
def list5(z,vp,vs,rho,vsvp2,mu,pr,y,lame,bulk):
    print("#Z,Vp, Vs, RhoB, (VS/VP)^2, Shear Modulus mu, Poisson's Ratio PR, Young's Modulus E, Lame's constant lambda, Bulk Modulus K")
    for i in range(vp.size):
        print("%10.2f  %10.0f  %10.0f  %10.5f  %10.5f  %10.5f  %10.5f  %10.5f  %10.5f  %10.5f" \
        %(z[i],vp[i],vs[i],rho[i],vsvp2,mu[i],pr[i],y[i],lame[i],bulk[i]))


def list6(z,vp,vs,rho,vsvp2,mu,pr,y,lame,bulk,multp):
    print("#Z ft, Vp ft/s, Vs ft/s, RhoB gm/cc, (VS/VP)^2,Shear Modulus mu GPA, \
    Poisson's Ratio PR, Young's Modulus E GPA, Lame's constant lambda GPA, \
    Bulk Modulus K GPA,Young's Modulus E Mpsi, Lame's constant lambda Mpsi,Bulk Modulus K Mpsi")
    mu_mpsi = mu * multp
    y_mpsi = y * multp
    lame_mpsi = lame * multp
    bulk_mpsi = bulk * multp
    for i in range(vp.size):
        print("%10.2f  %10.0f  %10.0f  %10.5f %10.5f  %10.5f  %10.5f  %10.5f  %10.5f  \
        %10.5f  %10.5f  %10.5f  %10.5f  %10.5f" \
        %(z[i],vp[i],vs[i],rho[i],vsvp2[i],pr[i],mu[i],y[i],lame[i],bulk[i],\
        mu_mpsi[i],y_mpsi[i],lame_mpsi[i],bulk_mpsi[i]))

        
def smooth(a, wlen=11, mode='valid') :
    if wlen % 2 == 0:  #window has to be odd
        wlen +=1
    asmth = savgol_filter(a, wlen, 3) # window size 51, polynomial order 3
    return asmth

def check_nulls(df):
    allcols = df.columns.tolist()
    nullcols =False
    for i,c in enumerate(allcols):
        if df[c].isnull().all():
            print('Column# {}: {} is all null values'.format(i,c))
            nullcols =True
        else:
            ntotal = df[c].isnull().values.sum()
            print('Column# {}: {} has {:d} null values'.format(i,c,ntotal))   
    return nullcols


    
def md2xyz(mdlog,mdsrv,xsrv,ysrv,zsrv):
    """
    pass in all arguments as np.array
    
    """
    indx=np.digitize(mdlog,mdsrv)
    for j in range (0,mdlog.size):
        xlog=xsrv[indx[j]-1]+(((mdlog[j]-mdsrv[indx[j]-1])*(xsrv[indx[j]]-xsrv[indx[j]-1]))/(mdsrv[indx[j]]-mdsrv[indx[j]-1]))
        ylog=ysrv[indx[j]-1]+(((mdlog[j]-mdsrv[indx[j]-1])*(ysrv[indx[j]]-ysrv[indx[j]-1]))/(mdsrv[indx[j]]-mdsrv[indx[j]-1]))
        zlog=zsrv[indx[j]-1]+(((mdlog[j]-mdsrv[indx[j]-1])*(zsrv[indx[j]]-zsrv[indx[j]-1]))/(mdsrv[indx[j]]-mdsrv[indx[j]-1]))
    xyzdf=pd.DataFrame({'X':xlog,'Y':ylog,'Z':zlog})
    return xyzdf    


def md2tvd(mdlog,ds):
    """
    pass in all arguments as np.array
    md 0 x 1 y 2 z 3 
    """
    xsrv = ds[:,1]
    ysrv = ds[:,2]
    zsrv = ds[:,3]
    mdsrv = ds[:,0]
    xlog = np.array(mdlog)
    ylog = np.array(mdlog)
    zlog = np.array(mdlog)
    indx=np.digitize(mdlog,mdsrv)
    for j in range (mdlog.size):
        xlog[j]=xsrv[indx[j]-1]+(((mdlog[j]-mdsrv[indx[j]-1])*(xsrv[indx[j]]-xsrv[indx[j]-1]))/(mdsrv[indx[j]]-mdsrv[indx[j]-1]))
        ylog[j]=ysrv[indx[j]-1]+(((mdlog[j]-mdsrv[indx[j]-1])*(ysrv[indx[j]]-ysrv[indx[j]-1]))/(mdsrv[indx[j]]-mdsrv[indx[j]-1]))
        zlog[j]=zsrv[indx[j]-1]+(((mdlog[j]-mdsrv[indx[j]-1])*(zsrv[indx[j]]-zsrv[indx[j]-1]))/(mdsrv[indx[j]]-mdsrv[indx[j]-1]))
    return xlog,ylog,zlog   


    
    
    

def getcommandline():
    parser = argparse.ArgumentParser(description='Elastic Constants from Vp,Vs,rhob')
    parser.add_argument('--datafilename', help='Flat file with columns of Z Vp Vs RhoB most likely LAS. It will be munged to a proper csv')
    parser.add_argument('--datacols', nargs= 4,type=int,default=[0,1,2,3],
        help='Z Vp Vs  rho columns. Use only with --datafilename. dfv = 0 1 2 3')
    parser.add_argument('--headerlines',type=int,default=38,help='header lines to skip. dfv=38')
    parser.add_argument('--targetcol',type=int,default=None,help='extra column from las to add')
    parser.add_argument('--targetheader',default='Target',help='Target col header.default= Target')
    parser.add_argument('--nan',default=-999.2500,help='Null value. dfv= -999.2500')
    parser.add_argument('--delimiter',default = " ",help='data column delimiter. dfv = space')
    parser.add_argument('--dsfilename',help='Deviation survey file name to convert MD to TVD')
    parser.add_argument('--dsheader',type=int,default=15,help='Deviation survey header lines. default=15' )
    parser.add_argument('--dscols',type=int,nargs=4,default =[0,1,2,3],help='Deviation Survey MD X Y Z columns.default = 0 1 2 3')
    parser.add_argument('--dsflipzsign',action='store_true',default=False,help='Deviation survey flip sign of z values.default: leave as is ')
    
    parser.add_argument('--csvfilename',
            help='Input data file in csv format. Expected only 4 columns: TVDSS,DTP,DTS,RHOB. Only 1 header line.Most likely after munging ')
    parser.add_argument('--velocity',action='store_true',default=False,help='input columns are velocities. dfv= sonic dtp and dts')
    parser.add_argument('--tometric',action='store_true',default=False,help='convert from feet to meters only for VP and VS')
    parser.add_argument('--multiplier',nargs=3,type=float,default=[1.0,1.0,1.0,1.0] ,\
        help='Multipliers for Vp, Vs, rhob, dfv= 1.0 1.0 1.0 1.0')
    parser.add_argument('--shift',required=False,nargs=3, type=float,default=[0.0,0.0,0.0,0.0] ,\
        help='Shifts for xVp, Vs, rhob, dfv=0.0 0.0 0.0 0.0')
    parser.add_argument('--outmultiplier',default=1.0,type=float,help='from gpa to Mpsi multiply by 0.145038' )
    parser.add_argument('--tompsi',action='store_true',default = False,help='List elastic constants in MPSI. dfv=false')
    parser.add_argument('--bakusavlen',type=float,default=46,help='Bakus average length for Bakus filtering. default= 46 m Fmax = Vmax/Lambda min')
    parser.add_argument('--scaleqminmax',type=float,nargs=2,default=[0.0,200.0],help='MinMax Scale range.default = 0 200')
    parser.add_argument('--plotrange',type= float,nargs=2,help='TVDSS range to plot. dfv= all data set')
    parser.add_argument('--leastsqrfit',action='store_true',default=False,help="fit least sqr to data. dfv=False")
    parser.add_argument('--fitdegree',type=int,default=1,help='fit polynomial degree. dfv= 1, straight line')
    parser.add_argument('--hideplots',action='store_true',default=False,help='Hide all plots. dfv =show all plots')


    result=parser.parse_args()
    if not (result.datafilename or result.csvfilename):
        parser.print_help()
        exit()
    else:
        return result





def main():
    cmdl= getcommandline()
    m2f = 3.2808
    
    def devsurvey():
        ds=np.genfromtxt(cmdl.dsfilename,usecols=cmdl.dscols,skip_header=cmdl.dsheader)
        dskb = ds[0,3]
        print('KB {:4.2f}'.format(dskb))
        ds = ds[1:,:]
        if cmdl.dsflipzsign:
            ds[:,3] *= (-1.0)
        dscolnames= ['MD','X','Y','Z']
        dsdf = pd.DataFrame(data= ds,columns = dscolnames)
        dirsplitds,fextsplitds= os.path.split(cmdl.dsfilename)
        fnameds,fextnds= os.path.splitext(fextsplitds)
        dsfnamecsv = fnameds +'_devsur.csv'
        dsdf.to_csv(dsfnamecsv,index=False)
        print('Sucessfully generated {}'.format(dsfnamecsv))
        xl,yl,zl= md2tvd(wx['Z'].values,ds)

        
        
        """
        # Don't need to merge x,y,tvd df to wx logs data
        # Still gives a problem with concat???
        
        logstvdcols = ['TVX','TVY','TVD']
        tvdlogsdf = pd.DataFrame({'TVX':xl,'TVY':yl,'TVD':zl})        
        tvdlogsdf = tvdlogsdf[logstvdcols].copy()
        tvdlogsdf.index = range(len(tvdlogsdf))
        print(tvdlogsdf.shape)
        print(tvdlogsdf.head())
        allcols = tvdlogsdf.columns.tolist() + wx.columns.tolist() 
        
        logstvd = pd.concat([tvdlogsdf,wx],axis=1,ignore_index=True)
        logstvd.columns =allcols
        print(logstvd.head())
        
        logstvdfnamecsv = fname +'_tvd.csv'
        logstvdfnametxt = fname +'_tvd.txt'
        logstvd.to_csv(logstvdfnamecsv,index=False)
        print('Sucessfully generated {}'.format(logstvdfnamecsv))
        logstvd.to_csv(logsfnametxt,index=False,sep = ' ')
        print('Sucessfully generated {}'.format(logstvdfnametxt))
        
        """
        
        return zl

    
    
    
    
    
    #reading in a flat file assumption is las with header lines
    if cmdl.datafilename:
        colheadervel= ['Z','Vp','Vs','Rho']
        colheaderson= ['Z','DTC','DTS','Rho']
        if cmdl.targetcol:
            colheadervel.append(cmdl.targetheader)
            colheaderson.append(cmdl.targetheader)
        #input is ASCII flat file e.g. LAS
        w = pd.read_csv(cmdl.datafilename,delim_whitespace=True,header=0,skiprows= cmdl.headerlines)
        dirsplit,fextsplit= os.path.split(cmdl.datafilename)
        fname,fextn= os.path.splitext(fextsplit)
             
        
        if not cmdl.velocity:
            # cols2keepsonic=[0,17,16,10]
            cols2keepsonic=cmdl.datacols
            if cmdl.targetcol:
                cols2keepsonic.append(cmdl.targetcol)
            coltitles = [w.columns[h] for h in cols2keepsonic ]
            #print(coltitles)
        else:
            # cols2keepvel =[0,13,8,10]
            cols2keepvel =cmdl.datacols
            if cmdl.targetcol:
                cols2keepvel.append(cmdl.targetcol)
            coltitles = [w.columns[h] for h in cols2keepvel ]
            #print(coltitles)
            
        wx = w[coltitles].copy()
        wx[wx == -999.25] = np.nan
        if not check_nulls(wx):
            wx.dropna(axis=0,inplace=True)
            if not cmdl.velocity:
                wx.columns =colheaderson
                if cmdl.tometric:
                    wx['Vp'] = wx['DTC'].apply(lambda x: (1000000./x)/m2f)
                    wx['Vs'] = wx['DTS'].apply(lambda x: (1000000./x)/m2f)
                else:
                    wx['Vp'] = wx['DTC'].apply(lambda x: (1000000./x))
                    wx['Vs'] = wx['DTS'].apply(lambda x: (1000000./x))
                    
                wx.drop(['DTC','DTS'],axis=1,inplace=True)
                wx.insert(1,'VP',wx['Vp'])
                wx.insert(2,'VS',wx['Vs'])
                wx.drop(['Vp','Vs'],inplace=True, axis=1)
                
            else:
                wx.columns=colheadervel
            # ********** deviation survey                
            if cmdl.dsfilename:
                wx['Z']= devsurvey()
                outfname = os.path.join(dirsplit,fname) +"_tvd.csv"
            else:
                outfname = os.path.join(dirsplit,fname) +".csv"
            
            print(wx.head())
            wx.to_csv(outfname,index=False)
            print('Successfully generated {}'.format(outfname))
        else:
            print('***Data has full columns of nulls')

                
    else:
        #assumption is csv has already been munged and is in a good format
        #meaning: 1 header line and only 4 columns depth vp vs rho
        w = pd.read_csv(cmdl.csvfilename)
        if len(w.columns) >4 :
            targetdata = w[w.columns[-1]]
            targethead = w.columns[-1]
        vp = w[w.columns[1]].values
        vs = w[w.columns[2]].values
        rho = w[w.columns[3]]
        z = w[w.columns[0]].values
        mu=comp_mu(vs,rho,f2m=cmdl.tometric)#convert vs from ft/s to m/s to cm/s
        pr=comp_pr(vp,vs)
        vsvp2= vsvpsqr(vp,vs)
        young=comp_young(mu,pr)
        lame = comp_lame1(mu,pr)
        bulkmod= comp_bulkmod(lame,mu)
        lambdarho = comp_lambdarho(lame,rho)
        murho = comp_murho(mu,rho) 
        ai = comp_ai(vp,rho)
        
        
        if cmdl.tompsi:
            cmdl.outmultiplier = 0.14503773772999998   #convert gpa to mega psi

        #*********code for Q computation
        bakusavlen = cmdl.bakusavlen
        dz = np.diff(z[0:5])[0]    
        vavg = np.cumsum(z) / np.cumsum(z/vp)
        vrms = np.sqrt(np.cumsum(z*vp) / np.cumsum(z/vp))
        vbks = np.sqrt(np.cumsum(np.power(z,2)) / (np.cumsum(z*rho) * np.cumsum(z/(rho*np.power(vp,2)))))        
        lame1 = rho * (np.power(vp, 2.0) - 2 * np.power(vs, 2.0)) # Elastic  lambda
        mu1 = rho * np.power(vs, 2.0) #from Agile notebook
        a = c = rho * np.power(vp, 2.0) # Acoustic impedance, same as lambda + 2*mu

        # We don't seem to actually need these other parameters (or c, above, for that matter)
        # f = lam # Remember not to use f for frequency!
        # l = m = mu # Note that these double indentities result in only one object        
        # yhat = savitzky_golay(y, 51, 3) # window size 51, polynomial order 3
        A1 = 4 * smooth(mu1*(lame1 + mu1)/a, wlen = int(cmdl.bakusavlen/dz), mode='same')
        A = A1 + np.power(smooth(lame1/a, wlen = int(cmdl.bakusavlen/dz), mode='same'), 2.0) / smooth(1.0/a, int(cmdl.bakusavlen/dz), mode='same')
        C = 1.0 / smooth(1.0/a, wlen = int(cmdl.bakusavlen/dz), mode='same')
        F = smooth(lame1/a, wlen = int(cmdl.bakusavlen/dz), mode='same') / smooth(1.0/a, wlen = int(cmdl.bakusavlen/dz), mode='same')
        L = 1.0 / smooth(1.0/mu1, wlen = int(cmdl.bakusavlen/dz), mode='same')
        M = smooth(mu1, wlen = int(cmdl.bakusavlen/dz), mode='same')

        R = smooth(rho, int(cmdl.bakusavlen/dz), mode='same')
        vp0 = np.sqrt(C / R)
        vs0 = np.sqrt(L / R)

        ptemp = np.pi * np.log(vp0 / vp) / (np.log(vp0 / vp) + np.log(cmdl.bakusavlen/dz))
        Qp = 1.0 / np.tan(ptemp)
                          
        stemp = np.pi * np.log(vs0 / vs) / (np.log(vs0 / vs) + np.log(cmdl.bakusavlen/dz))
        Qs = 1.0 / np.tan(stemp)

        # Anisotropy calculations
        epsilon = (A - C) / (2.0 * C)
        delta = ((F + L)**2.0 - (C - L)**2.0) / (2.0 * C * (C - L))
        gamma = (M - L) / (2.0 * L)
        eta = (epsilon -delta) / (1 + 2 * delta) #anellipticity
        
        #qmin = 0, qmax = 200
        mmscale = MinMaxScaler((cmdl.scaleqminmax[0],cmdl.scaleqminmax[1]))
        
        #filter of arrays by -200 to +200 of Q values assumed to be reasonable
        #only for plotting
        zf = z[((Qp <= 200) & (Qp> -200)) & ((Qs  <=200) &(Qs > -200)) ]
        Qpf = Qp[((Qp <= 200) & (Qp> -200)) & ((Qs  <=200) &(Qs > -200)) ]
        # Qpf1 = Qpf.reshape(-1,1)
        Qpfscaled = mmscale.fit_transform(Qpf.reshape(-1,1)).reshape(-1,)
        Qsf = Qs[((Qp <= 200) & (Qp> -200)) & ((Qs  <=200) &(Qs > -200)) ]
        Qsfscaled = mmscale.fit_transform(Qsf.reshape(-1,1)).reshape(-1,)
        Qpfsmth = smooth(Qpf,int(cmdl.bakusavlen/dz), mode='same')
        Qpfsmthscaled = mmscale.fit_transform(Qpfsmth.reshape(-1,1)).reshape(-1,)
        Qsfsmth = smooth(Qsf,int(cmdl.bakusavlen/dz), mode='same')
        Qsfsmthscaled = mmscale.fit_transform(Qsfsmth.reshape(-1,1)).reshape(-1,)
        vp0f = vp0[((Qp <= 200) & (Qp> -200)) & ((Qs  <=200) &(Qs > -200)) ]
        vs0f = vs0[((Qp <= 200) & (Qp> -200)) & ((Qs  <=200) &(Qs > -200)) ]
        mu1f = mu1[((Qp <= 200) & (Qp> -200)) & ((Qs  <=200) &(Qs > -200)) ]
        lame1f = lame1[((Qp <= 200) & (Qp> -200)) & ((Qs  <=200) &(Qs > -200)) ]
        vbksf = vbks[((Qp <= 200) & (Qp> -200)) & ((Qs  <=200) &(Qs > -200)) ]
        vrmsf = vrms[((Qp <= 200) & (Qp> -200)) & ((Qs  <=200) &(Qs > -200)) ]
        vavgf = vavg[((Qp <= 200) & (Qp> -200)) & ((Qs  <=200) &(Qs > -200)) ]
        vpf = vp[((Qp <= 200) & (Qp> -200)) & ((Qs  <=200) &(Qs > -200)) ]
        vsf = vs[((Qp <= 200) & (Qp> -200)) & ((Qs  <=200) &(Qs > -200)) ]
        rhof = rho[((Qp <= 200) & (Qp> -200)) & ((Qs  <=200) &(Qs > -200)) ]
        youngf = young[((Qp <= 200) & (Qp> -200)) & ((Qs  <=200) &(Qs > -200)) ]
        bulkmodf =bulkmod[((Qp <= 200) & (Qp> -200)) & ((Qs  <=200) &(Qs > -200)) ]
        lambdarhof = lambdarho[((Qp <= 200) & (Qp> -200)) & ((Qs  <=200) &(Qs > -200)) ]
        murhof = murho[((Qp <= 200) & (Qp> -200)) & ((Qs  <=200) &(Qs > -200)) ]
        prf = pr[((Qp <= 200) & (Qp> -200)) & ((Qs  <=200) &(Qs > -200)) ]
        epsilonf =epsilon[((Qp <= 200) & (Qp> -200)) & ((Qs  <=200) &(Qs > -200)) ]
        deltaf = delta[((Qp <= 200) & (Qp> -200)) & ((Qs  <=200) &(Qs > -200)) ]
        gammaf = gamma[((Qp <= 200) & (Qp> -200)) & ((Qs  <=200) &(Qs > -200)) ]
        etaf = eta[((Qp <= 200) & (Qp> -200)) & ((Qs  <=200) &(Qs > -200)) ]
        vsvp2f = vsvp2[((Qp <= 200) & (Qp> -200)) & ((Qs  <=200) &(Qs > -200)) ]
        aif = ai[((Qp <= 200) & (Qp> -200)) & ((Qs  <=200) &(Qs > -200)) ]
        if len(w.columns) >4 :
            targetf = targetdata[((Qp <= 200) & (Qp> -200)) & ((Qs  <=200) &(Qs > -200))]
        if cmdl.outmultiplier != 1:
            list6(z,vp,vs,rho,vsvp2,mu,pr,young,lame,bulkmod,cmdl.outmultiplier)
            #list GPA and Mpsi for mu, young,lame and bulk
        else:
            dfcols = ['Z','VP','VS','RHOB','VSVP2','MU','PR','YOUNG','LAME',
                'BULKMOD','LAMBDARHO','MURHO','VP0','VS0','QP','QS','EPSILON',
                'DELTA','ETA','VAV','VRMS','VBAKUS','AI']
            if len(w.columns) >4 :
                dfcols.append(targethead)
            df2=pd.DataFrame({'Z':z,'VP':vp,'VS':vs,'RHOB':rho,'VSVP2': vsvp2,'MU':mu1,
                'PR':pr,'YOUNG':young,'LAME':lame1,'BULKMOD':bulkmod,'LAMBDARHO':lambdarho,
                'MURHO':murho,'VP0':vp0,'VS0':vs0,'QP':Qp,'QS':Qs,'EPSILON':epsilon,'DELTA':delta,
                'ETA':eta,'VAV':vavg,'VRMS':vrms,'VBAKUS':vbks,'AI':ai})
            if len(w.columns) >4 :
                df2[targethead] = targetdata
            df2 = df2[dfcols].copy()

            # output Qp and Qs smoothed and filtered, so  # of samples are different from the rest
            dfsmthcols = ['Z','VP','VS','RHOB','VSVP2','MU','PR','YOUNG','LAME',
                'BULKMOD','LAMBDARHO','MURHO','VP0','VS0','QPsmth','QSsmth','QPsmthscaled','QSsmthscaled','EPSILON',
                'DELTA','ETA','VAV','VRMS','VBAKUS','AI']
            if len(w.columns) >4 :
                dfsmthcols.append(targethead)
            # dfsmthcols = ['Z','QPsmth','QSsmth']
            df4 = pd.DataFrame({'Z':zf,'QPsmth':Qpfsmth,'QSsmth':Qsfsmth,'QPsmthscaled':Qpfsmthscaled,
                'QSsmthscaled':Qsfsmthscaled,'VP':vpf,'VS':vsf,'RHOB':rhof,
                'VSVP2': vsvp2f,'MU':mu1f,'PR':prf,'YOUNG':youngf,'LAME':lame1f,'BULKMOD':bulkmodf,
                'LAMBDARHO':lambdarhof,'MURHO':murhof,'VP0':vp0f,'VS0':vs0f,
                'EPSILON':epsilonf,'DELTA':deltaf,'ETA':etaf,'VAV':vavgf,'VRMS':vrmsf,'VBAKUS':vbksf,'AI':aif})
            if len(w.columns) >4 :
                df4[targethead] = targetdata
            df4 = df4[dfsmthcols].copy()
            
            dirsplit,fextsplit= os.path.split(cmdl.csvfilename)
            fname,fextn= os.path.splitext(fextsplit)
            csvoutfname = os.path.join(dirsplit,fname) +"_%d.csv" % cmdl.bakusavlen

            csvoutsmthfname = os.path.join(dirsplit,fname) +"_%dsmth.csv" % cmdl.bakusavlen

            df2.to_csv(csvoutfname,index=False,float_format ='%12.5f')
            df4.to_csv(csvoutsmthfname,index=False,float_format ='%12.5f')
            pdffname = os.path.join(dirsplit,fname)+'_%d.pdf' % cmdl.bakusavlen

        if cmdl.leastsqrfit:
            pcoeflist = []
            xyvarlist =[]
        if cmdl.plotrange:
            zmin,zmax= cmdl.plotrange[0],cmdl.plotrange[1]
            
            #original unfiltered data
            zx = z[(z >= zmin) & (z < zmax) ]
            vpx= vp[(z >= zmin) & (z< zmax)]            
            vsx= vs[(z >= zmin) & (z< zmax)]            
            rhox= rho[(z >= zmin) & (z< zmax)]            
            mu1x= mu1[(z >= zmin) & (z< zmax)]            
            prx= pr[(z >= zmin) & (z< zmax)]            
            vsvp2x= vsvp2[(z >= zmin) & (z< zmax)]            
            youngx= young[(z >= zmin) & (z< zmax)]            
            lamex= lame1[(z >= zmin) & (z< zmax)]            
            bulkmodx= bulkmod[(z >= zmin) & (z< zmax)]            
            lambdarhox= lambdarho[(z >= zmin) & (z< zmax)]            
            murhox= murho[(z >= zmin) & (z< zmax)]            
            vp0x= vp0[(z >= zmin) & (z < zmax)]
            vs0x= vs0[(z >= zmin) & (z < zmax)]
            Qpx= Qp[(z >= zmin) & (z < zmax)]
            Qsx= Qs[(z >= zmin) & (z < zmax)]
            epsilonx= epsilon[(z >= zmin) & (z < zmax)]
            deltax= delta[(z >= zmin) & (z < zmax)]
            gammax= gamma[(z >= zmin) & (z < zmax)]
            etax= eta[(z >= zmin) & (z < zmax)]
            vavgx= vavg[(z >= zmin) & (z < zmax)]
            vrmsx= vrms[(z >= zmin) & (z < zmax)]
            vbksx = vbks[(z >= zmin) & (z < zmax)]
            aix = ai[(z >= zmin) & (z < zmax)]
            zxf = zf[(zf >= zmin) & (zf < zmax) ]
            Qpxf= Qpf[(zf >= zmin) & (zf < zmax)]
            Qpxfscaled = mmscale.fit_transform(Qpxf.reshape(-1,1)).reshape(-1,)
            Qsxf= Qsf[(zf >= zmin) & (zf < zmax)]
            Qsxfscaled = mmscale.fit_transform(Qsxf.reshape(-1,1)).reshape(-1,)
            Qpxfsmth= Qpfsmth[(zf >= zmin) & (zf < zmax)]
            Qpxfsmthscaled = mmscale.fit_transform(Qpxfsmth.reshape(-1,1)).reshape(-1,)
            Qsxfsmth= Qsfsmth[(zf >= zmin) & (zf < zmax)]
            Qsxfsmthscaled = mmscale.fit_transform(Qsxfsmth.reshape(-1,1)).reshape(-1,)
            if len(w.columns) >4 :
                targetx = targetdata[((z >= zmin) & (z < zmax))]

            #filtered smoothed data
            # zf = z[((Qp <= 200) & (Qp> -200)) & ((Qs  <=200) &(Qs > -200)) ]
            zxf = zf[(zf >= zmin) & (zf < zmax) ]
            vpxf= vpf[(zf >= zmin) & (zf < zmax) ]            
            vsxf= vsf[(zf >= zmin) & (zf < zmax) ]            
            rhoxf= rhof[(zf >= zmin) & (zf < zmax) ]            
            mu1xf= mu1f[(zf >= zmin) & (zf < zmax) ]            
            prxf= prf[(zf >= zmin) & (zf < zmax) ]            
            vsvp2xf= vsvp2f[(zf >= zmin) & (zf < zmax) ]            
            youngxf= youngf[(zf >= zmin) & (zf < zmax) ]            
            lame1xf= lame1f[(zf >= zmin) & (zf < zmax) ]            
            bulkmodxf= bulkmodf[(zf >= zmin) & (zf < zmax) ]            
            lambdarhoxf= lambdarhof[(zf >= zmin) & (zf < zmax) ]            
            murhoxf= murhof[(zf >= zmin) & (zf < zmax) ]           
            vp0xf= vp0f[(zf >= zmin) & (zf < zmax) ]
            vs0xf= vs0f[(zf >= zmin) & (zf < zmax) ]
            Qpxf= Qpf[(zf >= zmin) & (zf < zmax) ]
            Qsxf= Qsf[(zf >= zmin) & (zf < zmax) ]
            epsilonxf= epsilonf[(zf >= zmin) & (zf < zmax) ]
            deltaxf= deltaf[(zf >= zmin) & (zf < zmax) ]
            gammaxf= gammaf[(zf >= zmin) & (zf < zmax) ]
            etaxf= etaf[(zf >= zmin) & (zf < zmax) ]
            vavgxf= vavgf[(zf >= zmin) & (zf < zmax) ]
            vrmsxf= vrmsf[(zf >= zmin) & (zf < zmax) ]
            vbksxf = vbksf[(zf >= zmin) & (zf < zmax)]
            aixf = aif[(zf >= zmin) & (zf < zmax)]
            # print('vbksxf size:',vbksxf.size)
            Qpxf= Qpf[(zf >= zmin) & (zf < zmax)]
            Qpxfscaled = mmscale.fit_transform(Qpxf.reshape(-1,1)).reshape(-1,)
            Qsxf= Qsf[(zf >= zmin) & (zf < zmax)]
            Qsxfscaled = mmscale.fit_transform(Qsxf.reshape(-1,1)).reshape(-1,)
            Qpxfsmth= Qpfsmth[(zf >= zmin) & (zf < zmax)]
            Qpxfsmthscaled = mmscale.fit_transform(Qpxfsmth.reshape(-1,1)).reshape(-1,)
            Qsxfsmth= Qsfsmth[(zf >= zmin) & (zf < zmax)]
            Qsxfsmthscaled = mmscale.fit_transform(Qsxfsmth.reshape(-1,1)).reshape(-1,)
            if len(w.columns) >4 :
                targetxf = targetdata[((z >= zmin) & (z < zmax))]
            
            
            rangefname = os.path.join(dirsplit,fname) +"_%d_%-.0f_%-.0f.csv" %(cmdl.bakusavlen,zmin,zmax)
            rangefnamepdf = os.path.join(dirsplit,fname) +"_%d_%-.0f_%-.0f.pdf" %(cmdl.bakusavlen,zmin,zmax)
            dfcols = ['Z','VP','VS','RHOB','VSVP2','MU','PR','YOUNG','LAME',
                'BULKMOD','LAMBDARHO','MURHO','VP0','VS0','QP','QS','EPSILON',
                'DELTA','ETA','VAV','VRMS','VBAKUS','AI']
            if len(w.columns) >4 :
                dfcols.append(targethead)
            df3=pd.DataFrame({'Z':zx,'VP':vpx,'VS':vsx,'RHOB':rhox,'VSVP2': vsvp2x,'MU':mu1x,
                'PR':prx,'YOUNG':youngx,'LAME':lamex,'BULKMOD':bulkmodx,'LAMBDARHO':lambdarhox,
                'MURHO':murhox,'VP0':vp0x,'VS0':vs0x,'QP':Qpx,'QS':Qsx,'EPSILON':epsilonx,'DELTA':deltax,
                'ETA':etax,'VAV':vavgx,'VRMS':vrmsx,'VBAKUS':vbksx,'AI':aix})
            if len(w.columns) >4 :
                df3[targethead] = targetdata
            df3 = df3[dfcols].copy()
            df3.to_csv(r'%s'%rangefname,index=False,float_format='%10.5f')

            
            # output Qp and Qs smoothed and filtered, so  # of samples are different from the rest
            # dfsmthcols = ['Z','QPsmth','QSsmth']
            dfsmthcols = ['Z','VP','VS','RHOB','VSVP2','MU','PR','YOUNG','LAME',
                'BULKMOD','LAMBDARHO','MURHO','VP0','VS0','QPsmth','QSsmth','QPsmthscaled','QSsmthscaled','EPSILON',
                'DELTA','ETA','VAV','VRMS','VBAKUS','AI']
            if len(w.columns) >4 :
                dfsmthcols.append(targethead)
            df4 = pd.DataFrame({'Z':zxf,'QPsmth':Qpxfsmth,'QSsmth':Qsxfsmth,'QPsmthscaled':Qpxfsmthscaled,
                'QSsmthscaled':Qsxfsmthscaled,'VP':vpxf,'VS':vsxf,'RHOB':rhoxf,
                'VSVP2': vsvp2xf,'MU':mu1xf,'PR':prxf,'YOUNG':youngxf,'LAME':lame1xf,'BULKMOD':bulkmodxf,
                'LAMBDARHO':lambdarhoxf,'MURHO':murhoxf,'VP0':vp0xf,'VS0':vs0xf,
                'EPSILON':epsilonxf,'DELTA':deltaxf,'ETA':etaxf,'VAV':vavgxf,'VRMS':vrmsxf,
                'VBAKUS':vbksxf,'AI':aixf})
            # df4 = pd.DataFrame({'Z':zf,'QPsmth':Qpfsmth,'QSsmth':Qsfsmth})
            if len(w.columns) >4 :
                df4[targethead] = targetdata
            df4 = df4[dfsmthcols].copy()
            csvoutsmthrange = os.path.join(dirsplit,fname) +"_%d%-.0f_%-.0fsmth.csv" %(cmdl.bakusavlen,zmin,zmax)
            df4.to_csv(csvoutsmthrange,index=False,float_format ='%12.5f')
            
            with PdfPages(rangefnamepdf) as pdf:
                plt.figure(figsize=(8,8))

                plt.subplot(2,2,1)
                plt.xlabel('VPx')
                plt.ylabel('Zx')
                plt.gca().invert_yaxis()
                plt.plot(vpx,zx)

                plt.subplot(2,2,2)
                plt.xlabel('VSx')
                plt.ylabel('Zx')
                plt.gca().invert_yaxis()
                plt.plot(vsx,zx)

                plt.subplot(2,2,3)
                plt.xlabel('YOUNGx')
                plt.ylabel('Zx')
                plt.gca().invert_yaxis()
                plt.plot(youngx,zx)

                plt.subplot(2,2,4)
                plt.xlabel('BULKMODx')
                plt.ylabel('Zx')
                plt.gca().invert_yaxis()
                plt.plot(bulkmodx,zx)

                pdf.savefig()
                if not cmdl.hideplots:
                    plt.show()
                plt.close()

                clmp=cm.get_cmap('rainbow_r')

                plt.figure(figsize=(8,8))
                plt.subplot(2,2,1)
                plt.xlabel('VPx')
                plt.ylabel('VSx')
                scplt=plt.scatter(vpx,vsx,c=zx,cmap=clmp,s=10)
                plt.colorbar(scplt)
                if cmdl.leastsqrfit :
                    pfitcoef=np.polyfit(vpx,vsx,cmdl.fitdegree)
                    xi=np.linspace(vpx.min(),vpx.max())
                    yi=np.polyval(pfitcoef,xi)
                    plt.plot(xi,yi,c='r',lw=3)
                    pcoeflist.append([pfitcoef[i] for i in range(pfitcoef.size)])
                    xyvarlist.append(('VPx','VSx'))

                plt.subplot(2,2,2)
                plt.xlabel('VSx')
                plt.ylabel('PRx')
                scplt=plt.scatter(vsx,prx,c=zx,cmap=clmp,s=10)
                plt.colorbar(scplt)
                if cmdl.leastsqrfit :
                    pfitcoef=np.polyfit(vsx,prx,cmdl.fitdegree)
                    xi=np.linspace(vsx.min(),vsx.max())
                    yi=np.polyval(pfitcoef,xi)
                    plt.plot(xi,yi,c='r',lw=3)
                    pcoeflist.append([pfitcoef[i] for i in range(pfitcoef.size)])
                    xyvarlist.append(('VSx','PRx'))


                plt.subplot(2,2,3)
                plt.xlabel('VPx')
                plt.ylabel('YOUNGx')
                scplt=plt.scatter(vpx,youngx,c=zx,cmap=clmp,s=10)
                plt.colorbar(scplt)

                if cmdl.leastsqrfit :
                    pfitcoef=np.polyfit(vpx,youngx,cmdl.fitdegree)
                    xi=np.linspace(vpx.min(),vpx.max())
                    yi=np.polyval(pfitcoef,xi)
                    plt.plot(xi,yi,c='r',lw=3)
                    pcoeflist.append([pfitcoef[i] for i in range(pfitcoef.size)])
                    xyvarlist.append(('VPx','YOUNGx'))



                plt.subplot(2,2,4)
                plt.xlabel('RHOBx')
                plt.ylabel('BULKMODx')
                scplt=plt.scatter(rhox,bulkmodx,c=zx,cmap=clmp,s=10)
                plt.colorbar(scplt)
                if cmdl.leastsqrfit :
                    pfitcoef=np.polyfit(rhox,bulkmodx,cmdl.fitdegree)
                    xi=np.linspace(rhox.min(),rhox.max())
                    yi=np.polyval(pfitcoef,xi)
                    plt.plot(xi,yi,c='r',lw=3)
                    pcoeflist.append([pfitcoef[i] for i in range(pfitcoef.size)])
                    xyvarlist.append(('RHOBx','BULMODx'))


                pdf.savefig()
                if not cmdl.hideplots:
                    plt.show()
                plt.close()


                clmp=cm.get_cmap('rainbow_r')

                plt.figure(figsize=(8,8))
                plt.subplot(2,2,1)
                plt.xlabel('VSVP2x')
                plt.ylabel('PRx')
                scplt=plt.scatter(vsvp2x,prx,c=zx,cmap=clmp,s=10)
                plt.colorbar(scplt)
                if cmdl.leastsqrfit :
                    pfitcoef=np.polyfit(vsvp2x,prx,cmdl.fitdegree)
                    xi=np.linspace(vsvp2x.min(),vsvp2x.max())
                    yi=np.polyval(pfitcoef,xi)
                    plt.plot(xi,yi,c='r',lw=3)
                    pcoeflist.append([pfitcoef[i] for i in range(pfitcoef.size)])
                    xyvarlist.append(('VSVP2x','PRx'))


                plt.subplot(2,2,2)
                plt.xlabel('BULKMODx')
                plt.ylabel('MUx')
                scplt=plt.scatter(bulkmodx,mu1x,c=zx,cmap=clmp,s=10)
                plt.colorbar(scplt)
                if cmdl.leastsqrfit :
                    pfitcoef=np.polyfit(bulkmodx,mu1x,cmdl.fitdegree)
                    xi=np.linspace(bulkmodx.min(),bulkmodx.max())
                    yi=np.polyval(pfitcoef,xi)
                    plt.plot(xi,yi,c='r',lw=3)
                    pcoeflist.append([pfitcoef[i] for i in range(pfitcoef.size)])
                    xyvarlist.append(('BULKMODx','MUx'))


                plt.subplot(2,2,3)
                plt.xlabel('RHOBx')
                plt.ylabel('YOUNGx')
                scplt=plt.scatter(rhox,youngx,c=zx,cmap=clmp,s=10)
                plt.colorbar(scplt)
                if cmdl.leastsqrfit :
                    pfitcoef=np.polyfit(rhox,youngx,cmdl.fitdegree)
                    xi=np.linspace(rhox.min(),rhox.max())
                    yi=np.polyval(pfitcoef,xi)
                    plt.plot(xi,yi,c='r',lw=3)
                    pcoeflist.append([pfitcoef[i] for i in range(pfitcoef.size)])
                    xyvarlist.append(('RHOBx','YOUNGx'))


                plt.subplot(2,2,4)
                plt.xlabel('VSx')
                plt.ylabel('YOUNGx')
                scplt=plt.scatter(vsx,youngx,c=zx,cmap=clmp,s=10)
                plt.colorbar(scplt)
                if cmdl.leastsqrfit :
                    pfitcoef=np.polyfit(vsx,youngx,cmdl.fitdegree)
                    xi=np.linspace(vsx.min(),vsx.max())
                    yi=np.polyval(pfitcoef,xi)
                    plt.plot(xi,yi,c='r',lw=3)
                    pcoeflist.append([pfitcoef[i] for i in range(pfitcoef.size)])
                    xyvarlist.append(('VSx','YOUNGx'))



                pdf.savefig()
                if not cmdl.hideplots:
                    plt.show()
                plt.close()

                plt.figure(figsize=(8,8))
                plt.subplot(1,1,1)
                plt.xlabel('LAMBDARHOx')
                plt.ylabel('MURHOx')
                scplt=plt.scatter(lambdarhox,murhox,c=zx,cmap=clmp,s=10)
                plt.colorbar(scplt)
                if cmdl.leastsqrfit :
                    pfitcoef=np.polyfit(lambdarhox,murhox,cmdl.fitdegree)
                    xi=np.linspace(lambdarhox.min(),lambdarhox.max())
                    yi=np.polyval(pfitcoef,xi)
                    plt.plot(xi,yi,c='r',lw=3)
                    pcoeflist.append([pfitcoef[i] for i in range(pfitcoef.size)])
                    xyvarlist.append(('LAMBDARHOx','MURHOx'))



                pdf.savefig()
                if not cmdl.hideplots:
                    plt.show()

                plt.figure(figsize=[8,5])
                plt.plot(zx, vpx, alpha=0.5,label='vpx')
                plt.plot(zx, vrmsx,label='vrmsx')
                plt.plot(zx, vavgx,label='vavgx')
                plt.plot(zx, vp0x, 'k', lw=2,label='vbakus')
                plt.plot(zx, aix, alpha=0.5,label='aix')
                plt.xlabel('Z')
                plt.ylabel('VELOCITY/ AI')
                plt.legend()
                pdf.savefig()
                if not cmdl.hideplots:
                    plt.show()
                plt.close()


 
 

                fig,ax = plt.subplots(2,1,figsize=[12,8])
                # plt.figure(figsize=[12,5])
                ax[0].plot(zxf, Qpxf, label='Qp')
                ax[0].plot(zxf, Qsxf, label='Qs')
                ax[0].legend()
                ax[0].set_xlabel('Z')
                ax[0].set_ylabel('Q')
                
                ax[1].plot(zxf, Qpxfsmth, label='Qp smooth')
                ax[1].plot(zxf, Qsxfsmth, label='Qs smooth')
                ax[1].legend()
                ax[1].set_xlabel('Z')
                ax[1].set_ylabel('Q')
                pdf.savefig()
                if not cmdl.hideplots:
                    plt.show()
                plt.close()

                fig,ax = plt.subplots(2,1,figsize=[12,8])
                # plt.figure(figsize=[12,5])
                ax[0].plot(zxf, Qpxfscaled, label='Qpscaled')
                ax[0].plot(zxf, Qsxfscaled, label='Qsscaled')
                ax[0].legend()
                ax[0].set_xlabel('Z')
                ax[0].set_ylabel('Q')
                
                ax[1].plot(zxf, Qpxfsmthscaled, label='Qp smooth scaled')
                ax[1].plot(zxf, Qsxfsmthscaled, label='Qs smooth scaled')
                ax[1].legend()
                ax[1].set_xlabel('Z')
                ax[1].set_ylabel('Q')
                pdf.savefig()
                if not cmdl.hideplots:
                    plt.show()
                plt.close()



 
                
                fig,ax = plt.subplots(2,1,figsize=[8,8])

                ax[0].plot(zx, vpx, 'k', lw=0.5)
                ax[0].plot(zx, vp0x, 'r', lw=2)
                ax[0].grid()
                ax[0].set_ylabel('Velocity ')
                ax[0].set_xlabel('Depth ')

                ax[1].plot(zx, 1/Qpx, lw=0.5)
                ax[1].fill_between(zx, 1/Qpx, alpha=0.3)
                #ax[1].grid()
                ax[1].set_ylabel('1/Qp')
                ax[1].set_xlabel('Depth ')
                pdf.savefig()
                if not cmdl.hideplots:
                    plt.show()
                plt.close()

                plt.figure(figsize=[8,8])

                plt.subplot(121)
                plt.plot(vpx, zx, 'k', lw=0.7,label = 'VP')
                plt.plot(vp0x, zx, 'r', lw=2,label = 'VP0')
                plt.plot(vsx, zx, 'k', lw=0.3,label = 'VS')
                plt.plot(vs0x, zx, 'g', lw=2,label = 'VS0')
                plt.gca().invert_yaxis()
                plt.grid()
                plt.xlabel('Velocity ')
                plt.ylabel('Depth ')
                plt.legend(loc='best')                

                plt.subplot(122)
                plt.plot(epsilonx, zx, 'r', label='$\epsilon$')
                plt.plot(deltax, zx, 'b', label='$\delta$')
                plt.plot(gammax, zx, 'g', label='$\gamma$')
                plt.plot(etax, zx, 'm', label='$\eta$')
                plt.gca().invert_yaxis()
                plt.grid()
                plt.xlabel('$\epsilon$ / $\delta$ / $\gamma$/ $\eta$')
                plt.ylabel('Depth (m)')
                plt.legend(loc='best')                
                pdf.savefig()
                if not cmdl.hideplots:
                    plt.show()
                plt.close()
                


                # plt.close()
                if cmdl.leastsqrfit:
                    pcarray = np.array(pcoeflist)
                    coefdf0 = pd.DataFrame(pcarray)

                    coefdf1 = pd.DataFrame(xyvarlist,columns=['XVAR','YVAR'])
                    coefdf0[['XVAR','YVAR']] = coefdf1[['XVAR','YVAR']]
                    coefrangefname = os.path.join(dirsplit,fname) +"_%d%-.0f_%-.0fcoef.csv" %(cmdl.bakusavlen,zmin,zmax)
                    coefdf0.to_csv(coefrangefname,index=False)


        #plot full range
        else:
            with PdfPages(pdffname) as pdf:
                plt.figure(figsize=(8,8))
                plt.subplot(2,2,1)
                plt.xlabel('VP')
                plt.ylabel('Z')
                plt.gca().invert_yaxis()
                plt.plot(vp,z)

                plt.subplot(2,2,2)
                plt.xlabel('VS')
                plt.ylabel('Z')
                plt.gca().invert_yaxis()
                plt.plot(vs,z)

                plt.subplot(2,2,3)
                plt.xlabel('Young Mod E')
                plt.ylabel('Z')
                plt.gca().invert_yaxis()
                plt.plot(young,z)

                plt.subplot(2,2,4)
                plt.xlabel('Bulk Mod K')
                plt.ylabel('Z')
                plt.gca().invert_yaxis()
                plt.plot(bulkmod,z)

                pdf.savefig()
                if not cmdl.hideplots:
                    plt.show()
                plt.close()

                clmp=cm.get_cmap('rainbow_r')

                plt.figure(figsize=(8,8))
                # plt.figure(2)
                plt.subplot(2,2,1)
                plt.xlabel('VP')
                plt.ylabel('VS')
                scplt=plt.scatter(vp,vs,c=z,cmap=clmp,s=10)
                plt.colorbar(scplt)
                if cmdl.leastsqrfit :
                    pfitcoef=np.polyfit(vp,vs,cmdl.fitdegree)
                    xi=np.linspace(vp.min(),vp.max())
                    yi=np.polyval(pfitcoef,xi)
                    plt.plot(xi,yi,c='r',lw=3)
                    pcoeflist.append([pfitcoef[i] for i in range(pfitcoef.size)])
                    #pcoeflist.append([pfitcoef])
                    xyvarlist.append(('VP','VS'))


                plt.subplot(2,2,2)
                plt.xlabel('VS')
                plt.ylabel('PR')
                scplt=plt.scatter(vs,pr,c=z,cmap=clmp,s=10)
                plt.colorbar(scplt)
                if cmdl.leastsqrfit :
                    pfitcoef=np.polyfit(vs,pr,cmdl.fitdegree)
                    xi=np.linspace(vs.min(),vs.max())
                    yi=np.polyval(pfitcoef,xi)
                    plt.plot(xi,yi,c='r',lw=3)
                    pcoeflist.append([pfitcoef[i] for i in range(pfitcoef.size)])
                    #pcoeflist.append([pfitcoef])
                    xyvarlist.append(('VS','PR'))



                plt.subplot(2,2,3)
                plt.xlabel('VP')
                plt.ylabel('Young E')
                scplt=plt.scatter(vp,young,c=z,cmap=clmp,s=10)
                plt.colorbar(scplt)
                if cmdl.leastsqrfit :
                    pfitcoef=np.polyfit(vp,young,cmdl.fitdegree)
                    xi=np.linspace(vp.min(),vp.max())
                    yi=np.polyval(pfitcoef,xi)
                    plt.plot(xi,yi,c='r',lw=3)
                    pcoeflist.append([pfitcoef[i] for i in range(pfitcoef.size)])
                    #pcoeflist.append([pfitcoef])
                    xyvarlist.append(('VP','YOUNG'))

                plt.subplot(2,2,4)
                plt.xlabel('RHO')
                plt.ylabel('BULKMOD')
                scplt=plt.scatter(rho,bulkmod,c=z,cmap=clmp,s=10)
                plt.colorbar(scplt)
                if cmdl.leastsqrfit :
                    pfitcoef=np.polyfit(rho,bulkmod,cmdl.fitdegree)
                    xi=np.linspace(rho.min(),rho.max())
                    yi=np.polyval(pfitcoef,xi)
                    plt.plot(xi,yi,c='r',lw=3)
                    pcoeflist.append([pfitcoef[i] for i in range(pfitcoef.size)])
                    #pcoeflist.append([pfitcoef])
                    xyvarlist.append(('RHO','BULKMOD'))

                pdf.savefig()
                if not cmdl.hideplots:
                    plt.show()
                plt.close()

                clmp=cm.get_cmap('rainbow_r')

                plt.figure(figsize=(8,8))
                # plt.figure(2)
                plt.subplot(2,2,1)
                plt.xlabel('VSVPSQR')
                plt.ylabel('PR')
                scplt=plt.scatter(vsvp2,pr,c=z,cmap=clmp,s=10)
                plt.colorbar(scplt)
                if cmdl.leastsqrfit :
                    pfitcoef=np.polyfit(vsvp2,pr,cmdl.fitdegree)
                    xi=np.linspace(vsvp2.min(),vsvp2.max())
                    yi=np.polyval(pfitcoef,xi)
                    plt.plot(xi,yi,c='r',lw=3)
                    pcoeflist.append([pfitcoef[i] for i in range(pfitcoef.size)])
                    #pcoeflist.append([pfitcoef])
                    xyvarlist.append(('VSVPSQR','PR'))


                # plt.figure(2)
                plt.subplot(2,2,2)
                plt.xlabel('BULKMOD')
                plt.ylabel('MU')
                scplt=plt.scatter(bulkmod,mu,c=z,cmap=clmp,s=10)
                plt.colorbar(scplt)
                if cmdl.leastsqrfit :
                    pfitcoef=np.polyfit(bulkmod,mu,cmdl.fitdegree)
                    xi=np.linspace(bulkmod.min(),bulkmod.max())
                    yi=np.polyval(pfitcoef,xi)
                    plt.plot(xi,yi,c='r',lw=3)
                    pcoeflist.append([pfitcoef[i] for i in range(pfitcoef.size)])
                    #pcoeflist.append([pfitcoef])
                    xyvarlist.append(('BULKMOD','MU'))


                # plt.figure(2)
                plt.subplot(2,2,3)
                plt.xlabel('RHOB')
                plt.ylabel('YOUNG')
                scplt=plt.scatter(rho,young,c=z,cmap=clmp,s=10)
                plt.colorbar(scplt)
                if cmdl.leastsqrfit :
                    pfitcoef=np.polyfit(rho,young,cmdl.fitdegree)
                    xi=np.linspace(rho.min(),rho.max())
                    yi=np.polyval(pfitcoef,xi)
                    plt.plot(xi,yi,c='r',lw=3)
                    pcoeflist.append([pfitcoef[i] for i in range(pfitcoef.size)])
                    #pcoeflist.append([pfitcoef])
                    xyvarlist.append(('RHOB','YOUNG'))


                # plt.figure(2)
                plt.subplot(2,2,4)
                plt.xlabel('VS')
                plt.ylabel('YOUNG')
                scplt=plt.scatter(vs,young,c=z,cmap=clmp,s=10)
                plt.colorbar(scplt)
                if cmdl.leastsqrfit :
                    pfitcoef=np.polyfit(vs,young,cmdl.fitdegree)
                    xi=np.linspace(vs.min(),vs.max())
                    yi=np.polyval(pfitcoef,xi)
                    plt.plot(xi,yi,c='r',lw=3)
                    pcoeflist.append([pfitcoef[i] for i in range(pfitcoef.size)])
                    #pcoeflist.append([pfitcoef])
                    xyvarlist.append(('VS','YOUNG'))

                pdf.savefig()
                if not cmdl.hideplots:
                    plt.show()
                plt.close()
                #plot lambda rho vs mu rho
                # plt.figure(3)
                plt.figure(figsize=(8,8))
                plt.subplot(1,1,1)
                plt.xlabel('LAMBDA_RHO')
                plt.ylabel('MU_RHO')
                scplt=plt.scatter(lambdarho,murho,c=z,cmap=clmp,s=10)
                plt.colorbar(scplt)
                if cmdl.leastsqrfit :
                    pfitcoef=np.polyfit(lambdarho,murho,cmdl.fitdegree)
                    xi=np.linspace(lambdarho.min(),lambdarho.max())
                    yi=np.polyval(pfitcoef,xi)
                    plt.plot(xi,yi,c='r',lw=3)
                    pcoeflist.append([pfitcoef[i] for i in range(pfitcoef.size)])
                    xyvarlist.append(('LAMBDA_RHO','MU_RHO'))



                pdf.savefig()
                if not cmdl.hideplots:
                    plt.show()
                plt.close()
                
                
                plt.figure(figsize=[12,5])
                plt.plot(z, vp, alpha=0.5,label='vp')
                plt.plot(z, vrms,label='vrms')
                plt.plot(z, vavg,label='vavg')
                plt.plot(z, vp0, 'k', lw=2,label='vbakus')
                plt.plot(z, ai, alpha=0.5,label='ai')
                plt.xlabel('Z')
                plt.ylabel('VELOCITY / AI')
                plt.legend()
                pdf.savefig()
                if not cmdl.hideplots:
                    plt.show()
                plt.close()

                
                fig,ax = plt.subplots(2,1,figsize=[12,8])
                # plt.figure(figsize=[12,5])
                ax[0].plot(zf, Qpf, label='Qp')
                ax[0].plot(zf, Qsf, label='Qs')
                ax[0].legend()
                ax[0].set_xlabel('Z')
                ax[0].set_ylabel('Q')
                
                ax[1].plot(zf, Qpfsmth, label='Qp smooth')
                ax[1].plot(zf, Qsfsmth, label='Qs smooth')
                ax[1].legend()
                ax[1].set_xlabel('Z')
                ax[1].set_ylabel('Q')
                pdf.savefig()
                if not cmdl.hideplots:
                    plt.show()
                plt.close()
                
                
                fig,ax = plt.subplots(2,1,figsize=[12,8])
                # plt.figure(figsize=[12,5])
                ax[0].plot(zf, Qpfscaled, label='Qp scaled')
                ax[0].plot(zf, Qsfscaled, label='Qs scaled')
                ax[0].legend()
                ax[0].set_xlabel('Z')
                ax[0].set_ylabel('Q')
                
                ax[1].plot(zf, Qpfsmthscaled, label='Qp smooth scaled')
                ax[1].plot(zf, Qsfsmthscaled, label='Qs smooth scaled')
                ax[1].legend()
                ax[1].set_xlabel('Z')
                ax[1].set_ylabel('Q')
                pdf.savefig()
                if not cmdl.hideplots:
                    plt.show()
                plt.close()
                

                
                
                fig,ax = plt.subplots(2,1,figsize=[12,8])

                ax[0].plot(z, vp, 'k', lw=0.5)
                ax[0].plot(z, vp0, 'r', lw=2)
                ax[0].grid()
                ax[0].set_ylabel('Velocity ')
                ax[0].set_xlabel('Depth ')

                ax[1].plot(z, 1/Qp, lw=0.5)
                ax[1].fill_between(z, 1/Qp, alpha=0.3)
                #ax[1].grid()
                ax[1].set_ylabel('1/Qp')
                ax[1].set_xlabel('Depth ')
                pdf.savefig()
                if not cmdl.hideplots:
                    plt.show()
                plt.close()

                plt.figure(figsize=[8,8])

                plt.subplot(121)
                plt.plot(vp, z, 'k', lw=0.7,label = 'VP')
                plt.plot(vp0, z, 'r', lw=2,label = 'VP0')
                plt.plot(vs, z, 'k', lw=0.3,label = 'VS')
                plt.plot(vs0, z, 'g', lw=2,label = 'VS0')
                plt.gca().invert_yaxis()
                plt.grid()
                plt.xlabel('Velocity ')
                plt.ylabel('Depth ')
                plt.legend(loc='best')                

                plt.subplot(122)
                plt.plot(epsilon, z, 'r', label='$\epsilon$')
                plt.plot(delta, z, 'b', label='$\delta$')
                plt.plot(gamma, z, 'g', label='$\gamma$')
                plt.plot(eta, z, 'm', label='$\eta$')
                plt.gca().invert_yaxis()
                plt.grid()
                plt.xlabel('$\epsilon$ / $\delta$ / $\gamma$/ $\eta$')
                plt.ylabel('Depth (m)')
                plt.legend(loc='best')                
                pdf.savefig()
                if not cmdl.hideplots:
                    plt.show()
                plt.close()

                if cmdl.leastsqrfit:
                    #print "pcoeflist:%d " %len(pcoeflist),pcoeflist
                    pcarray = np.array(pcoeflist)
                    coefdf0 = pd.DataFrame(pcarray)

                    coefdf1 = pd.DataFrame(xyvarlist,columns=['XVAR','YVAR'])
                    coefdf0[['XVAR','YVAR']] = coefdf1[['XVAR','YVAR']]
                    coeffname = os.path.join(dirsplit,fname) +"_%d_coef.csv" % cmdl.bakusavlen
                    coefdf0.to_csv(coeffname,index=False)

if __name__== '__main__':
    main()