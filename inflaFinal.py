# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 10:50:13 2016

@author: e612727
"""

import pandas as pd
import numpy
from datetime import datetime, timedelta
import datetime as dt
import xlwt

us=pd.read_excel('US data.xlsx',na_values=['NA'])
rates_return=pd.read_csv('rates_futures_returns.csv',na_values=['NA'])
pmi=pd.read_csv('PX_LAST.csv',na_values=['NA'])
us2=pd.read_excel('US data2.xlsx')

rates=rates_return.iloc[:,1:]
start_dt = dt.datetime(1999, 1, 1)
end_dt =  dt.datetime(2016, 4, 29)
freq = 'B' # business-days
dt_range = pd.date_range(start=start_dt, end=end_dt, freq=freq, normalize=True)
    
rates_return.index=dt_range
ust=pd.DataFrame(rates_return.iloc[:,15:20]) #['USD_10Y'])
ust.index=dt_range

cpi=pd.read_excel('us cpi.xlsx', na_values=['NA'])

rmparams = {}
rmparams['vol_halflife'] = 21
rmparams['corr_halflife'] = 126
rmparams['corr_cap'] = 1.0
rmparams['corr_type'] = 'shrinktoaverage'
rmparams['lag'] = 0


riskmodel_dict = calc_ewma_riskmodel(ust,
                          vol_halflife=rmparams['vol_halflife'],
                          vol_seed_period=rmparams['vol_halflife'],
                           corr_halflife=rmparams['corr_halflife'],
                            corr_seed_period=rmparams['corr_halflife'],
                            corr_type=rmparams['corr_type'],
                            corr_cap=rmparams['corr_cap'],
                            lag=rmparams['lag'])
fields=['USD_2Y','USD_5Y','USD_10Y','USD_20Y']
cpichng=cpi.iloc[1:257,0:4]
cpichng_update=inflaZ(cpichng)
cpichng_result=usalphaIC(cpichng_update,ust,fields)
a=cpichng_result['Z-score']
m=len(a)
z_1=a[0:int(m/2)]
z_2=a[int(m/2):]
cpichng_IC1=alphaIC(z_1,ust,fields)
cpichng_IC2=alphaIC(z_2,ust,fields)

cpiyoy=cpi.iloc[1:257,5:9]
cpiyoy_update=inflaZ(cpiyoy)
cpiyoy_result=usalphaIC(cpiyoy_update,ust,fields)
a=cpiyoy_result['Z-score']
m=len(a)
z_1=a[0:int(m/2)]
z_2=a[int(m/2):]
cpiyoy_IC1=alphaIC(z_1,ust,fields)
cpiyoy_IC2=alphaIC(z_2,ust,fields)

cpixchng=cpi.iloc[1:257,10:14] # delay issue
cpixchng_update=inflaZ(cpixchng,sub_date=cpiyoy_update.iloc[:,4])
cpixchng_result=usalphaIC(cpixchng_update,ust,fields)
a=cpixchng_result['Z-score']
m=len(a)
z_1=a[0:int(m/2)]
z_2=a[int(m/2):]
cpixchng_IC1=alphaIC(z_1,ust,fields)
cpixchng_IC2=alphaIC(z_2,ust,fields)

cpixyoy=cpi.iloc[1:257,15:19]
cpixyoy_update=inflaZ(cpixyoy,1)
cpixyoy_result=usalphaIC(cpixyoy_update,ust,fields)
a=cpixyoy_result['Z-score']
m=len(a)
z_1=a[0:int(m/2)]
z_2=a[int(m/2):]
cpixyoy_IC1=alphaIC(z_1,ust,fields)
cpixyoy_IC2=alphaIC(z_2,ust,fields)

pceyoy=us.iloc[1:257,0:4]
pceyoy_update=inflaZ(pceyoy,1)
pceyoy_result=usalphaIC(pceyoy_update,ust,fields)
a=pceyoy_result['Z-score']
m=len(a)
z_1=a[0:int(m/2)]
z_2=a[int(m/2):]
pceyoy_IC1=alphaIC(z_1,ust,fields)
pceyoy_IC2=alphaIC(z_2,ust,fields)

pcemom=us.iloc[1:257,5:9]
pcemom_update=inflaZ(pcemom)
pcemom_result=usalphaIC(pcemom_update,ust,fields)
a=pcemom_result['Z-score']
m=len(a)
z_1=a[0:int(m/2)]
z_2=a[int(m/2):]
pcemom_IC1=alphaIC(z_1,ust,fields)
pcemom_IC2=alphaIC(z_2,ust,fields)

ppichng=us.iloc[1:257,25:29]
ppichng_update=inflaZ(ppichng)
ppichng_result=usalphaIC(ppichng_update,ust,fields)

ppiyoy=us.iloc[1:257,30:34]
ppiyoy_update=inflaZ(ppiyoy,1)
ppiyoy_result=usalphaIC(ppiyoy_update,ust,fields)

#ppixchng =us.iloc[1:257,35:37]  # does not have eco release date
#ppixchng_update=inflaZ(ppixchng,sub_date=ppixyoy_update.iloc[:,4])
#ppixchng_result=usalphaIC(ppixchng,ust,fields)

ppixyoy=us.iloc[1:257,40:44]
ppixyoy_update=inflaZ(ppixyoy)
ppixyoy_result=usalphaIC(ppixyoy_update,ust,fields)

#################################
unemployt=us.iloc[1:257,45:49]
unemployt_update=inflaZ(unemployt,1)
unemployt_result=usalphaIC(unemployt_update,ust,fields,sign=1)

unemployN=us.iloc[1:257,50:54]  # has issue with delay
unemployN_update=inflaZ(unemployN,unemployt_update.iloc[:,4],transformation=1,sign=1)
unemployN_result=usalphaIC(unemployN_update,ust,fields)
#
ahe=us.iloc[1:257,55:59]
ahe_update=inflaZ(ahe,1)
ahe_result=usalphaIC(ahe_update,ust,fields)


aheyoy=us.iloc[1:257,60:64]
aheyoy_update=inflaZ(aheyoy,1)
aheyoy_result=usalphaIC(aheyoy_update,ust,fields)

pcedefy=us2.iloc[1:257,0:4]
pcedefy_update=inflaZ(pcedefy,1)
pcedefy_result=usalphaIC(pcedefy_update,ust,fields)

fdiufdyo=us2.iloc[1:257,5:9]
fdiufdyo_update=inflaZ(fdiufdyo,1)
fdiufdyo_result=usalphaIC(fdiufdyo_update,ust,fields)

imp1yoy=us2.iloc[1:257,10:14]
imp1yoy_update=inflaZ(imp1yoy,1)
imp1yoy_result=usalphaIC(imp1yoy_update,ust,fields)

mp1comm=us2.iloc[1:257,15:19]
mp1comm_update=inflaZ(mp1comm,sub_date=imp1yoy_update.iloc[:,4],transformation=1)
mp1comm_result=usalphaIC(mp1comm_update,ust,fields)

usrfausa=us2.iloc[1:257,20:24]
usrfausa_update=inflaZ(usrfausa,sub_date=imp1yoy_update.iloc[:,4],transformation=1)
usrfausa_result=usalphaIC(usrfausa_update,ust,fields)


napmpric=us2.iloc[1:257,25:29]
napmpric_update=inflaZ(napmpric,1)
napmpric_result=usalphaIC(napmpric_update,ust,fields)




#######################################
n=len(pmi.iloc[:,0])
time=[]
for i in range(0,n):
    t = str(pmi.iloc[i,0])
    t=dt.datetime.strptime(t,'%m/%d/%Y') 
    time.append(t)
    
    
pmi_manuinput_sa=pd.DataFrame(pmi['KXUSMIP Index'])
pmi_manuinput_sa.index=time
pmi_manuinput_sa=pmi_manuinput_sa.diff()
pmi_manuinput_nsa=pd.DataFrame(pmi['KXUSMIPU Index'])
pmi_manuinput_nsa.index=time
pmi_manuinput_nsa=pmi_manuinput_nsa.diff()

pmi_manuoutput_sa=pd.DataFrame(pmi['KXUSMOB Index'])
pmi_manuoutput_sa.index=time
pmi_manuoutput_sa=pmi_manuoutput_sa.diff()
pmi_manuoutput_nsa=pd.DataFrame(pmi['KXUSMOBU Index'])
pmi_manuoutput_nsa.index=time
pmi_manuoutput_nsa=pmi_manuoutput_nsa.diff()

pmi_serinput_sa=pd.DataFrame(pmi['KXUSEIP Index'])
pmi_serinput_sa.index=time
pmi_serinput_sa=pmi_serinput_sa.diff()
pmi_serinput_nsa=pd.DataFrame(pmi['KXUSEIPU Index'])
pmi_serinput_nsa.index=time
pmi_serinput_nsa=pmi_serinput_nsa.diff()

pmi_seroutput_sa=pd.DataFrame(pmi['KXUSEOP Index'])
pmi_seroutput_sa.index=time
pmi_seroutput_sa=pmi_seroutput_sa.diff()
pmi_seroutput_nsa=pd.DataFrame(pmi['KXUSEOPU Index'])
pmi_seroutput_nsa.index=time
pmi_seroutput_nsa=pmi_seroutput_nsa.diff()

pmi_compinput_sa=pd.DataFrame(pmi['KXUSCIP Index'])
pmi_compinput_sa.index=time
pmi_compinput_sa=pmi_compinput_sa.diff()
#pmi_compinput_nsa=pmi['KXUSCIPU Index']
#pmi_compinput_nsa.index=pmi.iloc[:,0]

pmi_compoutput_sa=pd.DataFrame(pmi['KXUSCOP Index'])
pmi_compoutput_sa.index=time
pmi_compoutput_sa=pmi_compoutput_sa.diff()
#pmi_compoutput_nsa=pmi['KXUSCOPU Index']
#pmi_compoutput_nsa.index=pmi.iloc[:,0]

pmi_min_result=usalphaIC(pmi_manuinput_sa,ust,fields,col=1)
pmi_minnsa_result=usalphaIC(pmi_manuinput_nsa,ust,fields,col=1)

pmi_mout_result=usalphaIC(pmi_manuoutput_sa,ust,fields,col=1)
pmi_moutnsa_result=usalphaIC(pmi_manuoutput_nsa,ust,fields,col=1)

pmi_sin_result=usalphaIC(pmi_serinput_sa,ust,fields,col=1)
pmi_sinnsa_result=usalphaIC(pmi_serinput_nsa,ust,fields,col=1)

pmi_sout_result=usalphaIC(pmi_seroutput_sa,ust,fields,col=1)
pmi_soutnsa_result=usalphaIC(pmi_seroutput_nsa,ust,fields,col=1)

pmi_cin_result=usalphaIC(pmi_compinput_sa,ust,fields,col=1)
pmi_cout_result=usalphaIC(pmi_compoutput_sa,ust,fields,col=1)

#################
lagInd=pd.DataFrame(np.array(d3_result['Z-score']),index=dt_range,columns=['CPI'])
lagInd['CPIX']=d6_result['Z-score']
lagInd['PPI']=d9_result['Z-score']
#lagInd['PPI']=d3_result['Z-score']
#lagInd['PPIX']=d4_result['Z-score']

employInd=pd.DataFrame(np.array(ew1_result['Z-score']),index=dt_range,columns=['Unemployment Rate'])
employInd['AHEYoY']=ew2_result['Z-score']

pmiInd=pd.DataFrame(np.array(pmi_min_result['Z-score']),index=dt_range,columns=['PMI Manu Input'])



lagInd_m=pd.DataFrame(lagInd.mean(axis=1))
m=len(lagInd_m)
lagInd_1=lagInd_m[0:int(m/2)]
lagInd_2=lagInd_m[int(m/2):]
lagInd_IC=alphaIC(lagInd_m,ust,fields)
lagInd_IC1=alphaIC(lagInd_1,ust,fields)
lagInd_IC2=alphaIC(lagInd_2,ust,fields)


employInd_m=pd.DataFrame(employInd.mean(axis=1))
employInd_IC=alphaIC(employInd_m,ust,fields)
m=len(employInd_m)
employInd_1=employInd_m[0:int(m/2)]
employInd_2=employInd_m[int(m/2):]
employInd_IC1=alphaIC(employInd_1,ust,fields)
employInd_IC2=alphaIC(employInd_2,ust,fields)


pmiInd_m=pd.DataFrame(pmiInd.mean(axis=1))
pmiInd_IC=alphaIC(pmiInd_m,ust,fields)
m=len(pmiInd_m)
pmiInd_1=pmiInd_m[0:int(m/2)]
pmiInd_2=pmiInd_m[int(m/2):]
pmiInd_IC1=alphaIC(pmiInd_1,ust,fields)
pmiInd_IC2=alphaIC(pmiInd_2,ust,fields)




totalComp=pd.DataFrame(np.array(lagInd_m),index=dt_range)
totalComp['employment']=employInd_m
totalComp['leading']=pmiInd_m
totalComp_m=pd.DataFrame(totalComp.mean(axis=1))
totalComp_IC=alphaIC(totalComp_m,ust,fields)
m=len(totalComp_m)
totalComp_1=totalComp_m[0:int(m/2)]
totalComp_2=totalComp_m[int(m/2):]
totalComp_IC1=alphaIC(totalComp_1,ust,fields)
totalComp_IC2=alphaIC(totalComp_2,ust,fields)





tc=pd.DataFrame(np.array(lagInd),index=dt_range)
tc['unemployment']=employInd['Unemployment Rate']
tc['wage']=employInd['AHEYoY']
tc['PMI Manu Input']=pmiInd['PMI Manu Input']


tc_m=pd.DataFrame(tc.mean(axis=1))
tc_IC=alphaIC(tc_m,ust,fields)
m=len(tc_m)
tc_1=tc_m[0:int(m/2)]
tc_2=tc_m[int(m/2):]
tc_IC1=alphaIC(tc_1,ust,fields)
tc_IC2=alphaIC(tc_2,ust,fields)




result=pd.DataFrame(np.array(d1_result['IC']),columns=['IC_1'])
result['pval1']=d1_result['pval']

result['ic2']=d2_result['IC']
result['pval2']=d2_result['pval']
result['ic3']=d3_result['IC']
result['pval3']=d3_result['pval']
result['ic4']=d4_result['IC']
result['pval4']=d4_result['pval']
result['ic5']=d5_result['IC']
result['pval5']=d5_result['pval']
result['ic6']=d6_result['IC']
result['pval6']=d6_result['pval']
result['ic7']=d7_result['IC']
result['pval7']=d7_result['pval']
result['ic8']=d8_result['IC']
result['pval8']=d8_result['pval']
result['ic9']=d9_result['IC']
result['pval9']=d9_result['pval']
result['ic10']=ew1_result['IC']
result['pval10']=ew1_result['pval']
result['ic11']=ew2_result['IC']
result['pval11']=ew2_result['pval']

result['ic12']=pmi_min_result['IC']
result['pval12']=pmi_min_result['pval']


result['icc1']=lagInd_IC['IC']
result['pvalc1']=lagInd_IC['pval']
result['icc2']=lagInd_IC1['IC']
result['pvalc2']=lagInd_IC1['pval']
result['icc3']=lagInd_IC2['IC']
result['pvalc3']=lagInd_IC2['pval']
#
result['icc4']=employInd_IC['IC']
result['pvalc4']=employInd_IC['pval']
result['icc5']=employInd_IC1['IC']
result['pvalc5']=employInd_IC1['pval']
result['icc6']=employInd_IC2['IC']
result['pvalc6']=employInd_IC2['pval']

result['icc7']=pmiInd_IC['IC']
result['pvalc7']=pmiInd_IC['pval']
result['icc8']=pmiInd_IC1['IC']
result['pvalc8']=pmiInd_IC1['pval']
result['icc9']=pmiInd_IC2['IC']
result['pvalc9']=pmiInd_IC2['pval']


result['cic1']=totalComp_IC['IC']
result['cpval1']=totalComp_IC['pval']
result['cic2']=totalComp_IC1['IC']
result['cpval2']=totalComp_IC1['pval']
result['cic3']=totalComp_IC2['IC']
result['cpval3']=totalComp_IC2['pval']



result['cic7']=tc_IC['IC']
result['cpva7']=tc_IC['pval']
result['cic8']=tc_IC1['IC']
result['cpval8']=tc_IC1['pval']
result['cic9']=tc_IC2['IC']
result['cpval9']=tc_IC2['pval']



result.to_csv('temp result.csv',index=False)
