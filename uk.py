# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 16:24:41 2016

@author: e612727
"""

import pandas as pd
import numpy
from datetime import datetime, timedelta
import datetime as dt
import xlwt
from csv import DictWriter

jp=pd.read_excel('uk.xlsx',na_values=['NA'])
rates_return=pd.read_csv('rates_futures_returns.csv',na_values=['NA'])
pmi=pd.read_csv('PX_LAST.csv',na_values=['NA'])
dmfx=pd.read_csv('dmfx_returns.csv')

rates=rates_return.iloc[:,1:]
start_dt = dt.datetime(1999, 1, 1)
end_dt =  dt.datetime(2016, 4, 29)
freq = 'B' # business-days
dt_range = pd.date_range(start=start_dt, end=end_dt, freq=freq, normalize=True)
    
rates_return.index=dt_range
ust=pd.DataFrame(rates_return['GBP_3M'])
ust['GBP_2Y']=rates_return['GBP_2Y']
ust['GBP_10Y']=rates_return['GBP_10Y']

dt_range1=pd.date_range(start=start_dt,end=dt.datetime(2015,12,17),freq=freq,normalize=True)    


fx=pd.DataFrame(dmfx['GBP'])
fx.index=dt_range1


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
fields=['GBP_3M','GBP_2Y','GBP_10Y']                           
d1=jp.iloc[1:257,0:4]
d1_update=inflaZ(d1,1)
d1_result=usalphaIC(d1_update,ust,fields)

d2=jp.iloc[1:257,5:9]
d2_update=inflaZ(d2)
d2_result=usalphaIC(d2_update,ust,fields)

d3=jp.iloc[1:257,10:14] 
d3_update=inflaZ(d3,1)
d3_result=usalphaIC(d3_update,ust,fields)

d4=jp.iloc[1:257,15:19] # does not have eco release date
d4_update=inflaZ(d4,sub_date=d5_update.iloc[:,4],transformation=1)
d4_result=usalphaIC(d4_update,ust,fields)

d5=jp.iloc[1:257,20:24]
d5_update=inflaZ(d5)
d5_result=usalphaIC(d5_update,ust,fields)

d6=jp.iloc[1:257,25:29]
d6_update=inflaZ(d6,1)
d6_result=usalphaIC(d6_update,ust,fields)

#d7 =jp.iloc[1:257,30:34]  # does not have eco release date
#d7_update=inflaZ(d7)
#d7_result=usalphaIC(d7_update,ust,fields)
#
#d8=jp.iloc[1:257,35:39]
#d8_update=inflaZ(d8)
#d8_result=usalphaIC(d8_update,ust,fields)

ew1=jp.iloc[1:257,30:34]  
ew1_update=inflaZ(ew1,1)
ew1_result=usalphaIC(ew1_update,ust,fields,sign=1)


ew2=jp.iloc[1:257,35:39] # does not have eco release date
ew2_update=inflaZ(ew2,sub_date=ew1_update.iloc[:,4],transformation=1)
ew2_result=usalphaIC(ew2_update,ust,fields)



#######################################
n=len(pmi.iloc[:,0])
time=[]
time2=dt.timedelta(1)
for i in range(0,n):
    t = str(pmi.iloc[i,0])
    t=dt.datetime.strptime(t,'%m/%d/%Y') 
    t=t+time2
    time.append(t)
   
pmi_manuinput_sa=pd.DataFrame(pmi['KXGBMIP Index'])
pmi_manuinput_sa.index=time
pmi_manuinput_sa=pmi_manuinput_sa.diff()


pmi_manuoutput_sa=pd.DataFrame(pmi['KXGBMOB Index'])
pmi_manuoutput_sa.index=time
pmi_manuoutput_sa=pmi_manuoutput_sa.diff()


pmi_serinput_sa=pd.DataFrame(pmi['KXGBEIP Index'])
pmi_serinput_sa.index=time
pmi_serinput_sa=pmi_serinput_sa.diff()


pmi_seroutput_sa=pd.DataFrame(pmi['KXGBEOP Index'])
pmi_seroutput_sa.index=time
pmi_seroutput_sa=pmi_seroutput_sa.diff()


pmi_compinput_sa=pd.DataFrame(pmi['KXGBCIP Index'])
pmi_compinput_sa.index=time
pmi_compinput_sa=pmi_compinput_sa.diff()
#pmi_compinput_nsa=pmi['KXUSCIPU Index']
#pmi_compinput_nsa.index=pmi.iloc[:,0]

pmi_compoutput_sa=pd.DataFrame(pmi['KXGBCHE Index'])
pmi_compoutput_sa.index=time
pmi_compoutput_sa=pmi_compoutput_sa.diff()
#pmi_compoutput_nsa=pmi['KXUSCOPU Index']
#pmi_compoutput_nsa.index=pmi.iloc[:,0]

pmi_min_result=usalphaIC(pmi_manuinput_sa,ust,fields,col=1)
pmi_mout_result=usalphaIC(pmi_manuoutput_sa,ust,fields,col=1)
pmi_sin_result=usalphaIC(pmi_serinput_sa,ust,fields,col=1)
pmi_sout_result=usalphaIC(pmi_seroutput_sa,ust,fields,col=1)
pmi_cin_result=usalphaIC(pmi_compinput_sa,ust,fields,col=1)
pmi_cout_result=usalphaIC(pmi_compoutput_sa,ust,fields,col=1)

##############################


lagInd=pd.DataFrame(np.array(d3_result['Z-score']),index=dt_range,columns=['CPI'])
lagInd['CPIX']=d6_result['Z-score']
#lagInd['PCE']=d2_result['Z-score']
#lagInd['PPI']=d3_result['Z-score']
#lagInd['PPIX']=d4_result['Z-score']

employInd=pd.DataFrame(np.array(ew1_result['Z-score']),index=dt_range,columns=['Unemployment Rate'])
employInd['AHEYoY']=ew2_result['Z-score']

pmiInd=pd.DataFrame(np.array(pmi_min_result['Z-score']),index=dt_range,columns=['PMI Manu Input'])
pmiInd['PMI Manu Output']=pmi_mout_result['Z-score']
pmiInd['PMI Ser Input']=pmi_sin_result['Z-score']
pmiInd['PMI Ser Output']=pmi_sout_result['Z-score']

pmiIndComp=pd.DataFrame(np.array(pmi_cin_result['Z-score']),index=dt_range,columns=['Composite Input'])
pmiIndComp['Composite Output']=pmi_cout_result['Z-score']

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

pmiIndComp_m=pd.DataFrame(pmiIndComp.mean(axis=1))
pmiIndComp_IC=alphaIC(pmiIndComp_m,ust,fields)
m=len(pmiIndComp_m)
pmiIndComp_1=pmiIndComp_m[0:int(m/2)]
pmiIndComp_2=pmiIndComp_m[int(m/2):]
pmiIndComp_IC1=alphaIC(pmiIndComp_1,ust,fields)
pmiIndComp_IC2=alphaIC(pmiIndComp_2,ust,fields)


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

totalCompC=lagInd_m
totalCompC['employment']=employInd_m
totalCompC['leading']=pmiIndComp_m
totalCompC_m=pd.DataFrame(totalCompC.mean(axis=1))
totalCompC_IC=alphaIC(totalCompC_m,ust,fields)
m=len(totalCompC_m)
totalCompC_1=totalCompC_m[0:int(m/2)]
totalCompC_2=totalCompC_m[int(m/2):]
totalCompC_IC1=alphaIC(totalCompC_1,ust,fields)
totalCompC_IC2=alphaIC(totalCompC_2,ust,fields)



tc=pd.DataFrame(np.array(lagInd),index=dt_range)
tc['unemployment']=employInd['Unemployment Rate']
tc['wage']=employInd['AHEYoY']
tc['PMI Manu Input']=pmiInd['PMI Manu Input']
tc['PMI Manu Output']=pmiInd['PMI Manu Output']
tc['PMI Ser Input']=pmiInd['PMI Ser Input']
tc['PMI Ser Output']=pmiInd['PMI Ser Output']

tc_m=pd.DataFrame(tc.mean(axis=1))
tc_IC=alphaIC(tc_m,ust,fields)
m=len(tc_m)
tc_1=tc_m[0:int(m/2)]
tc_2=tc_m[int(m/2):]
tc_IC1=alphaIC(tc_1,ust,fields)
tc_IC2=alphaIC(tc_2,ust,fields)


tcC=pd.DataFrame(np.array(lagInd),index=dt_range)
tcC['unemployment']=employInd['Unemployment Rate']
tcC['wage']=employInd['AHEYoY']
tcC['Composite Input']=pmiIndComp['Composite Input']
tcC['Composite Output']=pmiIndComp['Composite Output']

tcC_m=pd.DataFrame(tcC.mean(axis=1))
tcC_IC=alphaIC(tcC_m,ust,fields)
m=len(tcC_m)
tcC_1=tcC_m[0:int(m/2)]
tcC_2=tcC_m[int(m/2):]
tcC_IC1=alphaIC(tcC_1,ust,fields)
tcC_IC2=alphaIC(tcC_2,ust,fields)


result=pd.DataFrame(np.array(d1_result['IC']),columns=['IC_1'])
result['pval1']=d1_result['pval']

result['ic2']=d2_result['IC']
result['pval2']=d2_result['pval']
result['ic3']=d3_result['IC']
result['pval3']=d3_result['pval']
#result['ic4']=d4_result['IC']
#result['pval4']=d4_result['pval']
result['ic5']=d5_result['IC']
result['pval5']=d5_result['pval']
result['ic6']=d6_result['IC']
result['pval6']=d6_result['pval']
#result['ic7']=d7_result['IC']
#result['pval7']=d7_result['pval']
#result['ic8']=d8_result['IC']
#result['pval8']=d8_result['pval']
result['ic9']=ew1_result['IC']
result['pval9']=ew1_result['pval']
result['ic10']=ew2_result['IC']
result['pval0']=ew2_result['pval']

result['ic11']=pmi_min_result['IC']
result['pval11']=pmi_min_result['pval']
result['ic12']=pmi_mout_result['IC']
result['pval12']=pmi_mout_result['pval']
result['ic13']=pmi_sin_result['IC']
result['pval13']=pmi_sin_result['pval']
result['ic14']=pmi_sout_result['IC']
result['pval14']=pmi_sout_result['pval']
result['ic15']=pmi_cin_result['IC']
result['pval15']=pmi_cin_result['pval']
result['ic16']=pmi_cout_result['IC']
result['pval16']=pmi_cout_result['pval']

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

result['icc10']=pmiIndComp_IC['IC']
result['pvalc10']=pmiIndComp_IC['pval']
result['icc11']=pmiIndComp_IC1['IC']
result['pvalc11']=pmiIndComp_IC1['pval']
result['icc12']=pmiIndComp_IC2['IC']
result['pvalc12']=pmiIndComp_IC2['pval']

result['cic1']=totalComp_IC['IC']
result['cpval1']=totalComp_IC['pval']
result['cic2']=totalComp_IC1['IC']
result['cpval2']=totalComp_IC1['pval']
result['cic3']=totalComp_IC2['IC']
result['cpval3']=totalComp_IC2['pval']

result['cic4']=totalCompC_IC['IC']
result['cpva4']=totalCompC_IC['pval']
result['cic5']=totalCompC_IC1['IC']
result['cpval5']=totalCompC_IC1['pval']
result['cic6']=totalCompC_IC2['IC']
result['cpval6']=totalCompC_IC2['pval']

result['cic7']=tc_IC['IC']
result['cpva7']=tc_IC['pval']
result['cic8']=tc_IC1['IC']
result['cpval8']=tc_IC1['pval']
result['cic9']=tc_IC2['IC']
result['cpval9']=tc_IC2['pval']

result['cic10']=tcC_IC['IC']
result['cpva10']=tcC_IC['pval']
result['cic11']=tcC_IC1['IC']
result['cpval11']=tcC_IC1['pval']
result['cic12']=tcC_IC2['IC']
result['cpval12']=tcC_IC2['pval']

result.to_csv('temp result.csv',index=False)