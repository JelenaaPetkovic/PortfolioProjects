"""
Data set contains 18 regions of France and 88 features. The goal is to perform Comparison of the regions, Statistics for whole France, Map visualization.
Libraries used: Numpy, Pandas, Matplotlib, Seaborn, Cartopy 
"""
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs 
import cartopy.feature as cfeature

os.chdir("C:\\Users\\user\\Desktop\\Projekat")
data = pd.read_excel('population_France_2016_2.xlsx')

#Dealing with missing values
print('Shape of population data:')
print(data.shape)

data_copy=data.copy()
data1=data_copy.dropna()
print('Shape of population data after removing NaN values:')
print(data1.shape)

data2=data_copy.dropna(how='all') 
print(data2.shape)

missing=data.isnull().sum().sum()
print(missing)

data3=data_copy.dropna(axis='columns')
print(data3.shape)

data['partPop75p']=data['partPop75p'].fillna(100-data['partPop0_14']-data['partPop15_24']-data['partPop25_59']-data['partPop60_74'])
data['partPop75p']
data['partPop75pF']=data['partPop75pF'].fillna(100-data['partPop0_14F']-data['partPop15_24F']-data['partPop25_59F']-data['partPop60_74F'])
data['partPop75pF']

data.iloc[14:17, 17]
data.iloc[14:17,18]
mean_f_m=(data.iloc[14:17,18]-data.iloc[14:17, 17]).mean()
data.iloc[17:18,17]= data.iloc[17:18,17]+ mean_f_m 
data.iloc[17:18,18]
data['Male youth index'][17]
data['Male youth index']

data['partPop75pEt'][16]=round(100-data['partPop0_14Et'][16]-data['partPop15_24Et'][16]-data['partPop25_59Et'][16]-data['partPop60_74Et'][16],1)
data['partPop75pEt'][16]
round((data['partPop60_74Et'][16] + data['partPop60_74Et'][14])/2,1)
data['partPop60_74Et']=data['partPop60_74Et'].fillna(round((data['partPop60_74Et'][16] + data['partPop60_74Et'][14])/2,1)
data['partPop60_74Et']
data['partPop75pEt']=data['partPop75pEt'].fillna(data['partPop75pEt'][16])
data['partPop75pEt']
data['partPop15_24Et']=data['partPop15_24Et'].fillna(100-data['partPop0_14Et']-data['partPop75pEt']-data['partPop25_59Et']-data['partPop60_74Et'])
data['partPop15_24Et']
data['Youth index of foreigners']=data['Youth index of foreigners'].fillna(data['Youth index of foreigners'][16])
data['Youth index of foreigners']
data['Percentage of adults with French nationality'][15]=100-data['partPopEt'][15]
data['Percentage of adults with French nationality']
data['Er15_24']=data['Er15_24'].fillna(np.mean(data['Er15_24']))
data['Er15_24']=round(data['Er15_24'],1)
round(((data['partEmpPrecF'][16]-data['partEmpPrec'][16]) + (data['partEmpPrecF'][15]-data['partEmpPrec'][15])+ (data['partEmpPrecF'][14]-data['partEmpPrec'][14]))/3,1)
data['partEmpPrecF'][17]=data['partEmpPrec'][17]+round(((data['partEmpPrecF'][16]-data['partEmpPrec'][16]) + (data['partEmpPrecF'][15]-data['partEmpPrec'][15])+ (data['partEmpPrecF'][14]-data['partEmpPrec'][14]))/3,1)
data['partEmpPrecF']
data['partEmpPrecEt'][16]-data['partEmpPrecF'][16]
data['partEmpPrecEt'][17]=data['partEmpPrecF'][17] + (data['partEmpPrecEt'][16]-data['partEmpPrecF'][16])
data['partEmpPrecEt'][17]
data['partEmpPrecEt'][15]=data['partEmpPrecF'][15] + (data['partEmpPrecEt'][16]-data['partEmpPrecF'][16])
data['partEmpPrecEt'][14]=data['partEmpPrecF'][14] + (data['partEmpPrecEt'][16]-data['partEmpPrecF'][16])
data['partEmpPrecEt']
a=round((data['txAct25_54'][16]+ data['txAct25_54'][15]+ data['txAct25_54'][14])/3,1)
data['txAct25_54'][17]=a
data['txAct25_54'] [17]
data_new=data.fillna(round(np.mean(data),1))
                                                     
# Comparison of the Regions
regions=data_new['Name of region']
pop0_14= data_new['partPop0_14']
pop15_24=data_new['partPop15_24']
pop25_59=data_new['partPop25_59']
pop60_74=data_new['partPop60_74']
pop75=data_new['partPop75p']
fig=plt.figure(figsize=(10,6))

fig.add_subplot(151)
data1=np.asarray(pop0_14).reshape(18,1)
sns.heatmap(data1, cmap='YlGnBu', cbar=True,annot=True, fmt='g', yticklabels=regions, xticklabels=False, vmin=pop0_14.min(), vmax=pop0_14.max())
plt.ylabel('Regions', fontweight='bold')
plt.xlabel('% of population between 0-14 yo', rotation=8)

fig.add_subplot(152)
data2=np.asarray(pop15_24).reshape(18,1)
sns.heatmap(data2, cmap='YlGnBu', cbar=True,annot=True, fmt='g', yticklabels=False, xticklabels=False, vmin=pop15_24.min(), vmax=pop15_24.max())
plt.xlabel('% of population between 15-24 yo', rotation=8)

fig.add_subplot(153)
data3=np.asarray(pop25_59).reshape(18,1)
sns.heatmap(data3, cmap='YlGnBu', cbar=True,annot=True, fmt='g', yticklabels=False, xticklabels=False, vmin=pop25_59.min(), vmax=pop25_59.max())
plt.xlabel('% of population between 25-59 yo', rotation=8)
plt.title('Overview of percentages of population among ages, among all regions', fontweight='bold' )


fig.add_subplot(154)
data4=np.asarray(pop60_74).reshape(18,1)
sns.heatmap(data4, cmap='YlGnBu', cbar=True,annot=True, fmt='g', yticklabels=False, xticklabels='', vmin=pop60_74.min(), vmax=pop60_74.max())
plt.xlabel('% of population between 60-74 yo', rotation=8)

fig.add_subplot(155)
data5=np.asarray(pop75).reshape(18,1)
sns.heatmap(data5, cmap='YlGnBu', cbar=True,annot=True, fmt='g', yticklabels=False, xticklabels=False, vmin=pop75.min(), vmax=pop75.max())
plt.xlabel('% of population over 75 yo', rotation=8)

plt.autoscale()
plt.savefig('%agesallregions', bbox_inches="tight")

plt.show()

fig=plt.figure(figsize=(12,12))
m=plt.axes(projection=ccrs.PlateCarree())
m.set_extent([-80,20,0,60], ccrs.PlateCarree())
m.add_feature(cfeature.LAND)
m.add_feature(cfeature.OCEAN)
m.add_feature(cfeature.COASTLINE)
m.add_feature(cfeature.BORDERS, linestyle=':')
m.add_feature(cfeature.LAKES, alpha=0.5)
m.add_feature(cfeature.RIVERS)
m.coastlines()
m.stock_img()
labels=[" Guyane 36.9% (0-14 yo) & 17.3% (15-24 yo)", "Île-de-France 45.9% (25-59 yo)", "Corse 16.7% (60-74 yo) & 10.8% (75- yo)"]
size=[25*5, 40*5, 15*5]
longitude=[-52.326,2.352222, 8.7369]
latitude=[4.9372, 48.856613,41.9267]
scatter=m.scatter(longitude, latitude, s=size, c=range(len(labels)), transform=ccrs.PlateCarree())
handles,_= scatter.legend_elements(prop='colors')
plt.legend(handles, labels, loc=(1.00, 0.5), title='Legend');
plt.title('Map of regions with highest percentage of population among ages', fontweight='bold')
plt.savefig('mapazagodine.png', bbox_inches="tight")
plt.show()

import matplotlib.pyplot as plt
def addlabels(x,y):  
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha='center')
        
x=data_new['Name of region']  
y=data_new['partPopF']  
plt.figure(figsize=(10,5)) 
plt.bar(x,y, color='b')  
plt.xticks(x, rotation=86)  
addlabels(x,y) 
plt.ylabel('Percentage', fontweight='bold')
plt.xlabel('Region name', fontweight='bold')
plt.title(' Percentage of women in population among all regions', fontweight='bold')
plt.autoscale()
plt.savefig('zeneregioni.png', bbox_inches="tight") 

plt.show()
                                                     
fig=plt.figure(figsize=(12,12))
m0=plt.axes(projection=ccrs.PlateCarree())
m0.set_extent([-80,20,0,60], ccrs.PlateCarree())
m0.add_feature(cfeature.LAND)
m0.add_feature(cfeature.OCEAN)
m0.add_feature(cfeature.COASTLINE)
m0.add_feature(cfeature.BORDERS, linestyle=':')
m0.add_feature(cfeature.LAKES, alpha=0.5)
m0.add_feature(cfeature.RIVERS)
m0.coastlines()
m0.stock_img()
labels=["Guadeloupe 56.7%", "Île-de-France 51%"]
size=[56.7*3,51*3]
longitude=[-61.7292,2.352222 ]
latitude=[15.9958, 48.856613]
scatter=m0.scatter(longitude, latitude, s=size, c=range(len(labels)), transform=ccrs.PlateCarree())
handles,_= scatter.legend_elements(prop='colors')
plt.legend(handles, labels, loc=(1.00, 0.5), title='Legend');
plt.title('Map of regions with highest and lowest percentage of women', fontweight='bold')
plt.savefig('mapazazene.png', bbox_inches="tight")
plt.show()
                         
import matplotlib.pyplot as plt
def addlabels(x,y):
    for i in range(len(x)):     
        plt.text(i, y[i], y[i], ha='center')
x=data_new['Name of region']
y=data_new['partPopEt']
plt.figure(figsize=(10,5))
plt.bar(x,y, color='green')
plt.xticks(x, rotation=86)
addlabels(x,y)
plt.ylabel('Percentage', fontweight='bold' )
plt.xlabel('Region name', fontweight='bold')
plt.title(' Percentage of foreigners in the population among all regions', fontweight='bold')
plt.autoscale()
plt.savefig('stranciregioni.png', bbox_inches="tight") 

plt.show()
                                           
fig=plt.figure(figsize=(12,12))
m1=plt.axes(projection=ccrs.PlateCarree())
m1.set_extent([-80,20,0,60], ccrs.PlateCarree())
m1.add_feature(cfeature.LAND)
m1.add_feature(cfeature.OCEAN)
m1.add_feature(cfeature.COASTLINE)
m1.add_feature(cfeature.BORDERS, linestyle=':')
m1.add_feature(cfeature.LAKES, alpha=0.5)
m1.add_feature(cfeature.RIVERS)
m1.coastlines()
m1.stock_img()
labels=['Guyane 42.3%', 'Saint-Martin 40.3%', 'Martinique 10.4%', 'Guadeloupe 10.6%']
size=[42.3*5,40.3*5,10.4*5, 10.6*5] 
longitude=[-52.326, -63.0822, -61.0833, - 61.792]
latitude=[4.9372, 18.0731, 14.6, 15.9958]

scatter=m1.scatter(longitude, latitude, s=size, c=range(len(labels)), transform=ccrs.PlateCarree())
handles,_= scatter.legend_elements(prop='colors')
plt.legend(handles, labels, loc=(1.00, 0.5), title='Legend');
plt.title('Map of regions with highest and lowest percentage of foreigners', fontweight='bold')
plt.savefig('mapazastrance.png', bbox_inches="tight")
plt.show()

                                                     
import matplotlib.pyplot as plt
def addlabels(x,y):
    for i in range(len(x)):    
        plt.text(i, y[i], y[i], ha='center')  
x=data_new['Name of region']
y=data_new['partPopImmi']
plt.figure(figsize=(10,5))
plt.bar(x,y, color='tab:olive')
plt.xticks(x, rotation=86) 
addlabels(x,y) 
plt.ylabel('Percentage', fontweight='bold')
plt.xlabel('Region name', fontweight='bold')
plt.title(' Percentage of immigrants in the population among all regions', fontweight='bold')
plt.autoscale()
plt.savefig('imigrantiregioni.png', bbox_inches="tight") 

plt.show()

                                                     
fig=plt.figure(figsize=(12,12))
m3=plt.axes(projection=ccrs.PlateCarree())
m3.set_extent([-80,20,0,60], ccrs.PlateCarree())
m3.add_feature(cfeature.LAND)
m3.add_feature(cfeature.OCEAN)
m3.add_feature(cfeature.COASTLINE)
m3.add_feature(cfeature.BORDERS, linestyle=':')
m3.add_feature(cfeature.LAKES, alpha=0.5)
m3.add_feature(cfeature.RIVERS)
m3.coastlines()
m3.stock_img()
labels=['Saint-Martin 36.7%','Guadeloupe 9.2%']
size=[36.7*5, 9.2*5]  
longitude=[-63.0822,- 61.792]
latitude=[18.0731,15.9958]

scatter=m3.scatter(longitude, latitude, s=size, c=range(len(labels)), transform=ccrs.PlateCarree())
handles,_= scatter.legend_elements(prop='colors')
plt.legend(handles, labels, loc=(1.00, 0.5), title='Legend');
plt.title('Regions with highest and lowest percentage of immigrants in the population', fontweight='bold')
plt.savefig('mapazaimigrante.png', bbox_inches="tight")
plt.show()

                                                     
fig=plt.figure(figsize=(12,12))
m3=plt.axes(projection=ccrs.PlateCarree())
m3.set_extent([-70,-50,7,20], ccrs.PlateCarree())
m3.add_feature(cfeature.LAND)
m3.add_feature(cfeature.OCEAN)
m3.add_feature(cfeature.COASTLINE)
m3.add_feature(cfeature.BORDERS, linestyle=':')
m3.add_feature(cfeature.LAKES, alpha=0.5)
m3.add_feature(cfeature.RIVERS)
m3.coastlines()
m3.stock_img()
labels=['Saint-Martin 36.7%','Guadeloupe 9.2%']
size=[36.7*10, 9.2*10]  
longitude=[-63.0822,- 61.792]
latitude=[18.0731,15.9958]

scatter=m3.scatter(longitude, latitude, s=size, c=range(len(labels)), transform=ccrs.PlateCarree())
handles,_= scatter.legend_elements(prop='colors')
plt.legend(handles, labels, loc=(1.00, 0.5), title='Legend');
plt.title('Regions with highest and lowest percentage of immigrants in the population', fontweight='bold')
plt.savefig('mapazaimigranteuvecano.png', bbox_inches="tight")
plt.show()
                                                     

# ...
                                                     
# Statistics for whole France
values=[round(data_new['partPop0_14'].mean(),2), round(data_new['partPop15_24'].mean(),2), round(data_new['partPop25_59'].mean(),2),round(data_new['partPop60_74'].mean(),2), round(data_new['partPop75p'].mean(),2)]
ages=["0-14", "15-24", "25-59", "60-74","75-"]


fig=plt.figure() 
ax=fig.add_subplot(111)
ax.grid()  

fig.autofmt_xdate(rotation=20) 
plt.plot(ages, values, 'rx')
plt.xlabel('Ages') 
plt.ylabel('Values') 
plt.title('Overview of percentage of population among all ages', fontweight='bold')
plt.savefig('cela_populacija.png')
plt.show()

values=[round(data_new['partPop0_14F'].mean(),2), round(data_new['partPop15_24F'].mean(),2), round(data_new['partPop25_59F'].mean(),2),round(data_new['partPop60_74F'].mean(),2), round(data_new['partPop75pF'].mean(),2)]
ages=["0-14", "15-24", "25-59", "60-74","75-"]
fig=plt.figure() 
ax=fig.add_subplot(111)
ax.grid()  

fig.autofmt_xdate(rotation=20) 
plt.plot(ages, values, 'bx')
plt.xlabel('Ages') 
plt.ylabel('Values') 
plt.title('Overview of percentage of women among all ages', fontweight='bold')
plt.savefig('zene_cela_populacija.png')
plt.show()
                                                     
values=[round(data_new['partPop0_14Et'].mean(),2), round(data_new['partPop15_24Et'].mean(),2), round(data_new['partPop25_59Et'].mean(),2),round(data_new['partPop60_74Et'].mean(),2), round(data_new['partPop75pEt'].mean(),2)]   
ages=["0-14", "15-24", "25-59", "60-74","75-"]
fig=plt.figure() 
ax=fig.add_subplot(111)
ax.grid()  

fig.autofmt_xdate(rotation=20) 
plt.plot(ages, values, 'cx')
plt.xlabel('Ages') 
plt.ylabel('Values') 
plt.title('Overview of percentage of foreigners among all ages', fontweight='bold')
plt.savefig('stranci_cela_populacija.png')
plt.show()
                                                     
def addlabels(x,y):  
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha='center')
x=['percentage of women', 'percentage of men']
y=[round(data_new['partPopF'].mean(),1), 100-round(data_new['partPopF'].mean(),1)]
plt.figure(figsize=(3,3))
plt.bar(x,y,color=['tab:olive', 'tab:cyan'], alpha=0.5)
addlabels(x,y)
plt.xticks(x, rotation=12)
plt.ylabel('Percentage')
plt.title('Overview of percentage of women and men', fontweight='bold')

plt.savefig("wholefrancewomenandman", bbox_inches="tight" )
                                                     
def addlabels(x,y):  
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha='center')
x=['% of adults with French nationality', '% of foreigners']
y=[round(data_new['Percentage of adults with French nationality'].mean(),1), round(data_new['partPopEt'].mean(),1)]
plt.figure(figsize=(3.5,3.5))
plt.bar(x,y,color=['tab:blue', 'tab:brown'], alpha=0.5)
addlabels(x,y)
plt.xticks(x, rotation=60)
plt.ylabel('Percentage')
plt.title('Percentage of adults with French nationality and foreigners in the population', fontweight='bold')

plt.savefig("wholeFrancefrenchforeignersimmigrants", bbox_inches="tight" )
                                                     
barWidth=0.25  

bars1=[round(data_new['partPopBepCap'].mean(),1),round(data_new['partPopBac'].mean(),1), round(data_new['partPopBacSup'].mean(),1)]
bars2= [round(data_new['partPopBepCapF'].mean(),1), round(data_new['partPopBacF'].mean(),1), round(data_new['partPopBacSupF'].mean(),1)]
bars3= [round(data_new['partPopBepCapEt'].mean(),1), round(data_new['partPopBacEt'].mean(),1), round(data_new['partPopBacSupEt'].mean(),1)]

r1=np.arange(len(bars1)) 
r2=[x+ barWidth for x in r1]
r3=[x+ barWidth for x in r2]

plt.bar(r1, bars1, color='tab:orange', width=barWidth,edgecolor='white', label='population')
plt.bar(r2, bars2, color='tab:green', width=barWidth, edgecolor='white',label='women')
plt.bar(r3, bars3, color='tab:cyan', width=barWidth,edgecolor='white', label='foreigners')


plt.ylabel('Percentages')
plt.xlabel('Diploma levels')
plt.title('Percentage of population, women and foreigners with different diploma levels', fontweight='bold')
plt.xticks([r+barWidth for r in range(len(bars1))],["inferior to high school diploma", "equal to a high school diploma", "superior to or equal to a bachelor's degree"], rotation=10)

plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')  
plt.savefig('wholefrdiplomasgroupedbar', bbox_inches="tight" )
plt.show()
                                                     
x=["population without diploma", "women in the population without diploma", "foreigners without diploma"]
y=[round(data_new['partPopSansDip'].mean(),1), round(data_new['partPopSansDipF'].mean(),1), round(data_new['partPopSansDipEt'].mean(),1)]
plt.figure(figsize=(4,4))
plt.bar(x,y, color=['tab:cyan', 'tab:olive', 'tab:blue'], alpha=0.7)
plt.xticks(x,rotation=60)
plt.ylabel('Percentage')
plt.title('Percentage of the population, women and foreigners without a diploma', fontweight='bold')

plt.savefig('Wholefrancebezdiplome', bbox_inches='tight')
plt.show()
                                                     
x=["rate for 15-24 yo","rate for women between 15-24 yo", "rate for men between 15-24 yo", "rate for foreigners between 15-24 yo"]
y=[round(data_new['txScol15_24'].mean(),1), round(data_new['txScol15_24F'].mean(),1), round(data_new['txScol15_24H'].mean(),1), round(data_new['txScol15_24Et'].mean(),1)]
plt.figure(figsize=(5,5))
plt.bar(x,y, color=['tab:blue', 'tab:green', 'tab:orange', 'tab:purple'], alpha=0.5)
plt.xticks(x,rotation=10)
plt.ylabel('School enrollment rate')
plt.title('School enrollment rate for 15-24 years old, women, men and foreigners between 15-24 years old', fontweight='bold')
plt.savefig('Wholefranceschoolenrate', bbox_inches='tight')
plt.show()
                                                     
x=[" Female youth index", "Male youth index", " Youth index for foreigners"]
y=[round(data_new['Female youth index'].mean(),1), round(data_new['Male youth index'].mean(),1), round(data_new['Youth index of foreigners'].mean(),1)]

plt.figure(figsize=(4,4))
plt.bar(x,y, color='tab:purple', alpha=0.5)
plt.xticks(x,rotation=9)
plt.ylabel('Youth index')
plt.title('Female, male and youth index for foreigners', fontweight="bold")
plt.savefig('Wholefranceyouthindex', bbox_inches='tight')
plt.show()
                                  
x=["Employment rate for women", "Employment rate for men", "Employment rate of foreigners"]
y=[round(data_new['Employment rate F'].mean(),1), round(data_new['Employment rate M'].mean(),1), round(data_new['Employment rate Et'].mean(),1)]
plt.figure(figsize=(4,4))
plt.bar(x,y, color='tab:orange', alpha=0.5)
plt.xticks(x,rotation=9)
plt.ylabel('Employment rate')
plt.title('Employment rate for women, men and foreigners',fontweight='bold',ha='center')
plt.savefig('wholefranceemploymratepolovi', bbox_inches='tight')
plt.show()
                           
x=["Employment rate for women", "Employment rate for men", "Employment rate of foreigners"]
y=[round(data_new['Employment rate F'].mean(),1), round(data_new['Employment rate M'].mean(),1), round(data_new['Employment rate Et'].mean(),1)]
plt.figure(figsize=(4,4))
plt.bar(x,y, color='tab:orange', alpha=0.5)
plt.xticks(x,rotation=9)
plt.ylabel('Employment rate')
plt.title('Employment rate for women, men and foreigners',fontweight='bold',ha='center')
plt.savefig('wholefranceemploymratepolovi', bbox_inches='tight')
plt.show()
                                
x=["Employment rate of 15-24 yo", "Employment rate of 25-54 yo ", "Employment rate of 55-64 yo"]
y=[round(data_new['Er15_24'].mean(),1), round(data_new['Er25_54'].mean(),1), round(data_new['Er55_64'].mean(),1)]
plt.figure(figsize=(4,4))
plt.bar(x,y, color='tab:olive', alpha=0.7)
plt.xticks(x,rotation=15)
plt.ylabel('Employment rate')
plt.title('Employment rate among ages',fontweight='bold', ha='center')
plt.savefig('Wholefranceemploymrategodine', bbox_inches='tight')
plt.show()
                                                     
x=["% of low paying jobs among jobs", "% of low paying jobs among jobs done by women ", "% of low paying jobs among jobs done by foreigners"]
y=[round(data_new['partEmpPrec'].mean(),1), round(data_new['partEmpPrecF'].mean(),1), round(data_new['partEmpPrecEt'].mean(),1)]
plt.figure(figsize=(4,4))
plt.bar(x,y, color='tab:cyan', alpha=0.7)
plt.xticks(x,rotation=13)
plt.ylabel('Percentage')
plt.title('Percentage of low paying jobs among jobs, jobs done by women and jobs done by foreigners', fontweight='bold')
plt.savefig('wholefrancelowpayjobs', bbox_inches='tight')
plt.show()
                                                     



                                                     
                                                     
                                       





                                                     
                                                     
                                                     






