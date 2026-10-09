from pathlib import Path
import numpy as np, pandas as pd
rng=np.random.default_rng(42); out=Path(__file__).resolve().parent
n=900; dates=pd.date_range('2024-01-01',periods=n,freq='D')

def save(domain,df): df.to_csv(out/f'{domain}.csv',index=False)
# company
products=np.array(['Alpha','Beta','Gamma','Delta','Epsilon']); regions=np.array(['North','South','East','West'])
t=np.arange(n); demand=np.maximum(20,130+0.08*t+22*np.sin(2*np.pi*t/30)+rng.normal(0,14,n)); price=rng.uniform(40,130,n); cost=price*rng.uniform(.45,.72,n); units=np.maximum(5,demand+rng.normal(0,10,n)); sales=units*price; profit=(price-cost)*units-rng.uniform(500,4000,n); inv=np.maximum(5,demand*rng.uniform(.8,1.7,n)); risk=((inv<demand*0.8)|(profit<profit.mean()*.25)).astype(int)
save('company',pd.DataFrame({'Date':dates,'Product':rng.choice(products,n),'Region':rng.choice(regions,n),'Sales':sales.round(2),'Demand':demand.round(2),'Inventory':inv.round(2),'Customers':np.maximum(10,(units*rng.uniform(.3,.8,n)).astype(int)),'Cost':(cost*units).round(2),'Profit':profit.round(2),'Risk_Flag':risk}))
# hospital
for domain, dept, patient_base, extra in [
('hospital',['Cardiology','Neurology','Orthopedics','Pediatrics'],100, 'hospital'),
('college',['Computer','Management','Commerce','Arts'],500,'college'),
('municipality',['North','South','Central','East'],300,'municipality'),
('ngo',['North','South','Central','East'],220,'ngo'),
('community',['Area A','Area B','Area C','Area D'],140,'community'),
('event_organizer',['Tech Summit','Sports Meet','Cultural Fest','Career Fair'],700,'event')]:
    cat=rng.choice(dept,n); trend=np.arange(n)
    if extra=='hospital':
        patients=np.maximum(20,patient_base+20*np.sin(2*np.pi*trend/21)+rng.normal(0,15,n)+trend*.03); avail=np.maximum(40,patients+rng.normal(45,18,n)); occ=np.minimum(avail-2,patients*rng.uniform(.75,1.05,n)); wait=np.maximum(5,12+patients*.08+rng.normal(0,5,n)); staff=np.maximum(5,(avail/18+rng.normal(0,2,n)).astype(int)); risk=(occ/avail>.92).astype(int); df=pd.DataFrame({'Date':dates,'Department':cat,'Patients':patients.round(1),'Beds_Available':avail.round(1),'Beds_Occupied':occ.round(1),'Waiting_Time':wait.round(1),'Staff_Count':staff,'Capacity_Risk':risk})
    elif extra=='college':
        students=np.maximum(100,rng.normal(520,80,n)); attendance=np.clip(78+6*np.sin(2*np.pi*trend/45)+rng.normal(0,6,n),45,99); faculty=np.maximum(10,(students/25+rng.normal(0,4,n)).astype(int)); util=np.clip(attendance/100*.9+rng.normal(0,.08,n),.25,1); risk=(attendance<65).astype(int); df=pd.DataFrame({'Date':dates,'Department':cat,'Student_Count':students.astype(int),'Attendance':attendance.round(2),'Faculty_Count':faculty,'Classroom_Utilization':util.round(3),'Attendance_Risk':risk})
    elif extra=='municipality':
        pop=rng.uniform(0.7,1.6,n); waste=np.maximum(30,patient_base*pop+25*np.sin(2*np.pi*trend/30)+rng.normal(0,35,n)); complaints=np.maximum(5,waste*.08+rng.normal(0,15,n)); workers=np.maximum(5,(waste/45+rng.normal(0,2,n)).astype(int)); risk=(complaints>np.quantile(complaints,.8)).astype(int); df=pd.DataFrame({'Date':dates,'Zone':cat,'Population_Index':pop.round(2),'Waste_Collected':waste.round(2),'Complaints':complaints.round(2),'Workers':workers,'Service_Risk':risk})
    elif extra=='ngo':
        beneficiaries=np.maximum(20,patient_base+20*np.sin(2*np.pi*trend/35)+rng.normal(0,25,n)); donations=np.maximum(1000,beneficiaries*rng.uniform(25,70,n)); aid=donations*rng.uniform(.55,.9,n); volunteers=np.maximum(3,(beneficiaries/18+rng.normal(0,4,n)).astype(int)); risk=(volunteers<beneficiaries/25).astype(int); df=pd.DataFrame({'Date':dates,'District':cat,'Beneficiaries':beneficiaries.round(1),'Donations':donations.round(2),'Aid_Distributed':aid.round(2),'Volunteers':volunteers,'Coverage_Risk':risk})
    elif extra=='community':
        pop=rng.uniform(.5,1.8,n); req=np.maximum(5,patient_base*pop+25*np.sin(2*np.pi*trend/28)+rng.normal(0,20,n)); part=np.clip(45+12*np.sin(2*np.pi*trend/60)+rng.normal(0,8,n),5,95); shortage=np.maximum(0,req-rng.uniform(70,150,n)); risk=(shortage>np.quantile(shortage,.75)).astype(int); df=pd.DataFrame({'Date':dates,'Area':cat,'Population_Index':pop.round(2),'Resource_Requests':req.round(2),'Participation':part.round(2),'Shortage_Count':shortage.round(2),'Shortage_Risk':risk})
    else:
        capacity=np.maximum(300,rng.normal(1000,150,n)); attendance=np.minimum(capacity*.98,np.maximum(100,capacity*.72+90*np.sin(2*np.pi*trend/40)+rng.normal(0,80,n))); regs=np.maximum(attendance,attendance*rng.uniform(1,1.35,n)); revenue=np.maximum(1000,attendance*rng.uniform(15,80,n)); resources=np.maximum(10,attendance/45+rng.normal(0,4,n)); risk=(attendance/capacity>.9).astype(int); df=pd.DataFrame({'Date':dates,'Event':cat,'Attendance':attendance.round(1),'Capacity':capacity.round(1),'Registrations':regs.round(1),'Revenue':revenue.round(2),'Resources_Used':resources.round(1),'Capacity_Risk':risk})
    save(domain,df)
print('Generated seven domain datasets.')
