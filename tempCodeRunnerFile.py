import re
from tkinter import ON
from flask import Flask,render_template,request,redirect
import pandas as pd
import plotly
import plotly.express as px
from flask import jsonify
import numpy as np
from joblib import load
import warnings
warnings.filterwarnings("ignore")



df= pd.read_csv("seattle_data_final.csv")

Room_Type_List = sorted(list(df.room_type.unique()))[1:]
Label_List = sorted(list(df.Label.unique()))[1:]

item = Room_Type_List + Label_List

bathroom_label= sorted(df.bathrooms.unique())

amenity_list=['Kitchen','Iron','Carbon_Monoxide_Detector','Suitable_for_Events','Smoke_Detector',
 'Washer','Pets_Allowed','Shampoo','Air_Conditioning','Hot_Tub','Wireless_Internet', 'TV','Family/Kid_Friendly',
 'Cable_TV','24-Hour_Check-in', 'Gym','Laptop_Friendly_Workspace','Essentials','Hair_Dryer','Internet',
 'Pets_live_on_this_property','Cat(s)','Wheelchair_Accessible','Washer_/_Dryer','Indoor_Fireplace',
 'Lock_on_Bedroom_Door','Dryer','Dog(s)','Elevator_in_Building','First_Aid_Kit','Free_Parking_on_Premises',
 'Heating','Smoking_Allowed','Other_pet(s)','Breakfast','Doorman','Buzzer/Wireless_Intercom',
 'Fire_Extinguisher','Safety_Card','Pool','Hangers']

rf= load("app/property_classification.joblib")

def reset():
    input_dict = {"bathrooms":0,"security_deposit":0,"Total_Ammenities":0,"Total_Cost":0,"Total_Capacity":0}
    for i in item:
        input_dict[i]=0
    return (input_dict)

def recomender(bathrooms,security_deposit,Total_Ammenities,Total_Cost,Total_Capacity,Room_Type,Host_type):
    
    input_vector = reset()
    input_vector["bathrooms"] = bathrooms
    input_vector["security_deposit"]=security_deposit
    input_vector["Total_Ammenities"]=Total_Ammenities
    input_vector["Total_Cost"]=np.log(Total_Cost)
    input_vector["Total_Capacity"]=Total_Capacity
    if Room_Type in input_vector:
        input_vector[Room_Type]=1
    if (int(Host_type[-1])) in input_vector.keys():   
        input_vector[int(Host_type[-1])]=1

    input_list= list(input_vector.values())
    prediction = rf.predict([input_list])


    return prediction


def df_sorting_level1 (c,bedrooms,neighbourhood,accomodate,amenity):
    newdf = df[(df.property_type == c[0]) & (df.bedrooms==bedrooms) & (df.neighbourhood_group_cleansed==neighbourhood) & ((df.accommodates + df.guests_included) >= accomodate )]

    index= newdf.index
    newdf["matched_amenity_count"]=np.nan
    index1= []
    for i in index:
        am= df.loc[i,"amenities"]
        c=0
        for  j in amenity:
            if j in am:
                c=c+1
        newdf.loc[i,"matched_amenity_count"] = c
    
    return newdf




def df_sorting_level2 (c,bedrooms,neighbourhood,accomodate,amenity):
    newdf = df[(df.property_type == c[0]) & (df.bedrooms==bedrooms) & (df.neighbourhood_group_cleansed==neighbourhood) ]

    index= newdf.index
    newdf["matched_amenity_count"]=np.nan
    index1= []
    for i in index:
        am= df.loc[i,"amenities"]
        c=0
        for  j in amenity:
            if j in am:
                c=c+1
        newdf.loc[i,"matched_amenity_count"] = c
    
    return newdf



def df_sorting_level3 (c,bedrooms,neighbourhood,accomodate,amenity):
    newdf = df[(df.property_type == c[0]) & (df.bedrooms==bedrooms)]

    index= newdf.index
    newdf["matched_amenity_count"]=np.nan
    index1= []
    for i in index:
        am= df.loc[i,"amenities"]
        c=0
        for  j in amenity:
            if j in am:
                c=c+1
        newdf.loc[i,"matched_amenity_count"] = c
    
    return newdf




app = Flask(__name__)  
@app.route('/')  
def message():  
      return render_template("index.html",data=df, amenity = amenity_list )


@app.route('/predict',methods=['POST'])

def predict():
    
    init_features = [x for x in request.form.values()][:-1]
    multiselect = request.form.getlist('mymultiselect')

    neighbourhood = init_features[0]
    bedrooms = int( init_features[1])
    bathrooms= int(init_features[2])
    room_type = init_features[3]
    price = int(init_features[4])
    security_deposit = int(init_features[5])
    accomodates = int(init_features[6])
    label =init_features[-1]
    amenities= len(multiselect)


    c= recomender(bathrooms,security_deposit,amenities,price,accomodates,room_type,label)

    if c==1:
        p= "Independent House"
    else: 
        p = "Apartment"
    
    # newdf = df[(df.property_type == c[0]) & (df.bedrooms==bedrooms) & (df.neighbourhood_group_cleansed==neighbourhood) & ((df.accommodates + df.guests_included) >= accomodates )].head().reset_index()


    
    def making_new_df ():

        count=5

        newdf = df_sorting_level1(c,bedrooms,neighbourhood,accomodates,multiselect)
        newdf = newdf[newdf.matched_amenity_count >= len(multiselect)//2]
        if len(newdf) < 5 : 
            newdf = df_sorting_level2(c,bedrooms,neighbourhood,accomodates,multiselect)
            newdf = newdf[newdf.matched_amenity_count >= len(multiselect)//2]
            if len(newdf) < 5 : 
                newdf = df_sorting_level3(c,bedrooms,neighbourhood,accomodates,multiselect)
                newdf= newdf[newdf.matched_amenity_count >= len(multiselect)//2]
                count = len(newdf)

        return newdf,count



    newdf , count = making_new_df ()
    newdf=newdf.head().reset_index()



    return render_template("predict.html",data=df, amenity = amenity_list ,val= p,new=newdf )


if __name__ == '__main__':  
   app.run()  
