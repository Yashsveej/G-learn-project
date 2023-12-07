
import requests
url ='http://127.0.0.1:5000/predict'

input_data= {'features': [5.1,3.5,1.4,0.2]}

#send a POST request to the flask server 
response = requests.post(url, json= input_data)

#check the response
if response.status_code == 200:
    result = response.json()
    print(f"prediction: {result['prediction']}")
else:
    print(f"Error:{response.text}")


