import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly as pl
from pymongo import MongoClient

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_percentage_error

from django.shortcuts import render, redirect
from django.contrib.auth.hashers import make_password, check_password
from .forms import RegisterForm

client = MongoClient('mongodb://localhost:27017/')
db = client['prediction_db']
prediction_collection = db['predictions']
users_collection = db['users']

def register(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            firstname = form.cleaned_data['firstname']
            lastname = form.cleaned_data['lastname']
            email = form.cleaned_data['email']
            password = form.cleaned_data['password']

            # Check if user already exists by email
            if users_collection.find_one({'email': email}):
                return render(request, 'auth.html', {'form': form, 'error': 'Email already exists'})

            # Hash the password before storing
            hashed_password = make_password(password)

            # Insert into MongoDB
            user_data = {
                'firstname': firstname,
                'lastname': lastname,
                'email': email,
                'password': hashed_password
            }
            res = users_collection.insert_one(user_data)
            print(res)
            # Store user in session
            request.session['email'] = email
            return redirect('index')

    else:
        form = RegisterForm()

    return render(request, 'auth.html', {'form': form})


def user_login(request):
    if request.method == 'POST':
        email = request.POST['email']
        password = request.POST['password']

        user = users_collection.find_one({'email': email})

        if user and check_password(password, user['password']):
            # Store user in session
            request.session['email'] = email
            return redirect('index')

        return render(request, 'auth.html', {'error': 'Invalid email or password'})

    return render(request, 'auth.html')


def user_logout(request):
    request.session.flush()  # Clears session
    return redirect('login')

def index(request):
    return render(request, 'index.html')

def upload_file(request):
    global cat_features, encoding_map, num_vars, scaler, model
    if request.method == 'POST' and request.FILES.get('dataset'):
        dataset = request.FILES['dataset']
        fs = FileSystemStorage()
        filename = fs.save(dataset.name, dataset)
        file_path = fs.path(filename)

        # Load dataset
        try:
            data = pd.read_csv(file_path)
        except Exception as e:
            return render(request, 'index.html', {'error': f"Error loading file: {e}"})

        data.drop_duplicates(inplace=True)

        # Handling missing values
        data.fillna(data.select_dtypes(include=[np.number]).mean(), inplace=True)

        # Encoding categorical features
        cat_features = ['Gender', 'Marital_Status', 'Homeownership_Status', 'Education_Level',
                         'Occupation', 'Location', 'Employment_Status', 'Type_of_Housing',
                         'Primary_Mode_of_Transportation']

        encoding_map = {
            'Male': 1, "Female": 0, "Married": 1, "Single": 0, "Divorced": 2,
            "Own": 1, "Rent": 0, "Master's": 1, "High School": 0, "Bachelor's": 2,
            "Doctorate": 3, "Education": 1, "Finance": 0, "Healthcare": 2,
            "Others": 3, "Technology": 4, "Rural": 1, "Suburban": 0, "Urban": 2,
            "Full-time": 1, "Self-employed": 0, "Part-time": 2, "Apartment": 1,
            "Single-family home": 0, "Townhouse": 2, "Public transit": 1, "Biking": 0,
            "Car": 2, "Walking": 3
        }

        data[cat_features] = data[cat_features].apply(lambda x: x.map(encoding_map).fillna(0))

        # Train-test split
        df_train, df_test = train_test_split(data, train_size=0.8, random_state=0)

        # Normalize numeric data
        scaler = MinMaxScaler()
        num_vars = df_train.select_dtypes(include=[np.number]).columns.tolist()
        num_vars.remove('Income')

        df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
        df_test[num_vars] = scaler.transform(df_test[num_vars])

        y_train = df_train.pop('Income')
        x_train = df_train
        y_test = df_test.pop('Income')
        x_test = df_test

        # Train model
        model = LinearRegression()
        model.fit(x_train, y_train)

        # Predictions and evaluation
        y_test_pred = model.predict(x_test)
        r2 = r2_score(y_test, y_test_pred)
        mape = mean_absolute_percentage_error(y_test, y_test_pred)

        # Generate interactive plot with Plotly
        fig = px.scatter(x=y_test, y=y_test_pred, labels={'x': 'Actual Income', 'y': 'Predicted Income'},
                          title='Actual vs Predicted Income')
        plot_json = json.dumps(fig, cls=pl.utils.PlotlyJSONEncoder)

        return render(request, 'results.html', {
            'file_url': fs.url(filename),
            'r2_score': round(r2, 4),
            'mape': round(mape, 4),
            'plot_data': plot_json
        })

    return render(request, 'index.html')

def predict_income(request):
    if request.method == 'POST':
        try:
            input_data = {
                'Age': float(request.POST['Age']),
                'Education_Level': request.POST['Education_Level'],
                'Occupation': request.POST['Occupation'],
                'Number_of_Dependents': float(request.POST['Number_of_Dependents']),
                'Location': request.POST['Location'],
                'Work_Experience': float(request.POST['Work_Experience']),
                'Marital_Status': request.POST['Marital_Status'],
                'Employment_Status': request.POST['Employment_Status'],
                'Household_Size': float(request.POST['Household_Size']),
                'Homeownership_Status': request.POST['Homeownership_Status'],
                'Type_of_Housing': request.POST['Type_of_Housing'],
                'Gender': request.POST['Gender'],
                'Primary_Mode_of_Transportation': request.POST['Primary_Mode_of_Transportation'],
            }
            input_df = pd.DataFrame([input_data])

            for col in cat_features:
                input_df[col] = input_df[col].map(encoding_map).fillna(0)
            input_df[num_vars] = scaler.transform(input_df[num_vars])

            predicted_income = model.predict(input_df)[0]
            input_data['Predicted_Income'] = round(predicted_income, 2)

            # Store in MongoDB
            prediction_collection.insert_one(input_data)

            return render(request, 'results.html', {'predicted_income': round(predicted_income, 2)})
        except Exception as e:
            return render(request, 'results.html', {'error': f"Prediction Error: {e}"})
    
    return render(request, 'results.html')

def prediction_history(request):
    history = list(prediction_collection.find({}, {'_id': 0}))
    return render(request, 'history.html', {'history': history})

