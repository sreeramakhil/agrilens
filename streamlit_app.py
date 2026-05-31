# 1. Streamlit config (MUST BE FIRST)
import streamlit as st
st.set_page_config(
    page_title="AgriLens",
    layout="wide",
    page_icon="🌱",
    initial_sidebar_state="expanded"
)

# 2. Environment config
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 3. Other imports
import numpy as np
import joblib
import requests
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from fpdf import FPDF
from PIL import Image
import base64
import tempfile
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import re # Import regex module for string manipulation

# Disease details mapping
# Keys are standardized to match the output of the new normalize_key_for_details function
# (e.g., "Crop_Disease_Name" with no parentheses or multiple underscores)
DISEASE_DETAILS = {
    "Apple_Scab": {
        "growth_stage": "Early Spring",
        "cause": "Fungal spores in cool, wet weather",
        "nutrient_deficiency": "Iron, Boron, Manganese, Zinc deficiency",
        "solution": "Prune tree; apply fungicides",
        "fertilizer": "Micronutrient mix (Fe, B, Mn, Zn)"
    },
    "Apple_Black_Rot": {
        "growth_stage": "Late Spring",
        "cause": "Fungal infection, warm/humid conditions",
        "nutrient_deficiency": "Potassium, Calcium deficiency",
        "solution": "Remove infected fruit; use fungicides",
        "fertilizer": "Potassium-rich fertilizer"
    },
    "Apple_Cedar_Rust": {
        "growth_stage": "Spring",
        "cause": "Fungal spores from nearby junipers",
        "nutrient_deficiency": "Magnesium, Sulfur deficiency",
        "solution": "Remove nearby junipers; apply fungicides",
        "fertilizer": "Magnesium sulfate (Epsom salt)"
    },
    "Apple_healthy": {
        "growth_stage": "-",
        "cause": "-",
        "nutrient_deficiency": "-",
        "solution": "Maintain regular care",
        "fertilizer": "Balanced NPK fertilizer"
    },
    "Corn_Maize_Common_Rust": { # Standardized key for Corn Common Rust
        "growth_stage": "Mid-Summer",
        "cause": "Fungal spores in warm, humid weather",
        "nutrient_deficiency": "Nitrogen, Phosphorus deficiency",
        "solution": "Apply fungicides; crop rotation",
        "fertilizer": "High-nitrogen fertilizer"
    },
    "Corn_Maize_Gray_Leaf_Spot": { # Standardized key for Corn Gray Leaf Spot
        "growth_stage": "Late Summer",
        "cause": "Fungal spores in warm, humid weather",
        "nutrient_deficiency": "Potassium, Magnesium deficiency",
        "solution": "Remove infected leaves; apply fungicides",
        "fertilizer": "Potassium-rich fertilizer"
    },
    "Corn_Maize_Northern_Leaf_Blight": { # Standardized key for Corn Northern Leaf Blight
        "growth_stage": "Mid-Summer",
        "cause": "Fungal spores in warm, humid weather",
        "nutrient_deficiency": "Zinc, Manganese deficiency",
        "solution": "Crop rotation; apply fungicides",
        "fertilizer": "Micronutrient mix (Zn, Mn)"
    },
    "Corn_Maize_healthy": { # Standardized key for Corn healthy
        "growth_stage": "-",
        "cause": "-",
        "nutrient_deficiency": "-",
        "solution": "Maintain regular care",
        "fertilizer": "Balanced NPK fertilizer"
    },
    "Corn_Maize_Cercospora_Leaf_Spot": {
        "growth_stage": "Mid-Summer",
        "cause": "Fungal spores in warm, humid weather",
        "nutrient_deficiency": "Nitrogen, Potassium deficiency",
        "solution": "Remove infected leaves; apply fungicides",
        "fertilizer": "High-nitrogen fertilizer"
    },
    "Corn_Maize_Cercospora_leaf_spot_Gray_leaf_spot": {
        "growth_stage": "Mid to Late Summer",
        "cause": "Fungal spores (Cercospora and/or Gray Leaf Spot) in warm, humid weather",
        "nutrient_deficiency": "Nitrogen, Potassium, and Magnesium deficiency",
        "solution": "Remove infected leaves, apply fungicides, and practice crop rotation",
        "fertilizer": "Balanced fertilizer with sufficient N, K, and Magnesium"
    }
}

# Weather API - Replace with your actual API key
API_KEY = "502d8628d859f86e0af77481841f9b6f" 

# Custom CSS for styling
def local_css(file_name):
    """Loads a local CSS file and applies it to the Streamlit app."""
    try:
        with open(file_name) as f:
            st.html(f'<style>{f.read()}</style>')
    except FileNotFoundError:
        # If CSS file doesn't exist, continue without it
        pass

# Load custom CSS (assuming 'style.css' exists in the same directory)
local_css("style.css")

def get_current_weather(location):
    """Fetches current weather data for a given location."""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        
        return {
            "location": data['name'],
            "country": data['sys']['country'],
            "temperature": data['main']['temp'],
            "feels_like": data['main']['feels_like'],
            "humidity": data['main']['humidity'],
            "pressure": data['main']['pressure'],
            "visibility": data.get('visibility', 0) / 1000,  # Convert to km
            "wind_speed": data['wind']['speed'],
            "wind_direction": data['wind'].get('deg', 0),
            "weather_main": data['weather'][0]['main'],
            "weather_description": data['weather'][0]['description'],
            "weather_icon": data['weather'][0]['icon'],
            "clouds": data['clouds']['all'],
            "sunrise": datetime.fromtimestamp(data['sys']['sunrise']),
            "sunset": datetime.fromtimestamp(data['sys']['sunset']),
            "timestamp": datetime.now()
        }
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching current weather data: {e}. Please check the location or your internet connection.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching current weather data: {e}")
        return None

def get_forecast_weather(location):
    """Fetches 5-day weather forecast data for a given location."""
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={location}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        
        forecast_data = []
        for item in data['list']:
            forecast_data.append({
                'datetime': datetime.fromtimestamp(item['dt']),
                'temperature': item['main']['temp'],
                'humidity': item['main']['humidity'],
                'pressure': item['main']['pressure'],
                'weather': item['weather'][0]['description'],
                'weather_icon': item['weather'][0]['icon'],
                'wind_speed': item['wind']['speed'],
                'clouds': item['clouds']['all'],
                'rain': item.get('rain', {}).get('3h', 0) # Get rain volume in last 3 hours
            })
        
        return forecast_data
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching forecast data: {e}. Please check the location or your internet connection.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching forecast data: {e}")
        return None

def get_weather_report(location):
    """
    Fetches a simplified weather report for agricultural recommendations.
    This function is kept for compatibility with existing calls.
    """
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={location}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        rains = []
        # Check for rain in the next 24 hours (8 * 3-hour forecasts)
        for i in data["list"][:8]:
            desc = i["weather"][0]["description"]
            if "rain" in desc.lower():
                rains.append(i["dt_txt"])

        forecast = {
            "location": location,
            "next_24h_rain": len(rains) > 0,
            "rain_times": rains,
            "temperature": f"{data['list'][0]['main']['temp']} °C",
            "humidity": f"{data['list'][0]['main']['humidity']}%",
            "weather_icon": data['list'][0]['weather'][0]['icon']
        }

        forecast["advice"] = (
            "🌧️ Rain expected — watch for fungal issues and plan irrigation accordingly!"
            if forecast["next_24h_rain"]
            else "☀️ Dry weather — monitor irrigation needs and conserve water."
        )
        return forecast
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching weather report: {e}. Please check the location or your internet connection.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching weather report: {e}")
        return None

def create_temperature_chart(forecast_data):
    """Generates a Plotly chart for temperature trends over 5 days."""
    df = pd.DataFrame(forecast_data)
    df['date'] = df['datetime'].dt.strftime('%m/%d %H:%M')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['temperature'],
        mode='lines+markers',
        name='Temperature',
        line=dict(color='#ff6b6b', width=3),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="Temperature Forecast (5 Days)",
        xaxis_title="Date & Time",
        yaxis_title="Temperature (°C)",
        template="plotly_white",
        height=400
    )
    
    return fig

def create_humidity_pressure_chart(forecast_data):
    """Generates a Plotly chart for humidity and pressure trends."""
    df = pd.DataFrame(forecast_data)
    df['date'] = df['datetime'].dt.strftime('%m/%d %H:%M')
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Humidity (%)', 'Pressure (hPa)'),
        vertical_spacing=0.1
    )
    
    # Humidity
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['humidity'], 
                  mode='lines+markers', name='Humidity',
                  line=dict(color='#4ecdc4', width=2)),
        row=1, col=1
    )
    
    # Pressure
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['pressure'], 
                  mode='lines+markers', name='Pressure',
                  line=dict(color='#45b7d1', width=2)),
        row=2, col=1
    )
    
    fig.update_layout(height=500, template="plotly_white")
    fig.update_xaxes(title_text="Date & Time", row=2, col=1)
    
    return fig

def create_weather_summary_chart(forecast_data):
    """Generates a Plotly pie chart summarizing weather conditions."""
    df = pd.DataFrame(forecast_data)
    weather_counts = df['weather'].value_counts()
    
    fig = px.pie(
        values=weather_counts.values,
        names=weather_counts.index,
        title="Weather Conditions Distribution (Next 5 Days)"
    )
    
    fig.update_layout(height=400)
    return fig

def display_weather_dashboard():
    """Displays the interactive weather dashboard page."""
    st.header("🌦️ Live Weather Dashboard")
    st.markdown("Real-time weather monitoring and forecast for agricultural planning")
    
    # Location input
    col1, col2 = st.columns([3, 1])
    with col1:
        location = st.text_input(
            "📍 Enter Location (City, Country)",
            value="Bangalore, India",
            help="Enter city name and country for accurate weather data"
        )
    with col2:
        if st.button("🔄 Refresh Data", type="primary"):
            st.experimental_rerun() # Rerun the app to fetch fresh data
    
    if location:
        # Get current and forecast weather
        current_weather = get_current_weather(location)
        forecast_data = get_forecast_weather(location)
        
        if current_weather and forecast_data:
            # Current Weather Section
            st.subheader("Current Weather Conditions")
            
            # Current weather cards for key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.html(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           padding: 20px; border-radius: 15px; color: white; text-align: center;">
                    <h3>{current_weather['temperature']:.1f}°C</h3>
                    <p>Temperature</p>
                    <small>Feels like {current_weather['feels_like']:.1f}°C</small>
                </div>
                """)
            
            with col2:
                st.html(f"""
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                           padding: 20px; border-radius: 15px; color: white; text-align: center;">
                    <h3>{current_weather['humidity']}%</h3>
                    <p>Humidity</p>
                    <small>{current_weather['pressure']} hPa</small>
                </div>
                """)
            
            with col3:
                st.html(f"""
                <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                           padding: 20px; border-radius: 15px; color: white; text-align: center;">
                    <h3>{current_weather['wind_speed']:.1f} m/s</h3>
                    <p>Wind Speed</p>
                    <small>{current_weather['visibility']:.1f} km visibility</small>
                </div>
                """)
            
            with col4:
                st.html(f"""
                <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                           padding: 20px; border-radius: 15px; color: white; text-align: center;">
                    <h3>{current_weather['weather_main']}</h3>
                    <p>{current_weather['weather_description'].title()}</p>
                    <img src="http://openweathermap.org/img/wn/{current_weather['weather_icon']}@2x.png" width="50">
                </div>
                """)
            
            st.markdown("---")
            
            # Detailed current conditions in two columns
            st.subheader("Detailed Current Conditions")
            col1, col2 = st.columns(2)
            
            with col1:
                st.html(f"""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px;">
                    <h4>🌡️ Temperature Details</h4>
                    <p><strong>Current:</strong> {current_weather['temperature']:.1f}°C</p>
                    <p><strong>Feels Like:</strong> {current_weather['feels_like']:.1f}°C</p>
                    <p><strong>Humidity:</strong> {current_weather['humidity']}%</p>
                    <p><strong>Pressure:</strong> {current_weather['pressure']} hPa</p>
                </div>
                """)
            
            with col2:
                st.html(f"""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px;">
                    <h4>💨 Wind & Visibility</h4>
                    <p><strong>Wind Speed:</strong> {current_weather['wind_speed']:.1f} m/s</p>
                    <p><strong>Wind Direction:</strong> {current_weather['wind_direction']}°</p>
                    <p><strong>Visibility:</strong> {current_weather['visibility']:.1f} km</p>
                    <p><strong>Cloud Cover:</strong> {current_weather['clouds']}%</p>
                </div>
                """)
            
            # Sunrise and Sunset times
            st.html(f"""
            <div style="background: linear-gradient(90deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%); 
                       padding: 15px; border-radius: 10px; margin: 20px 0;">
                <h4>🌅 Sun Times</h4>
                <div style="display: flex; justify-content: space-around;">
                    <div><strong>Sunrise:</strong> {current_weather['sunrise'].strftime('%H:%M')}</div>
                    <div><strong>Sunset:</strong> {current_weather['sunset'].strftime('%H:%M')}</div>
                </div>
            </div>
            """)
            
            st.markdown("---")
            
            # Forecast Charts Section
            st.subheader("Weather Forecast Charts")
            
            # Temperature chart
            temp_chart = create_temperature_chart(forecast_data)
            st.plotly_chart(temp_chart, use_container_width=True)
            
            # Humidity and Pressure charts side-by-side
            col1, col2 = st.columns(2)
            
            with col1:
                humidity_pressure_chart = create_humidity_pressure_chart(forecast_data)
                st.plotly_chart(humidity_pressure_chart, use_container_width=True)
            
            with col2:
                weather_summary_chart = create_weather_summary_chart(forecast_data)
                st.plotly_chart(weather_summary_chart, use_container_width=True)
            
            st.markdown("---")
            
            # Agricultural Recommendations based on current weather
            st.subheader("🌾 Agricultural Recommendations")
            
            recommendations = []
            
            if current_weather['temperature'] > 30:
                recommendations.append("🌡️ **High Temperature Alert**: Consider providing shade for sensitive crops and increase irrigation frequency to prevent heat stress.")
            elif current_weather['temperature'] < 5:
                recommendations.append("❄️ **Low Temperature Alert**: Protect crops from frost damage. Consider covering sensitive plants or using irrigation for frost protection.")
            
            if current_weather['humidity'] > 80:
                recommendations.append("💧 **High Humidity**: Monitor for fungal diseases. Ensure good air circulation around plants and consider preventative fungicide applications.")
            elif current_weather['humidity'] < 30:
                recommendations.append("🏜️ **Low Humidity**: Increase irrigation and consider mulching to retain soil moisture and reduce evaporation.")
            
            if current_weather['wind_speed'] > 10:
                recommendations.append("💨 **Strong Winds**: Secure tall plants and greenhouses to prevent structural damage. Check for wind burn on leaves regularly.")
            
            # Check for rain in forecast
            rain_forecast = [item for item in forecast_data if item['rain'] > 0]
            if rain_forecast:
                recommendations.append("🌧️ **Rain Expected**: Prepare drainage systems to prevent waterlogging and consider delaying pesticide applications that could be washed away.")
            else:
                recommendations.append("☀️ **Dry Conditions**: Plan irrigation schedule carefully and monitor soil moisture levels closely to avoid drought stress.")
            
            if recommendations:
                st.html("<div style='color:#111; font-size:1.1rem;'>" + "<br>".join(recommendations) + "</div>")
            else:
                st.html("<div style='color:#111; font-size:1.1rem;'>🌱 <b>Optimal Conditions</b>: Current weather conditions are favorable for most agricultural activities. Continue with regular monitoring.</div>")
            # Hourly forecast table
            st.subheader("📊 Detailed Hourly Forecast")
            
            # Create forecast DataFrame for display (next 24 hours)
            df_display = pd.DataFrame(forecast_data[:24]) 
            df_display['Time'] = df_display['datetime'].dt.strftime('%m/%d %H:%M')
            df_display['Temp (°C)'] = df_display['temperature'].round(1)
            df_display['Humidity (%)'] = df_display['humidity']
            df_display['Wind (m/s)'] = df_display['wind_speed'].round(1)
            df_display['Weather'] = df_display['weather'].str.title()
            df_display['Rain (mm)'] = df_display['rain'].round(1)
            
            st.dataframe(
                df_display[['Time', 'Temp (°C)', 'Humidity (%)', 'Wind (m/s)', 'Weather', 'Rain (mm)']],
                use_container_width=True
            )
            
            # Weather alerts section
            st.subheader("⚠️ Weather Alerts")
            alerts = []
            
            # Check for extreme conditions in forecast for the next 24 hours
            max_temp = max([item['temperature'] for item in forecast_data[:24]])
            min_temp = min([item['temperature'] for item in forecast_data[:24]])
            max_wind = max([item['wind_speed'] for item in forecast_data[:24]])
            total_rain = sum([item['rain'] for item in forecast_data[:24]])
            
            if max_temp > 35:
                alerts.append(f"🔥 **Heat Warning**: Maximum temperature expected: {max_temp:.1f}°C. Take precautions to protect crops from extreme heat.")
            if min_temp < 0:
                alerts.append(f"🧊 **Frost Warning**: Minimum temperature expected: {min_temp:.1f}°C. Implement frost protection measures immediately.")
            if max_wind > 15:
                alerts.append(f"💨 **Wind Warning**: Maximum wind speed expected: {max_wind:.1f} m/s. Secure vulnerable structures and plants.")
            if total_rain > 20:
                alerts.append(f"🌧️ **Heavy Rain Warning**: Total rainfall expected: {total_rain:.1f} mm. Ensure proper drainage to prevent waterlogging and root rot.")
            if alerts:
                st.html("<div style='color:#111; font-size:1.1rem;'>" + "<br>".join(alerts) + "</div>")
            else:
                st.html("<div style='color:#111; font-size:1.1rem;'>✅ No significant weather alerts for the next 24 hours.</div>")       

def load_model_and_classes(crop):
    """
    Loads the Keras model and class names for a specific crop disease model.
    """
    model_path = f"models/{crop.lower()}_model.h5"
    class_path = f"data/{crop.lower()}_class_names.npy"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Disease model not found for {crop}: {model_path}")
    if not os.path.exists(class_path):
        raise FileNotFoundError(f"Disease class names not found for {crop}: {class_path}")

    model = load_model(model_path)
    class_names = np.load(class_path, allow_pickle=True)
    return model, class_names

def load_crop_classifier():
    """
    Loads the general crop classifier model and its class names.
    Returns None if the model is not found.
    """
    model_path = "models/crop_classifier_apple_corn_unknown.h5"
    class_path = "data/crop_classifier_classes.npy"

    if not os.path.exists(model_path) or not os.path.exists(class_path):
        return None, None # Return None if model files don't exist
    
    model = load_model(model_path)
    class_names = np.load(class_path, allow_pickle=True)
    return model, class_names

def predict_disease(image_path, model, class_names, selected_crop):
    """
    Performs prediction on an uploaded image using a specific disease model.
    """
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0 # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = round(np.max(predictions[0]) * 100, 2)

    # Define a robust mapping for selected crop to its expected class name prefix
    crop_name_mapping = {
        "apple": "Apple",
        "corn": "Corn_(maize)",
        "grape": "Grape",
        "potato": "Potato",
        "tomato": "Tomato"
    }
    
    expected_prefix = crop_name_mapping.get(selected_crop.lower())

    if not expected_prefix or not predicted_class.startswith(expected_prefix):
        return "Incompatible Image", 0.0

    INTERNAL_CONFIDENCE_THRESHOLD = 60.0

    if confidence < INTERNAL_CONFIDENCE_THRESHOLD:
        return "Incompatible Image", 0.0
    
    return predicted_class, confidence

def predict_crop(image_path, model, class_names):
    """
    Performs prediction on an uploaded image using the general crop classifier model.
    """
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = round(np.max(predictions[0]) * 100, 2)
    
    # Assuming the classes are 'Apple', 'Corn', 'Unknown'
    return predicted_class, confidence

class PDF(FPDF):
    """Custom PDF class for generating reports."""
    def header(self):
        self.set_font("Arial", "B", 16)
        self.set_text_color(34, 139, 34)  # Forest green color
        self.cell(0, 10, "AgriLens Disease Detection Report", ln=True, align="C")
        if os.path.exists("assets/logo.png"):
            self.image("assets/logo.png", 10, 8, 25)
        self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def safe_text(text):
    """Encodes text to latin-1 to prevent PDF generation errors with special characters."""
    return text.encode("latin-1", "replace").decode("latin-1")

def generate_pdf(report_data, image_path, out_path="report.pdf"):
    """Generates a PDF report with analysis results and weather information."""
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Report Info Section
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, txt="Analysis Results", ln=True)
    pdf.set_font("Arial", size=12)
    
    pdf.cell(40, 10, txt="Date:", ln=0)
    pdf.cell(0, 10, txt=safe_text(report_data["date"]), ln=True)
    
    pdf.cell(40, 10, txt="Crop:", ln=0)
    pdf.cell(0, 10, txt=safe_text(report_data["crop"]), ln=True)
    
    pdf.cell(40, 10, txt="Plant Status:", ln=0)
    pdf.cell(0, 10, txt=safe_text(report_data["status"]), ln=True)
    
    pdf.cell(40, 10, txt="Disease:", ln=0)
    pdf.cell(0, 10, txt=safe_text(report_data["disease"]), ln=True)
    
    # ADDED Confidence back to PDF
    pdf.cell(40, 10, txt="Confidence:", ln=0)
    pdf.cell(0, 10, txt=safe_text(report_data["confidence"]), ln=True)
    
    pdf.ln(10)

    # Add uploaded image to PDF
    if os.path.exists(image_path):
        pdf.image(image_path, w=80, h=60)
        pdf.ln(10)

    # Weather section in PDF
    if "weather" in report_data:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, txt="Weather Report", ln=True)
        pdf.set_font("Arial", size=12)
        weather = report_data["weather"]
        
        pdf.cell(40, 10, txt="Location:", ln=0)
        pdf.cell(0, 10, txt=safe_text(weather["location"]), ln=True)
        
        pdf.cell(40, 10, txt="Temperature:", ln=0)
        pdf.cell(0, 10, txt=safe_text(weather["temperature"]), ln=True)
        
        pdf.cell(40, 10, txt="Humidity:", ln=0)
        pdf.cell(0, 10, txt=safe_text(weather["humidity"]), ln=True)
        
        pdf.cell(40, 10, txt="Rain Expected:", ln=0)
        pdf.cell(0, 10, txt="Yes" if weather["next_24h_rain"] else "No", ln=True)
        
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(40, 10, txt="Recommendation:", ln=0)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=safe_text(weather["advice"]))

    pdf.output(out_path)
    return out_path

def get_weather_icon(icon_code):
    """Returns the URL for a weather icon from OpenWeatherMap."""
    return f"http://openweathermap.org/img/wn/{icon_code}@2x.png"

def main():
    """Main function to run the Streamlit application."""
    # Sidebar with logo and navigation
    with st.sidebar:
        if os.path.exists("assets/logo.png"):
            st.image("assets/logo.png", width=150)
        st.title("AgriLens")
        st.markdown("---")
        page = st.radio(
            "Navigate",
            ["🏠 Home", "🔍 Disease Detection", "🌱 Crop Recommendation", "🌦️ Weather Dashboard"],
            label_visibility="collapsed"
        )
        st.markdown("---")
        st.html("""
        <div style="text-align: center;">
            <p>Smart farming solutions for modern agriculture</p>
        </div>
        """)

    # Main content area based on selected page
    if page == "🏠 Home":
        st.header("🌿 Welcome to AgriLens")
        st.html("""
        <div style="text-align: center; padding: 20px; background-color: #f0f8f0; border-radius: 10px;">
            <h3 style="color: #2e8b57;">Your Smart Crop Assistant</h3>
            <p>Empowering farmers with AI-driven agricultural insights</p>
        </div>
        """)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.html("""
            <div style="background-color: #e6f3ff; padding: 15px; border-radius: 10px; height: 200px;">
                <h4 style="color: #1e6fbb;">🌱 Crop Health</h4>
                <p>Detect diseases and nutrient deficiencies from leaf images with our advanced AI models.</p>
            </div>
            """)
        
        with col2:
            st.html("""
            <div style="background-color: #fff2e6; padding: 15px; border-radius: 10px; height: 200px;">
                <h4 style="color: #cc7a00;">🌾 Smart Recommendations</h4>
                <p>Get personalized crop suggestions based on your soil conditions and local weather.</p>
            </div>
            """)
        
        with col3:
            st.html("""
            <div style="background-color: #e6ffe6; padding: 15px; border-radius: 10px; height: 200px;">
                <h4 style="color: #2e8b57;">⛅ Weather Integration</h4>
                <p>Receive weather-aware farming advice to optimize your agricultural practices.</p>
            </div>
            """)
        
        with col4:
            st.html("""
            <div style="background-color: #f0e6ff; padding: 15px; border-radius: 10px; height: 200px;">
                <h4 style="color: #8b2e8b;'>🌦️ Live Weather Dashboard</h4>
                <p>Monitor real-time weather conditions and forecasts for better agricultural planning.</p>
            </div>
            """)
        
        st.markdown("---")

        st.subheader("How It Works")
        
        steps = [
            {"icon": "📷", "title": "Upload Image", "desc": "Take a clear photo of your crop leaves"},
            {"icon": "🔍", "title": "AI Analysis", "desc": "Our system detects diseases and nutrient issues"},
            {"icon": "📊", "title": "Get Report", "desc": "Receive detailed diagnosis and recommendations"},
            {"icon": "🌦️", "title": "Weather Monitoring", "desc": "Track weather conditions for optimal farming decisions"}
        ]
        
        cols = st.columns(4)
        for i, step in enumerate(steps):
            with cols[i]:
                st.html(f"""
                <div style="text-align: center; padding: 15px;">
                    <div style="font-size: 30px; margin-bottom: 10px;">{step['icon']}</div>
                    <h4>{step['title']}</h4>
                    <p>{step['desc']}</p>
                </div>
                """)

    elif page == "🌦️ Weather Dashboard":
        display_weather_dashboard()

    elif page == "🔍 Disease Detection":
        st.header("🔍 Disease & Nutrition Detection")
        st.markdown("Upload an image of your crop leaves to detect diseases or nutrient deficiencies.")
        
        # --- NEW: Load crop classifier model ---
        crop_classifier_model, crop_classifier_classes = load_crop_classifier()

        with st.expander("📌 Instructions", expanded=True):
            if crop_classifier_model:
                st.html("""
                - Upload a clear photo of the plant leaves (Apple or Corn supported)
                - Enter your location for weather-specific advice
                - Our AI will first identify the crop, then analyze it for diseases.
                """)
            else:
                st.markdown("""
                - Select your crop type from the dropdown
                - Enter your location for weather-specific advice
                - Upload a clear photo of the plant leaves
                - Our AI will analyze and provide recommendations
                """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # --- MODIFIED: Conditional Crop Selection ---
            if not crop_classifier_model:
                st.info("Automatic crop classifier not found. Please select a crop manually.")
                crop = st.selectbox(
                    "Select Crop",
                    ["Apple", "Corn", "Grape", "Potato", "Tomato"],
                    help="Choose the crop type you want to analyze"
                )
            else:
                st.success("✅ Automatic crop classifier is active.")
                crop = None # Crop will be determined by the model

            location = st.text_input(
                "📍 Enter Your Location (City, Country)",
                help="This helps us provide weather-specific recommendations"
            )
            
            image_file = st.file_uploader(
                "📤 Upload Leaf Image",
                type=["jpg", "jpeg", "png"],
                help="Upload a clear image of the plant leaves"
            )
        
        with col2:
            if image_file:
                image_file.seek(0)
                img = Image.open(image_file)
                st.image(img, caption="Uploaded Leaf Image", use_column_width=True)
        
        if st.button("Analyze", type="primary", use_container_width=True):
            if image_file and location:
                with st.spinner("Analyzing your crop..."):
                    temp_image_path = None
                    try:
                        image_file.seek(0)
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                            img = Image.open(image_file)
                            if img.mode in ('RGBA', 'LA'):
                                img = img.convert('RGB')
                            img.save(tmp_file.name, 'JPEG')
                            temp_image_path = tmp_file.name
                        
                        # --- NEW: Two-Step Analysis ---
                        # Step 1: Classify the crop if the model is available
                        if crop_classifier_model:
                            st.write("Step 1: Identifying crop type...")
                            predicted_crop, crop_confidence = predict_crop(temp_image_path, crop_classifier_model, crop_classifier_classes)
                            st.write(f"-> Detected Crop: **{predicted_crop}** (Confidence: {crop_confidence}%)")

                            if predicted_crop.lower() == 'unknown':
                                st.error("❌ The uploaded image could not be identified as a supported crop (Apple or Corn). Please upload a different image.")
                                return # Stop analysis
                            
                            # Set the crop for the next step
                            crop = predicted_crop

                        # If crop is still None (manual selection was active but nothing selected)
                        if not crop:
                            st.warning("Please select a crop to analyze.")
                            return

                        # Step 2: Run disease detection on the identified crop
                        st.write(f"Step 2: Analyzing for **{crop}** diseases...")
                        model, class_names = load_model_and_classes(crop)
                        pred_class, confidence = predict_disease(temp_image_path, model, class_names, crop) 
                        
                        def normalize_key_for_details(key):
                            key = key.lower()
                            key = key.replace("(", "").replace(")", "")
                            key = key.replace("___", "_")
                            key = key.replace(" ", "_")
                            key = re.sub(r'_+', '_', key)
                            key = key.strip('_')
                            return key

                        lookup_key = normalize_key_for_details(pred_class)
                        mapping_keys = {normalize_key_for_details(k): k for k in DISEASE_DETAILS.keys()}

                        matched_key = mapping_keys.get(lookup_key)
                        if matched_key:
                            details = DISEASE_DETAILS[matched_key]
                        else:
                            details = {
                                "growth_stage": "Not available",
                                "cause": "Not available",
                                "nutrient_deficiency": "Not available",
                                "solution": "Consult a local agricultural expert.",
                                "fertilizer": "General balanced fertilizer"
                            }
                        
                        forecast = get_weather_report(location)
                        
                        if forecast:
                            status = "Healthy" if "healthy" in pred_class.lower() else "Diseased"

                            # ADDED back Confidence-based messages
                            if status == "Diseased":
                                st.markdown(f"""
                                <div style="background-color: #ffe0b2; padding: 10px; border-radius: 8px; margin-bottom: 15px;">
                                    <strong>🚨 Disease Alert!</strong> The model is {confidence}% confident this is <strong>{pred_class}</strong>.
                                </div>
                                """)
                            elif status == "Healthy":
                                st.html(f"""
                                <div style="background-color: #c8e6c9; padding: 10px; border-radius: 8px; margin-bottom: 15px;">
                                    <strong>🌿 Plant Status:</strong> {status} with {confidence}% confidence. Your plant appears healthy!
                                </div>
                                """)

                            analysis_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            date, time = analysis_time.split()

                            st.success("Analysis Complete!")
                            
                            result_col1, result_col2 = st.columns(2)
                            
                            with result_col1:
                                st.html(f"""
                                <div style="background-color: #f0f8f0; padding: 15px; border-radius: 10px;">
                                    <h3 style="color: #2e8b57;">Results</h3>
                                    <p><strong>Crop:</strong> {crop}</p>
                                    <p><strong>Status:</strong> {status}</p>
                                    <p><strong>Diagnosis:</strong> {pred_class}</p>
                                    <p><strong>Confidence:</strong> {confidence}%</p>
                                    <p><strong>Growth Stage:</strong> {details['growth_stage']}</p>
                                    <p><strong>Cause:</strong> {details['cause']}</p>
                                    <p><strong>Nutrient Deficiency:</strong> {details['nutrient_deficiency']}</p>
                                    <p><strong>Solution:</strong> {details['solution']}</p>
                                    <p><strong>Recommended Fertilizer:</strong> {details['fertilizer']}</p>
                                </div>
                                """)
                            
                            with result_col2:
                                st.html(f"""
                                <div style="background-color: #e6f3ff; padding: 15px; border-radius: 10px;">
                                    <h3 style="color: #1e6fbb;">Weather Report</h3>
                                    <div style="display: flex; align-items: center;">
                                        <img src="{get_weather_icon(forecast['weather_icon'])}" width="50">
                                        <div style="margin-left: 10px;">
                                            <p><strong>Location:</strong> {forecast['location']}</p>
                                            <p><strong>Temperature:</strong> {forecast['temperature']}</p>
                                            <p><strong>Humidity:</strong> {forecast['humidity']}</p>
                                        </div>
                                    </div>
                                    <p><strong>Advice:</strong> {forecast['advice']}</p>
                                </div>
                                """)
                            
                            report_data = {
                                "date": date,
                                "time": time,
                                "crop": crop,
                                "analysis_type": "Disease Detection",
                                "status": status,
                                "disease": pred_class,
                                "confidence": f"{confidence}%", # ADDED back
                                "weather": forecast,
                            }

                            pdf_path = generate_pdf(report_data, temp_image_path)
                            
                            with open(pdf_path, "rb") as f:
                                st.download_button(
                                    "📄 Download Full Report (PDF)",
                                    f,
                                    file_name=f"agrilens_report_{crop.lower()}_{date}.pdf",
                                    mime="application/pdf",
                                    use_container_width=True
                                )
                        else:
                            st.warning("Weather data could not be fetched for the provided location. Please check the location and try again.")
                        
                    except FileNotFoundError as e:
                        st.error(f"Required model or class files are missing. Please ensure they are in the 'models/' and 'data/' directories. Error: {str(e)}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred during analysis: {str(e)}")
                        st.error(f"Error type: {type(e).__name__}")
                        if hasattr(e, 'args') and e.args:
                            st.error(f"Error message: {e.args[0]}")
                    finally:
                        if temp_image_path and os.path.exists(temp_image_path):
                            try:
                                os.unlink(temp_image_path)
                            except Exception as e:
                                st.warning(f"Could not delete temporary file: {e}")
            else:
                st.warning("Please upload an image and enter your location to analyze.")

    elif page == "🌱 Crop Recommendation":
        st.header("🌱 Smart Crop Recommendation")
        st.markdown("Get personalized crop suggestions based on your soil conditions and climate.")
        
        with st.expander("ℹ️ About This Tool", expanded=True):
            st.markdown("""
            Our AI model recommends the best crops to plant based on:
            - Soil nutrient levels (N, P, K)
            - Temperature and humidity
            - Soil pH and rainfall
            - Your local weather conditions
            """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Soil Parameters")
            N = st.slider("Nitrogen (N) level", 0, 150, 50, help="Nitrogen content in soil (kg/ha)")
            P = st.slider("Phosphorus (P) level", 0, 150, 50, help="Phosphorus content in soil (kg/ha)")
            K = st.slider("Potassium (K) level", 0, 150, 50, help="Potassium content in soil (kg/ha)")
            ph = st.slider("Soil pH", 0.0, 14.0, 7.0, 0.1, help="Soil pH level (0-14 scale, 7 is neutral)")
        
        with col2:
            st.subheader("Climate Parameters")
            temperature = st.slider("Temperature (°C)", -10.0, 50.0, 25.0, 0.1, help="Average temperature (°C)")
            humidity = st.slider("Humidity (%)", 0, 100, 60, help="Relative humidity level (%)")
            rainfall = st.slider("Rainfall (mm)", 0.0, 500.0, 100.0, 1.0, help="Annual rainfall (mm)")
            location = st.text_input("📍 Your Location (optional)", help="For weather-specific recommendations")
        
        if st.button("Get Recommendation", type="primary", use_container_width=True):
            with st.spinner("Analyzing your soil and climate..."):
                try:
                    if not os.path.exists("models/crop_recommendation_model.pkl"):
                        st.error("Crop recommendation model not found. Please ensure the model file exists in the 'models' directory.")
                        return
                    
                    model = joblib.load("models/crop_recommendation_model.pkl")
                    prediction = model.predict([[N, P, K, temperature, humidity, ph, rainfall]])
                    recommended_crop = prediction[0].title()
                    
                    st.success(f"🌾 Recommended Crop: **{recommended_crop}**")
                    
                    if location:
                        forecast = get_weather_report(location)
                        if forecast:
                            st.info(f"**Weather in {location}:** Current temperature: {forecast['temperature']}, Humidity: {forecast['humidity']}")
                            st.info(f"**Weather-based Recommendation:** {forecast['advice']}")
                        else:
                            st.warning("Could not fetch weather data for the provided location. Recommendation is based on soil and climate parameters only.")
                    
                    st.markdown("---")
                    st.subheader(f"About Growing {recommended_crop}")
                    st.info(f"Detailed growing tips for {recommended_crop} will be available here. This section can include information on optimal planting times, soil requirements, pest management, and harvesting techniques.")
                    
                except Exception as e:
                    st.error(f"Error generating recommendation: {str(e)}")
                    st.error(f"Error details: {type(e).__name__}")
                    if hasattr(e, 'args') and e.args:
                        st.error(f"Error message: {e.args[0]}")

    # Global footer displayed at the bottom of every page
    st.markdown("---")
    st.html("""
    <div style="text-align: center; padding: 10px; opacity: 0.85;">
        <p style="color: #64748b; font-size: 0.95rem; font-weight: 600; margin: 0; font-family: 'Outfit', sans-serif;">
            Built with ❤️ by Akhil
        </p>
    </div>
    """)

if __name__ == "__main__":
    main()

