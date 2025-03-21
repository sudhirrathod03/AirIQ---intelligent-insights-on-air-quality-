import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
from datetime import datetime
import io
from fpdf import FPDF
import tempfile 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, r2_score

# ----------------------------------------------------
# 1. Load & Preprocess the Data
# ----------------------------------------------------
st.title("Air Pollution Data Dashboard & Analysis")

file_path = "TechBlitz_DataScience_Dataset_preprocessed.csv"
try:
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # Remove extra spaces
    st.success("Dataset loaded successfully.")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()


rename_dict = {
    "Temperature": "Temperature (°C)",
    "Humidity": "Humidity (%)",
    "PM2.5": "PM2.5 Concentration (µg/m³)",
    "PM10": "PM10 Concentration (µg/m³)",
    "NO2": "NO2 Concentration (ppb)",
    "SO2": "SO2 Concentration (ppb)",
    "CO": "CO Concentration (ppm)",
    "Proximity_to_Industrial_Areas": "Proximity to Industrial Areas (km)",
    "Population_Density": "Population Density (people/km²)",
    "Air Quality": "Air Quality Levels"
}
df.rename(columns=rename_dict, inplace=True)

# Drop missing values
df.dropna(inplace=True)


if "Time" not in df.columns:
    df["Time"] = pd.date_range(start="2023-01-01", periods=len(df), freq="D")
df["Time"] = pd.to_datetime(df["Time"])  # Ensure Time is datetime type


if "lat" not in df.columns or "lon" not in df.columns:
    center_lat, center_lon = 28.6139, 77.2090  # Example: New Delhi
    np.random.seed(42)
    df["lat"] = center_lat + np.random.normal(0, 0.05, len(df))
    df["lon"] = center_lon + np.random.normal(0, 0.05, len(df))

# Add simulated Region column based on lat/lon
center_lat, center_lon = 28.6139, 77.2090
df["Region"] = np.where(
    (df["lat"] > center_lat) & (df["lon"] > center_lon), "North-East",
    np.where((df["lat"] > center_lat) & (df["lon"] <= center_lon), "North-West",
    np.where((df["lat"] <= center_lat) & (df["lon"] > center_lon), "South-East", "South-West"))
)


expected_features = [
    "Temperature (°C)", "Humidity (%)", "PM2.5 Concentration (µg/m³)", 
    "PM10 Concentration (µg/m³)", "NO2 Concentration (ppb)", "SO2 Concentration (ppb)",
    "CO Concentration (ppm)", "Proximity to Industrial Areas (km)", "Population Density (people/km²)"
]
target = "Air Quality Levels"

available_features = [col for col in expected_features if col in df.columns]
if target not in df.columns:
    st.error(f"Target column '{target}' not found!")
    st.stop()

# ----------------------------------------------------
# 2. Multi-Tab Layout
# ----------------------------------------------------
tabs = st.tabs([
    "Overview", "Time Series", "Comparative Plots", "Correlation", 
    "Maps", "Advanced Visuals", "ML Model", "Real-Time Prediction",
    "Additional Models", "3D Visualizations", "Insights"
])

# ----------------------------------------------------
# Tab 1: Interactive Dashboard (Enhanced Overview)
# ----------------------------------------------------
with tabs[0]:
    st.header("Interactive Dashboard")
    
    # Filters
    with st.expander("Filter Data"):
        # Time range filter
        min_date = df["Time"].min().date()
        max_date = df["Time"].max().date()
        date_range = st.date_input("Select Date Range", [min_date, max_date])
        start_date, end_date = date_range
        
        # Region filter
        regions = st.multiselect("Select Regions", df["Region"].unique(), default=df["Region"].unique())
        
        # Pollution source filter (Proximity to Industrial Areas)
        min_proximity = float(df["Proximity to Industrial Areas (km)"].min())
        max_proximity = float(df["Proximity to Industrial Areas (km)"].max())
        proximity_range = st.slider("Proximity to Industrial Areas (km)", min_proximity, max_proximity, (min_proximity, max_proximity))
    
    # Apply filters
    filtered_df = df[
        (df["Time"].dt.date >= start_date) & (df["Time"].dt.date <= end_date) &
        (df["Region"].isin(regions)) &
        (df["Proximity to Industrial Areas (km)"] >= proximity_range[0]) &
        (df["Proximity to Industrial Areas (km)"] <= proximity_range[1])
    ]
    
    # Display key metrics
    st.subheader("Key Metrics")
    col1, col2 = st.columns(2)
    with col1:
        avg_pm25 = filtered_df["PM2.5 Concentration (µg/m³)"].mean()
        st.metric("Average PM2.5", f"{avg_pm25:.2f} µg/m³")
    with col2:
        avg_aqi = filtered_df["Air Quality Levels"].mode()[0] if not filtered_df.empty else "N/A"
        st.metric("Most Common Air Quality Level", avg_aqi)
    
    # Visualizations
    st.subheader("Visualizations")
    col3, col4 = st.columns(2)
    with col3:
        fig_line = px.line(filtered_df, x="Time", y="PM2.5 Concentration (µg/m³)", title="PM2.5 Over Time")
        st.plotly_chart(fig_line, use_container_width=True)
    with col4:
        region_pm25 = filtered_df.groupby("Region")["PM2.5 Concentration (µg/m³)"].mean().reset_index()
        fig_bar = px.bar(region_pm25, x="Region", y="PM2.5 Concentration (µg/m³)", title="Average PM2.5 by Region")
        st.plotly_chart(fig_bar, use_container_width=True)

# ----------------------------------------------------
# Tab 2: Time Series Analysis
# ----------------------------------------------------
with tabs[1]:
    st.header("Time Series Analysis")
    st.write("Visualize trends in air pollution over time.")
    # Allow user to choose a pollutant
    pollutant = st.selectbox("Select a Pollutant", ["PM2.5 Concentration (µg/m³)",
                                                     "PM10 Concentration (µg/m³)",
                                                     "NO2 Concentration (ppb)",
                                                     "SO2 Concentration (ppb)",
                                                     "CO Concentration (ppm)"])
    fig_time = px.line(df, x="Time", y=pollutant, title=f"{pollutant} Over Time")
    st.plotly_chart(fig_time, use_container_width=True)

# ----------------------------------------------------
# Tab 3: Comparative Plots
# ----------------------------------------------------
with tabs[2]:
    st.header("Comparative Plots")
    st.write("Compare pollutant levels across different regions.")
    # Example: Box plot comparing PM2.5 concentrations by Region
    fig_box = px.box(df, x="Region", y="PM2.5 Concentration (µg/m³)", 
                     title="Distribution of PM2.5 Concentration by Region")
    st.plotly_chart(fig_box, use_container_width=True)

# ----------------------------------------------------
# Tab 4: Correlation Analysis
# ----------------------------------------------------
with tabs[3]:
    st.header("Correlation Analysis")
    st.write("View the correlation between different features.")
    fig_corr_tab, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(df[available_features].corr(), annot=True, cmap="viridis", ax=ax)
    st.pyplot(fig_corr_tab)

# ----------------------------------------------------
# Tab 5: Maps
# ----------------------------------------------------
with tabs[4]:
    st.header("Maps")
    st.write("Map view of air pollution measurement locations.")
    # Using Plotly's scatter_mapbox
    fig_map = px.scatter_mapbox(df, lat="lat", lon="lon", 
                                color="PM2.5 Concentration (µg/m³)", 
                                size="PM2.5 Concentration (µg/m³)",
                                hover_data=["Region"],
                                zoom=10, mapbox_style="open-street-map",
                                title="Air Pollution Map")
    st.plotly_chart(fig_map, use_container_width=True)

# ----------------------------------------------------
# Tab 6: Advanced Visuals
# ----------------------------------------------------
with tabs[5]:
    st.header("Advanced Visuals")
    st.write("Explore additional interactive visualizations.")
    # Example: Multi-line chart for several pollutants over time
    pollutants = ["PM2.5 Concentration (µg/m³)", "PM10 Concentration (µg/m³)", "NO2 Concentration (ppb)"]
    fig_adv = px.line(df, x="Time", y=pollutants, title="Multiple Pollutants Over Time")
    st.plotly_chart(fig_adv, use_container_width=True)

# ----------------------------------------------------
# Tab 7: ML Model (Train & Evaluate Random Forest)
# ----------------------------------------------------
with tabs[6]:
    st.header("ML Model")
    st.write("Train a Random Forest Classifier on the air pollution data.")
    
    X = df[available_features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    global rf_model
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    
    rf_accuracy = accuracy_score(y_test, y_pred)
    st.write("Random Forest Accuracy: ", rf_accuracy)
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

# ----------------------------------------------------
# Tab 8: Real-Time Prediction
# ----------------------------------------------------
with tabs[7]:
    st.header("Real-Time Prediction")
    st.write("Enter feature values for a prediction using the Random Forest model.")
    
    input_features = {}
    for feature in available_features:
        input_features[feature] = st.number_input(feature, value=float(df[feature].mean()))
    
    if st.button("Predict Air Quality"):
        if 'rf_model' in globals():
            input_df = pd.DataFrame([input_features])
            prediction = rf_model.predict(input_df)[0]
            st.success(f"Predicted Air Quality: {prediction}")
        else:
            st.error("Random Forest model not trained yet. Please train the model in the 'ML Model' tab.")

# ----------------------------------------------------
# Tab 9: Additional Models
# ----------------------------------------------------
with tabs[8]:
    st.header("Additional Models")
    st.write("Train a Gradient Boosting Classifier on the data.")
    
    X = df[available_features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    global gb_model
    gb_model = GradientBoostingClassifier(random_state=42)
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)
    gb_accuracy = accuracy_score(y_test, y_pred_gb)
    
    st.write("Gradient Boosting Accuracy: ", gb_accuracy)
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred_gb))

# ----------------------------------------------------
# Tab 10: 3D Visualizations
# ----------------------------------------------------
with tabs[9]:
    st.header("3D Visualizations")
    st.write("3D Scatter Plot of Temperature, Humidity, and PM2.5 Concentration.")
    
    # Increase the figure size using width and height
    fig_3d = px.scatter_3d(
        df,
        x="Temperature (°C)",
        y="Humidity (%)",
        z="PM2.5 Concentration (µg/m³)",
        color="Air Quality Levels",
        title="3D Scatter Plot: Temperature vs Humidity vs PM2.5",
        width=1000,  # Increase width
        height=800   # Increase height
    )
    
    # Optionally adjust margins or aspect ratio
    fig_3d.update_layout(
        margin=dict(l=0, r=0, b=0, t=50),
        scene=dict(aspectmode='cube')
    )
    
    # Disable use_container_width so the figure is not auto-resized
    st.plotly_chart(fig_3d, use_container_width=False)

# ----------------------------------------------------
# Tab 11: Insights
# ----------------------------------------------------
with tabs[10]:
    st.header("Insights")
    
    # Correlation Heatmap
    st.subheader("Correlation Analysis")
    fig_corr, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df[available_features].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig_corr)
    
    # Feature Importance from Random Forest
    st.subheader("Feature Importance (Random Forest)")
    if 'rf_model' in globals():
        importance = rf_model.feature_importances_
        fi_df = pd.DataFrame({"Feature": available_features, "Importance": importance}).sort_values(by="Importance", ascending=False)
        fig_fi, ax = plt.subplots()
        sns.barplot(x="Importance", y="Feature", data=fi_df, ax=ax)
        st.pyplot(fig_fi)
    else:
        st.warning("Train the Random Forest model in the 'ML Model' tab to see feature importance.")
    
    # Top Pollutants by Region
    st.subheader("Top Pollutants by Region")
    top_pollutants = df.groupby("Region")[available_features].mean().idxmax(axis=1)
    st.write(top_pollutants)
    
    # Scatter Plot for Model Accuracy Comparison
    st.subheader("Model Accuracy Comparison")
    # Recompute accuracies on a fresh train/test split
    X = df[available_features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_acc = accuracy_score(y_test, rf_model.predict(X_test)) if 'rf_model' in globals() else None
    gb_acc = accuracy_score(y_test, gb_model.predict(X_test)) if 'gb_model' in globals() else None
    models = []
    accuracies = []
    if rf_acc is not None:
        models.append("Random Forest")
        accuracies.append(rf_acc)
    if gb_acc is not None:
        models.append("Gradient Boosting")
        accuracies.append(gb_acc)
    fig_acc, ax = plt.subplots()
    ax.scatter(models, accuracies, s=100, color='purple')
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Comparison")
    st.pyplot(fig_acc)
    
    # PDF Export of Insights
    st.subheader("Export Insights to PDF")
    
    def generate_pdf_report():
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Air Pollution Data Insights", 0, 1, "C")
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, "C")
        
        # Helper function: add matplotlib figure to PDF using a temporary file
        def add_figure_to_pdf(fig, title):
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, title, 0, 1)
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                tmp_file.write(buf.getvalue())
                tmp_file_path = tmp_file.name
            pdf.image(tmp_file_path, x=10, y=None, w=190)
            pdf.ln(10)
        
        # Add Correlation Heatmap
        fig_corr_pdf, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(df[available_features].corr(), annot=True, cmap="coolwarm", ax=ax)
        add_figure_to_pdf(fig_corr_pdf, "Correlation Analysis")
        
      
        if 'rf_model' in globals():
            fig_fi_pdf, ax = plt.subplots(figsize=(6,4))
            importance = rf_model.feature_importances_
            fi_df = pd.DataFrame({"Feature": available_features, "Importance": importance}).sort_values(by="Importance", ascending=False)
            sns.barplot(x="Importance", y="Feature", data=fi_df, ax=ax)
            add_figure_to_pdf(fig_fi_pdf, "Feature Importance (Random Forest)")
        
        # Add Top Pollutants by Region (text output)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Top Pollutants by Region", 0, 1)
        top_pollutants = df.groupby("Region")[available_features].mean().idxmax(axis=1)
        for region, pollutant in top_pollutants.items():
            pdf.cell(0, 10, f"{region}: {pollutant}", 0, 1)
        
        # Add Model Accuracy Scatter Plot
        fig_acc_pdf, ax = plt.subplots()
        models_pdf = []
        acc_pdf = []
        if rf_acc is not None:
            models_pdf.append("Random Forest")
            acc_pdf.append(rf_acc)
        if gb_acc is not None:
            models_pdf.append("Gradient Boosting")
            acc_pdf.append(gb_acc)
        ax.scatter(models_pdf, acc_pdf, s=100, color='purple')
        ax.set_xlabel("Model")
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Accuracy Comparison")
        add_figure_to_pdf(fig_acc_pdf, "Model Accuracy Comparison")
        
        pdf_output = pdf.output(dest='S').encode('latin1')
        return pdf_output

    if st.button("Generate PDF Report"):
        pdf_bytes = generate_pdf_report()
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name="air_pollution_insights.pdf",
            mime="application/pdf"
        )
