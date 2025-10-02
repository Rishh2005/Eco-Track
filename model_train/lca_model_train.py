import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import io


# Full CSV data as a string with proper quoting and header
csv_data = '''Material,Process,Production_Volume_t,Ore_Grade_%,Transport_Distance_km,Energy_Source,Water_Source,Energy_Consumption_MWh,Water_Usage_L,Carbon_Emissions_tonnes_CO2,Land_Use_m2,Resource_Depletion_kg,Human_Toxicity_CTUh,Eco_Toxicity_CTUE,Acidification_kg_SO2,Photochemical_Ozone_kg,Particulate_Matter_kg,Smog_kg_O3,Impact_Category,Application
"Copper","Underground Mining",950,1.4,210,"Coal","Freshwater",14.0,2.6,15100,980000,1220,52400,9,942,294,2440,1.4,"SO2/CO2","Electrical Cables"
"Copper","Heap Leaching",1700,0.5,75,"Renewable","Desalinated",23.0,0.6,9800,1450000,570,29800,10,495,147,4100,2.2,"Particulates","Electronics,Pipes"
"Copper","Smelting/Refining",1200,0.9,185,"Mixed Grid","Recycled",46.0,1.8,13400,1230000,930,50100,10,860,275,3650,1.6,"NOx/SO2/CO2","Construction"
"Iron Ore","Open-pit Mining",4000,54.0,85,"Coal","Freshwater",0.0,0.0,15200,2300000,1420,73400,10,1650,365,5800,4.1,"Dust/NOx/CO2","Steelmaking"
"Iron Ore","Pelletizing",2750,59.0,120,"Mixed Grid","Desalinated",2.7,0.0,13100,1670000,1005,51300,9,1320,313,3200,3.4,"SO2/CO2","Construction"
"Iron Ore","Sintering",1900,61.0,140,"Renewable","Recycled",7.5,1.9,11400,1130000,680,39100,10,790,186,2600,2.8,"Particulates","Automotive"
"Nickel","Laterite HPAL",600,1.2,180,"Coal","Desalinated",5.2,1.7,13800,1580000,1570,37600,10,2010,430,1180,1.2,"SO2/CO2","Stainless Steel"
"Nickel","Sulfide Smelting",850,2.0,220,"Mixed Grid","Freshwater",10.5,2.3,11100,1210000,950,31900,10,1570,410,980,1.0,"CO2/SO2","Batteries"
"Nickel","Powder Production",400,1.1,90,"Renewable","Recycled",33.0,0.8,9200,890000,530,21400,9,870,151,630,0.6,"Dust/CO2","Electronics"
'''

# Read CSV data into DataFrame
df = pd.read_csv(io.StringIO(csv_data))

# Drop rows with missing target
df = df.dropna(subset=["Carbon_Emissions_tonnes_CO2"])

# Select numeric features for regression
features = [
    "Production_Volume_t",
    "Ore_Grade_%",
    "Transport_Distance_km",
    "Energy_Consumption_MWh",
    "Water_Usage_L",
    "Recycling_Rate_%",
    "Contamination_Level_%"
]
features = [f for f in features if f in df.columns]
X = df[features].astype(float)
y = df["Carbon_Emissions_tonnes_CO2"].astype(float)

# Train regression model
model = LinearRegression()
model.fit(X, y)
pred = model.predict(X)
mse = mean_squared_error(y, pred)
r2 = r2_score(y, pred)

# Calculate percentage contributions for a few key columns
summary = {}
for col in ["Energy_Consumption_MWh", "Water_Usage_L", "Recycling_Rate_%", "Contamination_Level_%"]:
    if col in df.columns:
        total = df[col].astype(float).sum()
        summary[col] = total
total_emissions = y.sum()

# Save summary report

# Save plain text report
with open('lca_report.txt', 'w') as f:
    f.write('LCA Model Report\n')
    f.write(f'Mean Squared Error: {mse:.2f}\n')
    f.write(f'R2 Score: {r2:.2f}\n')
    f.write(f'Total Carbon Emissions: {total_emissions:.2f}\n')
    for k, v in summary.items():
        percent = (v / total_emissions * 100) if total_emissions else 0
        f.write(f'{k}: {v:.2f} ({percent:.2f}% of total emissions)\n')

# Save HTML table report
table_rows = ''
for k, v in summary.items():
    percent = (v / total_emissions * 100) if total_emissions else 0
    table_rows += f'<tr><td>{k}</td><td>{v:.2f}</td><td>{percent:.2f}%</td></tr>'

html_report = f'''
<html>
<head>
    <style>
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
        th {{ background: #f2f2f2; }}
    </style>
</head>
<body>
    <h2>LCA Model Report</h2>
    <p><b>Mean Squared Error:</b> {mse:.2f}</p>
    <p><b>R2 Score:</b> {r2:.2f}</p>
    <p><b>Total Carbon Emissions:</b> {total_emissions:.2f}</p>
    <table>
        <thead>
            <tr><th>Metric</th><th>Value</th><th>% of Total Emissions</th></tr>
        </thead>
        <tbody>
            {table_rows}
        </tbody>
    </table>
</body>
</html>
'''
with open('lca_report.html', 'w', encoding='utf-8') as f:
    f.write(html_report)

# Plot actual vs predicted
plt.figure(figsize=(10,5))
plt.plot(y.values, label='Actual', marker='o')
plt.plot(pred, label='Predicted', marker='x')
plt.title('Actual vs Predicted Carbon Emissions')
plt.xlabel('Sample')
plt.ylabel('CO₂ (tonnes)')
plt.legend()
plt.tight_layout()
plt.savefig('carbon_emissions_chart.png')
plt.close()

# Pie chart for Energy Source distribution (if present)
if 'Energy_Source' in df.columns:
    plt.figure(figsize=(6,6))
    df['Energy_Source'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title('Energy Source Distribution')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('energy_source_pie.png')
    plt.close()

print('Model trained!')
print(f'Mean Squared Error: {mse:.2f}')
print(f'R2 Score: {r2:.2f}')
print('Charts saved as carbon_emissions_chart.png and energy_source_pie.png')
print('Report saved as lca_report.txt')
