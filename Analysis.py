import pandas as pd

# Load the dataset
file_path = "C:\\Users\\Nelvin\\Desktop\\Insecurity Data\\security_incidents_2025-03-10.csv"  # Update with your correct file path
df = pd.read_csv(file_path)

# Show basic information about the dataset
print(df.info())
print(df.head())

# Drop the first row if it contains incorrect headers
df = df.iloc[1:].reset_index(drop=True)

# Convert numerical columns to the correct format
numeric_cols = [
    "Year", "Month", "Day", "Nationals killed", "Nationals wounded", "Nationals kidnapped", 
    "Total nationals", "Internationals killed", "Internationals wounded", "Internationals kidnapped", 
    "Total internationals", "Total killed", "Total wounded", "Total kidnapped", "Total affected", 
    "Latitude", "Longitude"
]

df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

# Fill missing values
df["Month"].fillna(df["Month"].mode()[0], inplace=True)
df["Day"].fillna(df["Day"].mode()[0], inplace=True)
df["Latitude"].fillna(df["Latitude"].mean(), inplace=True)
df["Longitude"].fillna(df["Longitude"].mean(), inplace=True)

# Replace NaN values in victim-related columns with 0
victim_cols = [
    "Nationals killed", "Nationals wounded", "Nationals kidnapped", "Total nationals",
    "Internationals killed", "Internationals wounded", "Internationals kidnapped", "Total internationals",
    "Total killed", "Total wounded", "Total kidnapped", "Total affected"
]
df[victim_cols] = df[victim_cols].fillna(0)

print(df.info())  # Check that missing values have been handled
print(df.head())  # Display the first few rows of the cleaned dataset

import matplotlib.pyplot as plt
import seaborn as sns

# Count the number of incidents per year
# 1. Security Incidents Over the Years
incident_counts = df.groupby("Year").size().reset_index(name="Number of Incidents")
plt.figure(figsize=(10, 5))
sns.barplot(data=incident_counts, x="Year", y="Number of Incidents", color="orange")
plt.xticks(rotation=45)
plt.xlabel("Year")
plt.ylabel("Number of Incidents")
plt.title("Security Incidents Over the Years")
plt.savefig("incident_trend.png")
plt.show()


# 2. Most Common Means of Attack
attack_counts = df["Means of attack"].value_counts().reset_index()
attack_counts.columns = ["Means of Attack", "Number of Incidents"]
plt.figure(figsize=(10, 5))
sns.barplot(data=attack_counts.head(10), x="Number of Incidents", y="Means of Attack", color="blue")
plt.xlabel("Number of Incidents")
plt.ylabel("Means of Attack")
plt.title("Top 10 Most Common Means of Attack")
plt.savefig("means_of_attack.png")
plt.show()


# 3. Most Affected Countries
country_counts = df["Country"].value_counts().reset_index()
country_counts.columns = ["Country", "Number of Incidents"]
plt.figure(figsize=(10, 5))
sns.barplot(data=country_counts.head(10), x="Number of Incidents", y="Country", color="green")
plt.xlabel("Number of Incidents")
plt.ylabel("Country")
plt.title("Top 10 Most Affected Countries")
plt.savefig("affected_countries.png")
plt.show()

# Box plot of attack type vs. total affected people
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="Means of attack", y="Total affected", showfliers=False)
plt.xticks(rotation=90)
plt.xlabel("Means of Attack")
plt.ylabel("Total Affected People (Killed + Wounded + Kidnapped)")
plt.title("Impact of Different Means of Attack on Number of Affected People")
plt.show()

# 6. Correlation Heatmap for Victim Counts
victim_cols = ["Total affected", "Total killed", "Total wounded", "Total kidnapped"]
plt.figure(figsize=(10, 6))
sns.heatmap(df[victim_cols].corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix for Victim Counts")
plt.savefig("correlation_matrix.png")
plt.show()



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Prepare data
incident_counts = df.groupby("Year").size().reset_index(name="Total Incidents")
X = incident_counts["Year"].values.reshape(-1, 1)
y = incident_counts["Total Incidents"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict future incidents (2026 & 2027)
future_years = np.array([2026, 2027]).reshape(-1, 1)
future_predictions = model.predict(future_years)

print("Predicted Incidents for 2026:", future_predictions[0])
print("Predicted Incidents for 2027:", future_predictions[1])

# Visualizing Predictions
future_df = pd.DataFrame({"Year": [2026, 2027], "Predicted Incidents": future_predictions})
combined_df = pd.concat([incident_counts, future_df], ignore_index=True)

plt.figure(figsize=(10, 5))
plt.plot(combined_df["Year"], combined_df["Number of Incidents"], marker="o", label="Actual Incidents", linestyle="-")
plt.plot(combined_df["Year"], combined_df["Predicted Incidents"], marker="s", label="Predicted Incidents", linestyle="--", color="red")
plt.xlabel("Year")
plt.ylabel("Number of Incidents")
plt.title("Actual vs Predicted Security Incidents (Forecast for 2026 & 2027)")
plt.legend()
plt.savefig("predicted_incidents.png")
plt.show()


with pd.ExcelWriter("security_incidents_analysis.xlsx", engine="xlsxwriter") as writer:
    df.to_excel(writer, sheet_name="Cleaned Data", index=False)
    incident_counts.to_excel(writer, sheet_name="Incidents Per Year", index=False)
    attack_counts.to_excel(writer, sheet_name="Means of Attack", index=False)
    writer.save()

  
