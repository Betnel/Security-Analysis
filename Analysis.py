import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Load the dataset
file_path = "C:\\Users\\Nelvin\\Desktop\\Insecurity Data\\security_incidents_2025-03-10.csv"
df = pd.read_csv(file_path)

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

# Create PDF report
with PdfPages("C:/Users/Nelvin/Desktop/security_incident_report.pdf") as pdf:

    # 1. Security Incidents Over the Years
    incident_counts = df.groupby("Year").size().reset_index(name="Number of Incidents")
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=incident_counts, x="Year", y="Number of Incidents", color="orange")
    plt.xticks(rotation=45)
    plt.xlabel("Year")
    plt.ylabel("Number of Incidents")
    plt.title("Security Incidents Over the Years")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')
    pdf.savefig()
    plt.close()

    # 2. Most Common Means of Attack
    attack_counts = df["Means of attack"].value_counts().reset_index()
    attack_counts.columns = ["Means of Attack", "Number of Incidents"]
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=attack_counts.head(10), x="Number of Incidents", y="Means of Attack", color="blue")
    plt.xlabel("Number of Incidents")
    plt.ylabel("Means of Attack")
    plt.title("Top 10 Most Common Means of Attack")
    for p in ax.patches:
        width = p.get_width()
        ax.annotate(f"{int(width)}", (width + 1, p.get_y() + p.get_height() / 2), va='center')
    pdf.savefig()
    plt.close()

    # 3. Most Affected Countries
    country_counts = df["Country"].value_counts().reset_index()
    country_counts.columns = ["Country", "Number of Incidents"]
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=country_counts.head(10), x="Number of Incidents", y="Country", color="green")
    plt.xlabel("Number of Incidents")
    plt.ylabel("Country")
    plt.title("Top 10 Most Affected Countries")
    for p in ax.patches:
        width = p.get_width()
        ax.annotate(f"{int(width)}", (width + 1, p.get_y() + p.get_height() / 2), va='center')
    pdf.savefig()
    plt.close()

    # 4. Box Plot - Impact of Means of Attack (Top 10 only)
    top_attacks = df["Means of attack"].value_counts().head(10).index
    filtered_df = df[df["Means of attack"].isin(top_attacks)]

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=filtered_df, x="Means of attack", y="Total affected", showfliers=False)
    plt.xticks(rotation=90)
    plt.xlabel("Means of Attack")
    plt.ylabel("Total Affected People (Killed + Wounded + Kidnapped)")
    plt.title("Impact of Top 10 Attack Types on Number of Affected People")
    pdf.savefig()
    plt.close()

    # 5. Correlation Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df[victim_cols].corr(), annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Matrix for Victim Counts")
    pdf.savefig()
    plt.close()

    # 6. Forecasting with Random Forest + Improved Plot
    incident_counts.rename(columns={"Number of Incidents": "Total Incidents"}, inplace=True)
    X = incident_counts["Year"].values.reshape(-1, 1)
    y = incident_counts["Total Incidents"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    future_years = np.array([2026, 2027]).reshape(-1, 1)
    future_predictions = model.predict(future_years)
    future_df = pd.DataFrame({"Year": [2026, 2027], "Predicted Incidents": future_predictions})
    combined_df = pd.concat([incident_counts, future_df], ignore_index=True)

    plt.figure(figsize=(12, 6))
    plt.plot(combined_df["Year"], combined_df["Total Incidents"], marker="o", label="Actual Incidents", linestyle="-", color="blue")
    plt.plot(future_df["Year"], future_df["Predicted Incidents"], marker="s", label="Predicted Incidents", linestyle="--", color="red")

    for x, y in zip(combined_df["Year"], combined_df["Total Incidents"]):
        if not pd.isna(y):
            plt.text(x, y + 5, str(int(y)), ha='center', fontsize=9, color="blue")
    for x, y in zip(future_df["Year"], future_df["Predicted Incidents"]):
        plt.text(x, y + 5, str(int(y)), ha='center', fontsize=9, color="red")

    plt.xlabel("Year")
    plt.ylabel("Number of Incidents")
    plt.title("Actual vs Predicted Security Incidents (Forecast for 2026 & 2027)")
    plt.legend()
    plt.ylim(0, max(combined_df["Total Incidents"].max(), future_df["Predicted Incidents"].max()) + 40)
    plt.grid(True, linestyle='--', alpha=0.5)
    pdf.savefig()
    plt.close()

# Export Excel
with pd.ExcelWriter("C:/Users/Nelvin/Desktop/security_incidents_analysis.xlsx", engine="xlsxwriter") as writer:
    df.to_excel(writer, sheet_name="Cleaned Data", index=False)
    incident_counts.to_excel(writer, sheet_name="Incidents Per Year", index=False)
    attack_counts.to_excel(writer, sheet_name="Means of Attack", index=False)
