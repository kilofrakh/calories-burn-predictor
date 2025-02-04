import random
import pandas as pd

# Define parameters
num_rows = 200000
ages = list(range(18, 60))  # Runners between 18 and 60 years old
genders = ["Male", "Female"]
bmi_ranges = {"Male": (18, 30), "Female": (18, 28)}  # BMI ranges for males and females
distances = [5, 10, 15, 21, 42]  # Common race distances in km

# Generate data
data = []
for i in range(1, num_rows + 1):
    age = random.choice(ages)
    gender = random.choice(genders)
    bmi = round(random.uniform(*bmi_ranges[gender]), 1)
    distance = random.choice(distances)
    
    # Generate realistic times (minutes) based on distance, age, and BMI
    base_time_per_km = random.uniform(4, 8)  # Base time per km (min/km)
    time = round(distance * base_time_per_km * (1 + (bmi - 22) * 0.02 + (age - 30) * 0.01), 2)

    # Estimate calories burned using METs (simplified formula)
    met = 10 if distance >= 10 else 8  # Higher MET for longer distances
    weight = 70 + (bmi - 22) * 2  # Approximate weight assumption
    calories_burned = round(met * weight * (time / 60), 2)
    
    data.append([i, age, gender, bmi, distance, time, calories_burned])

# Create DataFrame
df = pd.DataFrame(data, columns=["Gender","Age","Height(cm)","Weight(kg)","BMI","Running Time(min)","Running Speed(km/h)","Distance(km)","Average Heart Rate","Calories Burned"])

# Save to CSV

file_path = r"C:\Users\YourUsername\Desktop\running_data.csv"
df.to_csv(file_path, index=False)


df.to_csv(file_path, index=False)

file_path
