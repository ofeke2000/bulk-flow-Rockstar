import csv
import json
import os

# Expand the file path
file_path = os.path.expanduser("~/bulk-flow-Rockstar/Data/mdpl2_rockstar_125_pid_-1/mdpl2_rockstar_pid_-1.csv")
json_path = "rockstar_125.jsonl"  # JSON Lines file

with open(file_path, mode='r', newline='') as file, open(json_path, "w") as json_file:
    reader = csv.DictReader(file)

    for row in reader:
        # Convert data types where necessary
        row["rockstarid"] = int(row["rockstarid"])
        row["pid"] = int(row["pid"])

        # Convert all other numeric fields to float
        for key in row:
            if key not in ["rockstarid", "pid"]:
                row[key] = float(row[key])

        # Write each row as a separate JSON object
        json_file.write(json.dumps(row) + "\n")

print(f"Data saved in {json_path} using JSON Lines format.")
