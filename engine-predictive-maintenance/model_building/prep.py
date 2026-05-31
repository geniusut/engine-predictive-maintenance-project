
# =========================================
# 1. Import Required Libraries
# =========================================

# for data manipulation
import pandas as pd
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# =========================================
# 2. Hugging Face Authentication
# =========================================
print("=" * 60)
print("STEP 1: Hugging Face Authentication")
print("=" * 60)

HF_TOKEN = os.getenv("HF_TOKEN")

api = HfApi(token=HF_TOKEN)

print("Hugging Face authentication successful.\n")

# =========================================
# 3. Load Dataset from Hugging Face
# =========================================

print("=" * 60)
print("STEP 2: Loading Dataset from Hugging Face")
print("=" * 60)

DATASET_PATH = (
    "hf://datasets/geniusut/engine-predictive-maintenance/engine_data.csv"
)

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.\n")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Info:")
print(df.info())

# =========================================
# 4. Renaming and Standardizing Column Names
# =========================================

print("\n" + "=" * 60)
print("STEP 3: Renaming and Standardizing Column Names")
print("=" * 60)

df.columns = [
    'Engine_RPM',
    'Lub_Oil_Pressure',
    'Fuel_Pressure',
    'Coolant_Pressure',
    'Lub_Oil_Temperature',
    'Coolant_Temperature',
    'Engine_Condition'
]

print("Updated Column Names:")
print(df.columns)

# =========================================
# 5. Check Missing Values
# =========================================

print("\n" + "=" * 60)
print("STEP 4: Missing Value Check")
print("=" * 60)

missing_values = df.isnull().sum()

print(missing_values)

if missing_values.sum() == 0:
    print("\nNo missing values found.")
else:
    print("\nMissing values detected.")

# =========================================
# 6. Check Duplicate Records
# =========================================

print("\n" + "=" * 60)
print("STEP 5: Duplicate Record Check")
print("=" * 60)

duplicate_count = df.duplicated().sum()

print(f"Duplicate Records: {duplicate_count}")

if duplicate_count == 0:
    print("No duplicate records found.")
else:
    print("Duplicate records detected.")


# =========================================
# 7. Define Features and Target
# =========================================

print("\n" + "=" * 60)
print("STEP 6: Feature and Target Split")
print("=" * 60)

target_col = "Engine_Condition"

X = df.drop(target_col, axis=1)

y = df[target_col]

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)

# =========================================
# 8. Train-Test Split
# The dataset was divided into training and testing datasets using an 80:20 split ratio.
# Stratified sampling was applied during the split process to preserve the target class
# distribution across both training and testing datasets. This helps ensure balanced model evaluation
# despite moderate class imbalance within the dataset.
# =========================================

print("\n" + "=" * 60)
print("STEP 7: Train-Test Split")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train Feature Shape:", X_train.shape)
print("Test Feature Shape :", X_test.shape)

print("Train Target Shape :", y_train.shape)
print("Test Target Shape  :", y_test.shape)


# =========================================
# 9. Combine Features and Target
# =========================================

print("\n" + "=" * 60)
print("STEP 8: Combining Features and Target")
print("=" * 60)

train_df = X_train.copy()
train_df[target_col] = y_train

test_df = X_test.copy()
test_df[target_col] = y_test

print("Train Dataset Shape:", train_df.shape)
print("Test Dataset Shape :", test_df.shape)

# =========================================
# 10. Create Processed Data Folder
# =========================================

print("\n" + "=" * 60)
print("STEP 9: Creating Processed Data Folder")
print("=" * 60)

processed_path = "engine-predictive-maintenance/data/processed"

os.makedirs(processed_path, exist_ok=True)

print(f"Processed folder created at: {processed_path}")


# =========================================
# 11. Save Train and Test Datasets
# =========================================

print("\n" + "=" * 60)
print("STEP 10: Saving Processed Datasets")
print("=" * 60)

train_path = f"{processed_path}/train.csv"
test_path = f"{processed_path}/test.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"Train dataset saved at: {train_path}")
print(f"Test dataset saved at : {test_path}")


print("Files saved locally!")


# =========================================
# 12. Upload Processed Datasets to HF
# =========================================

print("\n" + "=" * 60)
print("STEP 11: Uploading Processed Datasets")
print("=" * 60)

repo_id = "geniusut/engine-predictive-maintenance"

files_to_upload = [
    train_path,
    test_path
]

for file_path in files_to_upload:

    print(f"Uploading: {file_path}")

    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id=repo_id,
        repo_type="dataset"
    )

    print(f"Uploaded successfully: {file_path}")

print("Train & Test datasets uploaded successfully!")
