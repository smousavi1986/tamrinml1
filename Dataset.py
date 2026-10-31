
# =====================================
# حل تمرین پیش پردازش داده ها
# =====================================

# مرحله 1: وارد کردن کتابخانه ها

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# مرحله 2: خواندن فایل CSV

df = pd.read_csv("data.csv")

print("اطلاعات اولیه:")
print(df.head())

print("تعداد سطر و ستون:")
print(df.shape)


# مرحله 3: بررسی داده های گمشده و تکراری

print("مقادیر گمشده:")
print(df.isnull().sum())

print("تعداد داده های تکراری:")
print(df.duplicated().sum())


# مرحله 4: تبدیل تاریخ ها

df["loan_start"] = pd.to_datetime(df["loan_start"])

df["loan_end"] = pd.to_datetime(df["loan_end"])


# مرحله 5: ایجاد متغیرهای جدید

# محاسبه مدت زمان وام به روز

df["duration_days"] = (
    df["loan_end"] - df["loan_start"]
).dt.days

# استخراج سال شروع وام

df["start_year"] = df["loan_start"].dt.year


# مرحله 6: حذف ستون های غیرضروری

df = df.drop(
    columns=[
        "client_id",
        "loan_id",
        "loan_start",
        "loan_end"
    ]
)


# مرحله 7: جدا کردن ویژگی ها و هدف

X = df.drop("repaid", axis=1)

y = df["repaid"]


# مرحله 8: تبدیل اطلاعات دسته ای به عددی

X = pd.get_dummies(
    X,
    columns=["loan_type"],
    dtype=int
)


# مرحله 9: تقسیم داده ها به آموزش و آزمون

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("تعداد داده های آموزش:", len(X_train))

print("تعداد داده های آزمون:", len(X_test))


# مرحله 10: تشخیص داده های پرت با روش IQR

Q1 = X_train["rate"].quantile(0.25)

Q3 = X_train["rate"].quantile(0.75)

IQR = Q3 - Q1

low = Q1 - 1.5 * IQR

high = Q3 + 1.5 * IQR

print("تعداد داده های پرت نرخ سود:")

print(
    ((X_train["rate"] < low) |
     (X_train["rate"] > high)).sum()
)


# محدود کردن داده های پرت

X_train.loc[:, "rate"] = X_train["rate"].clip(
    lower=low, upper=high
)

X_test.loc[:, "rate"] = X_test["rate"].clip(
    lower=low, upper=high
)


# مرحله 11: تبدیل لگاریتمی نرخ سود

X_train.loc[:, "rate"] = np.log1p(X_train["rate"])

X_test.loc[:, "rate"] = np.log1p(X_test["rate"])


# مرحله 12: مقیاس بندی متغیرهای عددی

columns = [
    "loan_amount",
    "duration_days",
    "start_year",
    "rate"
]

scaler = StandardScaler()

X_train.loc[:, columns] = scaler.fit_transform(
    X_train[columns]
)

X_test.loc[:, columns] = scaler.transform(
    X_test[columns]
)


# مرحله 13: ذخیره فایل های نهایی

train_data = X_train.copy()

train_data["repaid"] = y_train

test_data = X_test.copy()

test_data["repaid"] = y_test

train_data.to_csv(
    "train_preprocessed.csv",
    index=False
)

test_data.to_csv(
    "test_preprocessed.csv",
    index=False
)


# نمایش نتیجه

print("پیش پردازش با موفقیت انجام شد!")

print("ابعاد داده آموزش:", train_data.shape)

print("ابعاد داده آزمون:", test_data.shape)
