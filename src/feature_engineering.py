import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Average monthly spend
    df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)

    # Long-term customer flag
    df["IsLongTermCustomer"] = (df["tenure"] > 24).astype(int)

    return df
