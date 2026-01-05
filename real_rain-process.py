import pandas as pd

def read_rain():
    
    # 读取气象站信息
    df_meta = pd.read_excel("data/climate/meta.xlsx",usecols=["F_站号","经度","纬度","高程","名称"])
    df_meta_dict = {
        row[1]["F_站号"]: {"经度":row[1]["经度"],"纬度":row[1]["纬度"],"高程":row[1]["高程"],"名称":row[1]["名称"]}
        for row in df_meta.iterrows()
    }

    # 读取降水数据
    df_rain = pd.read_excel("data/climate/rain.xlsx").query("year == 2021")
    df_rain["date"] = pd.to_datetime(
        df_rain[["year", "month", "day"]]
    )

    df_rain = df_rain.sort_values("date").reset_index(drop=True)

    df_rain_dict = {
        dot: df_rain[dot].to_numpy()
        for dot in df_rain.columns[3:]
    }

    return df_meta_dict, df_rain_dict
