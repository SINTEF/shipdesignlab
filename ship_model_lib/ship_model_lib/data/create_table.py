import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from scipy import interpolate

from ship_model_lib.types import ShipType

ship_type = ShipType.bulk_capesize
is_laden = True
suffix = "laden" if is_laden else "ballast"
new_column_name = f"{ship_type.value}_{suffix}"

try:
    df = pd.read_csv("drag_coefficient.csv", index_col=0)
except EmptyDataError:
    index = np.linspace(0, 180, 19)
    df = pd.DataFrame(index=index)
print(new_column_name)
df_temp = pd.read_csv("temp.csv")
value = df_temp["value"].values
# index = value[:int(value.size/2)]
# value = value[int(value.size/2):]
# print(index)
# print(value)
# value = interpolate.PchipInterpolator(index, value)(df.index)
interval = 180 / (value.size - 1)
if interval != 10 and interval > 0:
    value = interpolate.PchipInterpolator(
        np.arange(0, 180 + interval, interval), value
    )(df.index)
if interval > 0:
    df[new_column_name] = value
    df.to_csv("drag_coefficient.csv", index=True)
