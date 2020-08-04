from tinydb import TinyDB
import pandas as pd
db=TinyDB("database/jav.json")
marked_db=TinyDB("database/jav_marked.json")

df=pd.DataFrame(db.all())
markeddf=pd.DataFrame(marked_db.all())

consolidated=pd.merge(df,markeddf,on="idx")
consolidated.to_csv("consolidated.csv",index=False)