
from sqlalchemy import create_engine
import pandas as pd

# load data
engine = create_engine('sqlite:///../data/DisasterDB.db')
df = pd.read_sql_table('messages', engine)

print(df.drop(['id', 'message', 'original', 'genre'], axis=1).astype(int).sum(axis=0))