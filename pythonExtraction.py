from sqlalchemy import create_engine, text
import json
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

# choose some type of style for the printouts

'''add database url'''
# If using default settings, it's \
# sqlite:///<path to config dir>/home-assistant_v2.db
DB_URL = "sqlite:///./home-assistant_v2.db"
engine = create_engine(DB_URL)


''' Test out pulling from the database by showing how many times entities make changes to the database'''
print(list(engine.execute("SELECT entity_id, COUNT(*) FROM states \
GROUP BY entity_id ")))

'''test out printing this to a pandas output'''
# pull entities from the database
entities = engine.execute("SELECT entity_id, COUNT(*) FROM states \
GROUP BY entity_id")
# format the entities into a pandas dataframe, note the columns are: entityName, #ofChangesMade
dataframe = pd.DataFrame(entities.fetchall())

# displaying the data as a bar chart
printPlot = ordered_indexed_df.plot(kind='bar', title='#ofChanges vs EntityName', figsize=(10, 10), legend=False)

# specifying labels for the X and Y axes
printPlot.set_xlabel('Number of Changes')
printPlot.set_ylabel('Entity name')