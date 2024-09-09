import numpy as np
import pandas as pd
from faker import Faker 
import random 
import plotly.graph_objects as go
from itertools import combinations 
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans
import plotly.express as px 
import warnings 
from tqdm import tqdm 

# To avoid unnecessary warnings
warnings.filterwarnings("ignore", category = DeprecationWarning)

# to generate random items
np.random.seed(40)
random.seed(40)

def generate_data(num_destinations = 15, destinations = [], num_customers = 150, num_bookings = 100):

    # Giving the List of popular destnations rather than generating random/fake data
    bookings = []

    for _ in range(num_bookings):
        customer_id = random.randint(1, num_customers)
        destination_count = random.randint(1, 5)
        booking_by_customer = random.sample(destinations, destination_count)
        bookings.append({
            'customer_id' : customer_id,
            'destinations' : booking_by_customer
        })

    df = pd.DataFrame(bookings)
    # Explode the destinations list into separate rows
    df_encoded = df.explode('destinations').pivot_table(
        index = 'customer_id',
        columns = 'destinations',
        aggfunc = lambda x: 1,
        fill_value = 0
    )
    print(df)
    return df_encoded


# Apriori algorithm
def apriori(df_encoded, destinations, min_support = 0.1, min_conf = 0.5):
    def support(destination_group):
        return (df_encoded[list(destination_group)].sum(axis = 1) == len(destination_group)).mean()
    
    #classifing items
    destination_sets = [frozenset([dest]) for dest in destinations]
    rules = []

    # Making combinations to check whether there is an association between them
    for size in range(2, len(destinations) + 1):
        destination_sets = [comb for comb in combinations(destinations, size) if support(comb) >= min_support] 
    
        for destination_combo in destination_sets:
            destination_combo = frozenset(destination_combo)
            for i in range(1, len(destination_combo)):
                for antecedent in combinations(destination_combo, i):
                    antecedent = frozenset(antecedent)
                    consequent = destination_combo - antecedent
                    confidence = support(destination_combo) / support(consequent)
                    if confidence >= min_conf:
                        lift = confidence / support(consequent)
                        rules.append({
                            'antecedent' : ','.join(antecedent),
                            'consequent' : ','.join(consequent),
                            'support' : support(destination_combo),
                            'confidence' : confidence,
                            'lift' : lift
                        })
        
        if len(rules) >= 10:
            break 
    
    print(pd.DataFrame(rules).sort_values('lift', ascending = False))
    return pd.DataFrame(rules).sort_values('lift', ascending = False)


def visualize_apriori_rules(rules, top_n = 10):
    top_rules = rules.head(top_n)
    fig = px.scatter_3d(
        top_rules, x = "support", y = "confidence", z = "lift",
        color = "lift", size = "support",
        hover_name = "antecedent", hover_data = ["consequent"],
        labels = {"support": "support", "confidence": "confidence", "lift": "lift"},
        title = f"Top {top_n} Association Rules"
    )
    return fig


def main():
    print("Creating data....")

    destinations = ["Agra", "Jaipur", "Varanasi", "Kerala", "Goa", "Ladakh", "Udaipur", "Hampi", "Manali", "Darjeeling", "Ooty", "Kanya Kumari", "Kedharnadh", "Rameswaram"]

    df_encoded = generate_data(num_destinations = 15, destinations = destinations, num_customers = 50, num_bookings = 100)
    print("Creating data completed..")
    print(df_encoded)

    print("Performing Apripri Algo...")
    rules = apriori(df_encoded, destinations, min_support = 0.1, min_conf = 0.5)

    if not rules.empty:
        print(f"Apriori algo complete. And found {len(rules)} rules.")
        viz = visualize_apriori_rules(rules)
        viz.write_html("Apriori3D.html")
        print("Apriori rules saved as 'Apriori3D.html'.")
    else:
        print("apriori algo failed")


if __name__ == "__main__":
    main()