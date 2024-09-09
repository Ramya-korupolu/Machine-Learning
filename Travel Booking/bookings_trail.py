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



def generate_data(num_destinations = 15, destinations = [], num_customers = 50, num_bookings = 100):

    # Giving the List of popular destnations rather than generating random/fake data
    bookings = []

    for _ in range(num_bookings):
        customer_id = random.randint(1, num_customers)
        destination_count = random.randint(1, 5)
        booking_by_customer = random.sample(destinations, destination_count)
        # accommodation = random.choice(accommodations)
        bookings.append({
            'customer_id' : customer_id,
            'destinations' : booking_by_customer
            # 'accommodations' : accommodation
        })

    df = pd.DataFrame(bookings)
    print(df)

    # Explode the destinations list into separate rows
    cust_dest_encoded = df.explode('destinations').pivot_table(
        index = 'customer_id',
        columns = 'destinations',
        aggfunc = lambda x: 1,
        fill_value = 0
    )

    return df, cust_dest_encoded



def add_accommodations(data, accommodations, num_customers = 50, num_bookings = 100):

    acc_bookings = []

    for _ in range(num_bookings):
        acc_bookings.append(random.choice(accommodations))
    
    # adding a new column of accommodations
    data['accommodation'] = acc_bookings
    print(data)

    # pivot table between destination and accommodation 
    dest_acc_encoded = data.explode('destinations').pivot_table(
        index = 'destinations',
        columns = 'accommodation',
        aggfunc = len,
        fill_value = 0
    )
    return dest_acc_encoded



# Apriori algorithm
def apriori(cust_dest_encoded, destinations, min_support = 0.1, min_conf = 0.5):
    def support(destination_group):
        return (cust_dest_encoded[list(destination_group)].sum(axis = 1) == len(destination_group)).mean()
    
    #classifing items
    destination_sets = [frozenset([dest]) for dest in destinations]
    destination_rules = []

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
                        destination_rules.append({
                            'antecedent' : ','.join(antecedent),
                            'consequent' : ','.join(consequent),
                            'support' : support(destination_combo),
                            'confidence' : confidence,
                            'lift' : lift
                        })
        
        if len(destination_rules) >= 10:
            break 
    
    # print(pd.DataFrame(destination_rules).sort_values('lift', ascending = False))
    return pd.DataFrame(destination_rules).sort_values('lift', ascending = False)



def apriori_rules_bt_acc_dest(dest_acc_encoded, destinations, accommodations, min_support = 0.1, min_conf = 0.5):

    # dest_acc_encoded = dest_acc_encoded.drop(columns=['customer_id'])
    dest_acc_rules = []

    # cheking that what the most likely accommodation for each destination
    most_likely_accommodation_count = dest_acc_encoded.max(axis = 1)
    # print(most_likely_accommodation_count)

    for index, row in dest_acc_encoded.iterrows():
        antecedent = index
        max_value = most_likely_accommodation_count[antecedent]
        consequent = [col[1] for col in row.index if row[col] == max_value]
        dest_acc_rules.append({
            'destination ' : antecedent,
            ' Most likely booked accomodation' : ','.join(consequent)
        })

    # print(pd.DataFrame(dest_acc_rules))
    
    return pd.DataFrame(dest_acc_rules)



def perform_kmeans(cust_dest_encoded, n_clusters = 3, update_intervals = 5):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(cust_dest_encoded)

    Kmeans = KMeans(n_clusters = n_clusters, random_state = 40, max_iter = 80)

    with tqdm(total = Kmeans.max_iter, desc = "K-Means Clustering..") as pbar:
        for i in range(Kmeans.max_iter):
            Kmeans.fit(cust_dest_encoded)
            pbar.update(1)

            if i % update_intervals == 0:
                yield Kmeans.labels_ 
            
            if Kmeans.n_iter_ <= i + 1:
                break 
    return Kmeans.labels_



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



def visualize_kmeans_clusters(cust_dest_encoded, cluster_labels):
    pca = PCA(n_components = 3)
    pca_result = pca.fit_transform(cust_dest_encoded)

    fig = px.scatter_3d(
        x = pca_result[:, 0], y = pca_result[:, 1], z = pca_result[:, 2],
        color = cluster_labels,
        title = "Travels Cluster Visualization"
    )
    return fig 



def main():
    print("Creating data....")

    destinations = ["Agra", "Jaipur", "Varanasi", "Kerala", "Goa", "Ladakh", "Udaipur", "Hampi", "Manali", "Darjeeling", "Ooty", "Kanya Kumari", "Kedharnadh", "Rameswaram"]
    accommodations = ["Hotel", "Resort", "Hostel", "Vacation Rental", "Guesthouse", "Homestay", "Motel", "Camping", "Eco-Lodge"]

    # generating data
    data, cust_dest_encoded = generate_data(num_destinations = 15, destinations = destinations, num_customers = 50, num_bookings = 100)
    print("Creating data completed..")
    print(cust_dest_encoded)

    print("Performing Apripri Algo to find association between destinations...")
    destination_rules = apriori(cust_dest_encoded, destinations, min_support = 0.1, min_conf = 0.5)
    print(destination_rules)

    print("Adding new column accommodations....")
    dest_acc_encoded = add_accommodations(data, accommodations, num_customers = 50, num_bookings = 100)
    print("Updating data completed..")
    print(dest_acc_encoded)

    print("Performing Apripri Algo to find the rules between destinations and accomodations...")
    dest_acc_rules = apriori_rules_bt_acc_dest(dest_acc_encoded, destinations, accommodations, min_support = 0.1, min_conf = 0.5)
    print(dest_acc_rules)
    

    if not destination_rules.empty:
        print(f"Apriori algo complete. And found {len(destination_rules)} rules.")
        viz = visualize_apriori_rules(destination_rules)
        viz.write_html("Apriori_dest_rules_3D.html")
        print("Apriori rules saved as 'Apriori_dest_rules_3D.html'.")
    else:
        print("apriori algo failed")

    print("Performing K-means")
    Kmeans_generator = perform_kmeans(cust_dest_encoded, n_clusters = 3, update_intervals = 5)
    print(Kmeans_generator)

    for i, labels in enumerate(Kmeans_generator):
        print(f"Kmeans iterator - {i * 5}")
        vizualize = visualize_kmeans_clusters(cust_dest_encoded, labels)
        vizualize.write_html(f"Traveler_cluster_3d_step_{i}.html")
        print(f"Intermediate visuals saved as traveler_clusters_3d_step_{i}")

    final_labels = labels
    print("K-means clustering complete.")

    final_viz = visualize_kmeans_clusters(cust_dest_encoded, labels)
    final_viz.write_html("Traveler_cluster_3d_final.html")
    print("Final traveler cluster saved.")

    print("Analysis completed")

if __name__ == "__main__":
    main()