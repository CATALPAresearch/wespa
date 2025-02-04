import networkx as nx
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import pandas as pd

from .settings import *
from .util import prnt
from .util import print_all_output


class Collaboration_Graph:

    def get_moodle_author_id(author_id):
        # FixMe
        return author_id

    def get_cohesion_graph(self, df_cohesion_net, user_mapping='pseudomize'):
        """Create a cohesion graph"""
        # Extract unique students (authors)
        students = df_cohesion_net['author'].drop_duplicates().tolist()

        # Create adjacency matrix
        mat = np.zeros((len(students), len(students)), dtype=int)
        student_index = {student: i for i, student in enumerate(students)}

        # Fill the adjacency matrix
        for _, row in df_cohesion_net.iterrows():
            author = row['author']
            left_neighbor = row.get('left_neighbor', None)
            right_neighbor = row.get('right_neighbor', None)

            if pd.notna(left_neighbor) and left_neighbor != 0 and left_neighbor != author:
                row_index = student_index[author]
                col_index = student_index[left_neighbor]
                mat[row_index, col_index] += 1

            if pd.notna(right_neighbor) and right_neighbor != 0 and right_neighbor != author:
                print(right_neighbor, author, len(student_index))
                row_index = student_index[author]
                col_index = student_index[right_neighbor]
                mat[row_index, col_index] += 1

        # Map user IDs based on user_mapping type
        user_ids = (df_cohesion_net[['group', 'author']]
                    .drop_duplicates()
                    .sort_values('group'))

        if user_mapping == 'pseudomize':
            user_ids['user'] = range(1, len(user_ids) + 1)
        elif user_mapping == 'moodle':
            user_ids['user'] = user_ids['author'].apply(self.get_moodle_author_id)  
        else:
            user_ids['user'] = user_ids['author']

        user_ids = user_ids.drop(columns=['group'])
        student_mapping = dict(zip(user_ids['author'], user_ids['user']))

        # Update column and row names with mapped user IDs
        new_labels = [student_mapping.get(student, student) for student in students]
        
        # Create graph from adjacency matrix
        G = nx.from_numpy_array(mat, create_using=nx.DiGraph())
        
        mapping = {i: new_labels[i] for i in range(len(new_labels))}
        G = nx.relabel_nodes(G, mapping)

        return G


    def get_group_graph_measures(self, plotgr, the_group, the_week=0):
        # Check if the graph has 1 or fewer edges
        if len(plotgr.edges) <= 1:
            degrees = pd.DataFrame({
                'outdegree': [d for _, d in plotgr.out_degree()],
                'indegree': [d for _, d in plotgr.in_degree()]
            })

            return {
                'group': the_group,
                'week': the_week,
                'activeUsers': len(degrees),
                'isolatedUsers': 1,
                'usersProvideLotHelp': 0,
                'usersProvideAvgHelp': 0,
                'usersProvideLittleHelp': 0,
                'usersProvideNoHelp': 0,
                'usersReceiveLotHelp': 0,
                'usersReceiveAvgHelp': 0,
                'usersReceiveLittleHelp': 0,
                'usersReceiveNoHelp': 0,
                'closeCouples': 0,
                'avgCouples': 0,
                'loseCouples': 0,
                'density1': 0,
                'density2': 0
            }

        # Degree measures
        degrees = pd.DataFrame({
            'outdegree': dict(plotgr.out_degree()).values(),
            'indegree': dict(plotgr.in_degree()).values()
        })

        # Extract edge weights
        edge_weights = np.array([data['weight'] for _, _, data in plotgr.edges(data=True)])
        edge_weight_mean = np.mean(edge_weights)
        edge_weight_std = np.std(edge_weights)

        edge_weight_upper_threshold = np.percentile(edge_weights, 75)
        edge_weight_lower_threshold = np.percentile(edge_weights, 25)

        # Convert edge list into DataFrame
        edge_list = [(u, v, d['weight']) for u, v, d in plotgr.edges(data=True)]
        pairs = pd.DataFrame(edge_list, columns=['a', 'b', 'weight'])

        pairs['aboveThreshold'] = pairs['weight'] >= edge_weight_upper_threshold
        pairs['belowThreshold'] = pairs['weight'] < edge_weight_lower_threshold
        pairs['isCouple'] = pairs.apply(lambda row: ((row['b'], row['a']) in plotgr.edges), axis=1)

        # Help measures
        help_receiver = pairs.groupby('b')['weight'].sum().reset_index()
        help_receiver_upper_threshold = np.percentile(help_receiver['weight'], 75)
        help_receiver_lower_threshold = np.percentile(help_receiver['weight'], 25)

        help_giving = pairs.groupby('a')['weight'].sum().reset_index()
        help_giving_upper_threshold = np.percentile(help_giving['weight'], 75)
        help_giving_lower_threshold = np.percentile(help_giving['weight'], 25)

        # Isolated users: in-degree = 0 and out-degree = 0
        number_of_isolated_users = len(degrees[(degrees['indegree'] == 0) & (degrees['outdegree'] == 0)])

        # Users who received no help
        number_of_users_got_no_help = len(degrees[(degrees['indegree'] == 0) & (degrees['outdegree'] > 0)])

        # Users who provided no help
        number_of_users_not_helping = len(degrees[(degrees['indegree'] > 0) & (degrees['outdegree'] == 0)])

        # Couples based on mutual interactions
        close_couples = len(pairs[(pairs['isCouple']) & (pairs['aboveThreshold'])])
        avg_couples = len(pairs[(pairs['isCouple']) & (~pairs['aboveThreshold']) & (~pairs['belowThreshold'])])
        lose_couples = len(pairs[(pairs['isCouple']) & (pairs['belowThreshold'])])

        # Help analysis
        number_of_users_got_lot_help = len(help_receiver[help_receiver['weight'] > help_receiver_upper_threshold])
        number_of_users_got_avg_help = len(help_receiver[
            (help_receiver['weight'] < help_receiver_upper_threshold) & 
            (help_receiver['weight'] > help_receiver_lower_threshold)
        ])
        number_of_users_got_little_help = len(help_receiver[help_receiver['weight'] <= help_receiver_lower_threshold])

        number_of_users_give_lot_help = len(help_giving[help_giving['weight'] > help_giving_upper_threshold])
        number_of_users_give_avg_help = len(help_giving[
            (help_giving['weight'] < help_giving_upper_threshold) & 
            (help_giving['weight'] > help_giving_lower_threshold)
        ])
        number_of_users_give_little_help = len(help_giving[help_giving['weight'] <= help_giving_lower_threshold])

        # Network density
        density1 = nx.density(plotgr)
        density2 = nx.density(plotgr.to_undirected())

        return {
            'group': the_group,
            'week': the_week,
            'activeUsers': len(degrees),
            'isolatedUsers': number_of_isolated_users,
            'usersProvideLotHelp': number_of_users_give_lot_help,
            'usersProvideAvgHelp': number_of_users_give_avg_help,
            'usersProvideLittleHelp': number_of_users_give_little_help,
            'usersProvideNoHelp': number_of_users_not_helping,
            'usersReceiveLotHelp': number_of_users_got_lot_help,
            'usersReceiveAvgHelp': number_of_users_got_avg_help,
            'usersReceiveLittleHelp': number_of_users_got_little_help,
            'usersReceiveNoHelp': number_of_users_got_no_help,
            'closeCouples': close_couples,
            'avgCouples': avg_couples,
            'loseCouples': lose_couples,
            'density1': density1,
            'density2': density2
        }


    def plot_graph_network(self, plotgr, group_id):
        """
        Plots a network graph with custom styling, saving the output as a PDF.

        Parameters:
        - plotgr: NetworkX DiGraph (directed graph)
        - group_id: ID of the group
        - output_path: Path where the output PDF should be saved
        - semester: Semester identifier
        """
        global output_path, semester
        np.random.seed(1234)  # Set seed for reproducibility

        # Define layout
        pos = nx.spring_layout(plotgr, seed=1234)  # Equivalent to layout_nicely in igraph

        # Prepare plot figure
        plt.figure(figsize=(10, 10))

        # Draw nodes
        node_labels = {node: str(node) for node in plotgr.nodes}
        nx.draw_networkx_nodes(plotgr, pos, node_size=700, node_color="skyblue", edgecolors="black")
        nx.draw_networkx_labels(plotgr, pos, labels=node_labels, font_size=10, font_color="white")

        # Draw edges with weights
        edge_labels = {edge[:2]: edge[2]['weight'] for edge in plotgr.edges(data=True)}
        nx.draw_networkx_edges(plotgr, pos, edge_color="#cccccc", arrowstyle="-|>", arrowsize=10, width=1.5)
        nx.draw_networkx_edge_labels(plotgr, pos, edge_labels=edge_labels, font_color="blue", font_size=8)

        # Ensure output directory exists
        os.makedirs(os.path.join(output_path, "group-graphs"), exist_ok=True)

        # Save as PDF
        output_file = os.path.join(output_path, f"group-graphs/{project_name}-{semester}-05-group-graph-{group_id}.pdf")
        plt.savefig(output_file, format="pdf", bbox_inches="tight")

        # Show plot (optional)
        plt.show()

        print(f"Graph saved to {output_file}")


    def check_random_group(self, author_relations):
        """Manual check"""
        # Step 1: Get all unique groups
        all_groups = author_relations['group'].drop_duplicates().sort_values().tolist()
        # Step 2: Make a dry run for one group
        # Example:
        example_group = random.choice(all_groups)
        print('Example group '+ str(example_group))
        df_cohesion_net = author_relations[author_relations['group'] == example_group]
        print('Cohesion_net')
        print(df_cohesion_net)
        plotgr = self.get_cohesion_graph(df_cohesion_net)
        print('Graph')
        print(plotgr)
        #plot_graph_network(plotgr, example_group)
        measure = pd.DataFrame([self.get_group_graph_measures(plotgr, example_group)]).T
        print(measure)


    def create_graph_for_all_groups(self, author_relations, save_plot: False, save_output: False):
        """Iterate over all groups to build the graph and calculate the measures"""
        # Step 1: Get all unique groups
        all_groups = author_relations['group'].drop_duplicates().sort_values().tolist()
            
        # Step 2: Iterate over all groups and create plot, store plot, and compute graph measures
        graph_measures_list = []
        for i, group in enumerate(all_groups):
            prnt(f'Compute graph of group {group}: {i+1}/{len(all_groups)}')

            # Create graph and calculate measures
            df_cohesion_net = author_relations[author_relations['group'] == group]
            plotgr = self.get_cohesion_graph(df_cohesion_net)  # Generate cohesion graph

            # Store graph as plot
            if save_plot:
                self.plot_graph_network(plotgr, group)

            # Compute graph measures
            measure = self.get_group_graph_measures(plotgr, group)

            # Append results
            graph_measures_list.append(measure)

        # Prepare output
        graph_measures = pd.DataFrame(graph_measures_list)
        if save_output:
            output_file = os.path.join(output_path, f'{project_name}-{semester}-05-group-graph-measures.csv')
            graph_measures.to_csv(output_file, index=False)
            print(f"Graph measures saved to {output_file}")
        
        return graph_measures
    

    def create_json_graph_for_all_groups(self, author_relations, last_modified = 0, save_to_file=False):
        """Creates a file containing a JSON object describing the group cohesian graph"""
        target_week = "week-x"  # Change as needed

        # List of all groups
        all_groups = sorted(author_relations['group'].unique())

        # Process each group
        for group in all_groups:
            print(group)
            #if group > 4330:
            #    continue  # Skip groups greater than 4330
            
            g = self.get_cohesion_graph(author_relations[author_relations['group'] == group])
            
            nodes = [{'id': int(v), 'group': 1} for v in g.nodes()]
            
            edges = []
            if g.number_of_edges() > 0:
                for u, v, data in g.edges(data=True):
                    edges.append({
                        'source': int(u),
                        'target': int(v),
                        'value': float(data.get('weight', 1))
                    })
            
            output = {
                'last_modified': last_modified,
                'nodes': nodes,
                'links': edges
            }
            
            if save_to_file:
                output_dir = f"{output_path}/json/{target_week}/"
                os.makedirs(output_dir, exist_ok=True)
                json_file = os.path.join(output_dir, f"g{group}.json")
                
                with open(json_file, 'w') as f:
                    json.dump(output, f, indent=4)
                
                print(f"JSON file saved for group {group}: {json_file}")
            else:
                print(str(output))

