from graphviz import Digraph

def create_system_architecture_diagram(output_filename='system_architecture_diagram'):
    dot = Digraph('SystemArchitecture', format='svg')
    # Set graph attributes for a clean top-to-bottom layout with orthogonal edges
    dot.attr(rankdir='TB', size='8,12', splines='ortho', nodesep='1', ranksep='2')
    dot.attr('node', shape='rectangle', style='filled', fillcolor='#E0E5E8', fontname='Helvetica', fontsize='12')
    dot.attr('edge', fontname='Helvetica', fontsize='10')
    
    # Define nodes grouped by layers for clarity
    # Data Layer
    dot.node('DataSources', 'Data Sources\n(CSVs, DBs, Sensors)', shape='cylinder', fillcolor='#A3CEF1')
    dot.node('CNNOutput', 'CNN Output\n(Features)', shape='folder', fillcolor='#F6D55C')
    dot.node('RouteData', 'Route Data\n(Graphs, Maps)', shape='folder', fillcolor='#ED553B')
    dot.node('LSTMOutput', 'LSTM Output\n(Predictions)', shape='folder', fillcolor='#3CAEA3')
    
    # Models Layer
    dot.node('CNN', 'CNN Detection\n(YOLOv8)', fillcolor='#F6D55C')
    dot.node('LSTM', 'LSTM Module\n(Time Series Prediction)', fillcolor='#3CAEA3')
    dot.node('Pathfinding', 'Pathfinding Algorithms\n(Dijkstra, Yen)', fillcolor='#ED553B')
    dot.node('CatBoost', 'CatBoost Models\n(Prediction)', fillcolor='#173F5F')
    
    # Application Layer
    dot.node('BackendApp', 'Backend Application\n(Flask API)', fillcolor='#20639B')
    
    # User Interface Layer
    dot.node('UserInterface', 'User Interface\n(Web Pages)', shape='oval', fillcolor='#F4A261')
    
    # Define edges (data flow) with essential connections only for clarity
    dot.edge('DataSources', 'CNN', label='Raw Data')
    dot.edge('CNN', 'CNNOutput', label='Detected Features')
    dot.edge('CNNOutput', 'LSTM', label='Feature Data')
    dot.edge('LSTM', 'LSTMOutput', label='Predicted Traffic')
    dot.edge('LSTMOutput', 'BackendApp', label='Traffic Predictions')
    dot.edge('RouteData', 'Pathfinding', label='Graph Data')
    dot.edge('BackendApp', 'Pathfinding', label='Request Routes')
    dot.edge('Pathfinding', 'BackendApp', label='Optimized Routes')
    dot.edge('BackendApp', 'CatBoost', label='Request Predictions')
    dot.edge('CatBoost', 'BackendApp', label='Prediction Results')
    dot.edge('BackendApp', 'UserInterface', label='API Responses')
    dot.edge('UserInterface', 'BackendApp', label='User Requests')
    
    # Interconnections between models to show data dependencies clearly
    dot.edge('CNN', 'Pathfinding', label='Feature Data', style='dashed')
    dot.edge('LSTM', 'Pathfinding', label='Traffic Data', style='dashed')
    
    # Rank constraints to align nodes horizontally by layer
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('DataSources')
        s.node('CNNOutput')
        s.node('RouteData')
        s.node('LSTMOutput')
    
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('CNN')
        s.node('LSTM')
        s.node('Pathfinding')
        s.node('CatBoost')
    
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('BackendApp')
    
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('UserInterface')
    
    # Clusters for visual grouping
    with dot.subgraph(name='cluster_data') as c:
        c.attr(style='dashed', color='gray', label='Data')
        c.node('DataSources')
        c.node('CNNOutput')
        c.node('RouteData')
        c.node('LSTMOutput')
    
    with dot.subgraph(name='cluster_models') as c:
        c.attr(style='dashed', color='gray', label='Models')
        c.node('CNN')
        c.node('LSTM')
        c.node('Pathfinding')
        c.node('CatBoost')
    
    with dot.subgraph(name='cluster_app') as c:
        c.attr(style='dashed', color='gray', label='Application')
        c.node('BackendApp')
    
    with dot.subgraph(name='cluster_ui') as c:
        c.attr(style='dashed', color='gray', label='User Interface')
        c.node('UserInterface')
    
    # Render the diagram to file
    dot.render(output_filename, view=False)
    print(f"System architecture diagram generated: {output_filename}.svg")

    # Define nodes
    dot.node('DataSources', 'Data Sources\n(CSVs, DBs, Sensors)', shape='cylinder', fillcolor='#A3CEF1')
    dot.node('CNN', 'CNN Detection\n(YOLOv8)', fillcolor='#F6D55C')
    dot.node('CNNOutput', 'CNN Output\n(Features)', shape='folder', fillcolor='#F6D55C')
    dot.node('LSTM', 'LSTM Module\n(Time Series Prediction)', fillcolor='#3CAEA3')
    dot.node('LSTMOutput', 'LSTM Output\n(Predictions)', shape='folder', fillcolor='#3CAEA3')
    dot.node('Pathfinding', 'Pathfinding Algorithms\n(Dijkstra, Yen)', fillcolor='#ED553B')
    dot.node('RouteData', 'Route Data\n(Graphs, Maps)', shape='folder', fillcolor='#ED553B')
    dot.node('BackendApp', 'Backend Application\n(Flask API)', fillcolor='#20639B')
    dot.node('CatBoost', 'CatBoost Models\n(Prediction)', fillcolor='#173F5F')
    dot.node('UserInterface', 'User Interface\n(Web Pages)', shape='oval', fillcolor='#F4A261')

    # Define edges (data flow)
    dot.edge('DataSources', 'CNN', label='Raw Data')
    dot.edge('CNN', 'CNNOutput', label='Detected Features')
    dot.edge('CNNOutput', 'LSTM', label='Feature Data')
    dot.edge('LSTM', 'LSTMOutput', label='Predicted Traffic')
    dot.edge('LSTMOutput', 'BackendApp', label='Traffic Predictions')
    dot.edge('LSTM', 'Pathfinding', label='Traffic Data')
    dot.edge('CNN', 'Pathfinding', label='Feature Data')
    dot.edge('RouteData', 'Pathfinding', label='Graph Data')
    dot.edge('BackendApp', 'Pathfinding', label='Request Routes')
    dot.edge('Pathfinding', 'BackendApp', label='Optimized Routes')
    dot.edge('BackendApp', 'CatBoost', label='Request Predictions')
    dot.edge('CatBoost', 'BackendApp', label='Prediction Results')
    dot.edge('BackendApp', 'UserInterface', label='API Responses')
    dot.edge('UserInterface', 'BackendApp', label='User Requests')

    # Cluster for Models
    with dot.subgraph(name='cluster_models') as c:
        c.attr(style='dashed', color='gray', label='Models')
        c.node('CNN')
        c.node('LSTM')
        c.node('CatBoost')

    # Cluster for Data
    with dot.subgraph(name='cluster_data') as c:
        c.attr(style='dashed', color='gray', label='Data')
        c.node('DataSources')
        c.node('CNNOutput')
        c.node('LSTMOutput')
        c.node('RouteData')

    # Cluster for Application
    with dot.subgraph(name='cluster_app') as c:
        c.attr(style='dashed', color='gray', label='Application')
        c.node('BackendApp')
        c.node('Pathfinding')
        c.node('UserInterface')

    # Render the diagram to file
    dot.render(output_filename, view=False)
    print(f"System architecture diagram generated: {output_filename}.svg")

if __name__ == '__main__':
    create_system_architecture_diagram()
