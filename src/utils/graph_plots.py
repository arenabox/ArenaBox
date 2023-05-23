import networkx as nx
import numpy as np
import plotly.graph_objects as go


def make_edge(dims, width=0.6, clr='#888'):
    if len(dims) == 2:
        return  go.Scatter(
                    x=dims[0],
                    y=dims[1],
                    line=dict(width=width,color=clr),
                    hoverinfo='none',
                    mode='lines')
    else:
        return go.Scatter3d(
            x=dims[0],
            y=dims[1],
            z=dims[2],
            line=dict(width=width, color=clr),
            hoverinfo='none',
            mode='lines')

def create_edge_trace(G, dim=2):
    edge_trace = []
    if dim==2:
        for edge in G.edges():
            wgt = 1
            clr = '#888'
            x0, y0 = G.nodes[edge[0]]['pos']
            x1, y1 = G.nodes[edge[1]]['pos']
            if 'color' in G[edge[0]][edge[1]]:
                clr = G[edge[0]][edge[1]]['color']
            if 'weight' in G[edge[0]][edge[1]]:
                wgt = G[edge[0]][edge[1]]['weight']*0.5
            dims = [[x0,x1,None], [y0,y1,None]]
            edge_trace.append(make_edge(dims, wgt,clr))
    else:
        for edge in G.edges():
            wgt = 0.6
            clr = '#888'
            x0, y0, z0 = G.nodes[edge[0]]['pos']
            x1, y1, z1 = G.nodes[edge[1]]['pos']
            if 'color' in G[edge[0]][edge[1]]:
                clr = G[edge[0]][edge[1]]['color']
            if 'weight' in G[edge[0]][edge[1]]:
                wgt = G[edge[0]][edge[1]]['weight'] * 0.5
            dims = [[x0, x1, None], [y0, y1, None], [z0, z1, None]]
            edge_trace.append(make_edge(dims, wgt, clr))
    return edge_trace
def create_node_trace(G, hover_text, node_weights, color, dim=2):
    node_x = []
    node_y = []
    node_z = []
    if dim==2:
        for node in G.nodes():
            x, y = G.nodes[node]['pos']
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            hovertext=hover_text,
            marker=dict(
                reversescale=True,
                color=color,
                size=node_weights,
                line_width=2))
    else:
        for node in G.nodes():
            x, y, z = G.nodes[node]['pos']
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)

        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            hoverinfo="text",
            hovertext=hover_text,
            # marker_color=colors,
            marker=dict(
                showscale=True,
                colorscale='Greens',
                reversescale=True,
                color=color,
                size=node_weights,
                line_width=2
            )
        )
    return node_trace

def create_network_graph(traces, title, name, subjects=None, dim=2):
    if dim==2:
        fig = go.Figure(data=traces,
                     layout=go.Layout(
                          paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        title=title,
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                     )
                    )
        #fig.show()
    else:
        fig = go.Figure(data=traces,
                        layout=go.Layout(
                            title=title,
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            scene=dict(
                                xaxis_title=subjects[0],
                                yaxis_title=subjects[1],
                                zaxis_title=subjects[2]),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="X Axis Title"),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="Y Axis Title"),
                        )
                        )
    fig.write_html(name, auto_open=True)


def create_random_graph(num_of_nodes, edge_data):
    G = nx.random_geometric_graph(num_of_nodes, 0.125)
    G.remove_edges_from(list(G.edges()))
    for src, tgts in edge_data.items():
        for tgt, wgt in tgts.items():
            G.add_edge(src,tgt, weight=wgt)
    return G
def create_pruned_graph(all_topics, node_data, edge_data, thres):
    G = nx.random_geometric_graph(len(all_topics), 0.125)
    G.remove_edges_from(list(G.edges()))
    node_list = list(all_topics.keys())
    node_colors = []
    node_weights = []
    for node in node_list:
        code=[255,255,255]
        pos = node_data[node][:3]
        node_colors.append(f"rgb{tuple(np.array(code*pos, dtype=int))}")
        node_weights.append(node_data[node][3])

    for src, tgts in edge_data.items():
        for tgt, wgt in tgts.items():
            if wgt>thres:
                G.add_edge(src,tgt, weight=wgt)
    return G, node_colors, node_weights


def create_within_community_graph(node_list, colab_info, edge_info, subjects, edgecolormap):

    G = nx.Graph()
    for node in node_list:
        G.add_node(node, pos=colab_info.loc[node, subjects].values)

    for src, tgts in edge_info.items():
        for tgt in tgts:
            if tgt == src or colab_info.loc[src, 'Class'] != colab_info.loc[tgt, 'Class']:
                continue
            G.add_edge(src, tgt, color=edgecolormap[colab_info.loc[src, 'Class']])

    return G

def create_collabotation_graph(node_list, colab_info, edge_info, subjects, edgecolormap):
    # Colaboration
    """
    name = 'network_plot/colab_network.html'
    title = '<br>Colaboration Network'
    """
    G = nx.Graph()
    for node in node_list:
        G.add_node(node, pos=colab_info.loc[node, subjects].values)

    for src, tgts in edge_info.items():
        for tgt in tgts:
            if tgt == src:
                continue
            G.add_edge(src, tgt, color=edgecolormap[colab_info.loc[src, 'Class']])

    return G


def create_collabotation_outside_form_graph(node_list, colab_info, edge_info, subjects, edgecolormap):
    # Colaboration outside community
    """
    name = 'network_plot/outside_form_colab_network.html'
    title = '<br>Colaboration outside own form'
    """

    G = nx.Graph()
    for node in node_list:
        G.add_node(node, pos=colab_info.loc[node, subjects].values)

    for src, tgts in edge_info.items():
        for tgt in tgts:
            if tgt == src or colab_info.loc[src, 'Class'] == colab_info.loc[tgt, 'Class']:
                continue
            G.add_edge(src, tgt, color=edgecolormap[colab_info.loc[src, 'Class']])


    return G


def create_new_discourse_graph(node_list, colab_info, edge_info, subjects, edgecolormap, id2class, class2color):
    # Classes based on discourse position
    # if respective discorsive position is < 0.4 then we say organisation has shifted its discourse
    """
    name = 'network_plot/new_discourse.html'
    title = '<br>Clusters based on discoursive positioning'
    """

    G = nx.Graph()
    node_colors = []
    for node in node_list:
        if colab_info.loc[node, id2class[colab_info.loc[node, 'Class']]] < 0.4:
            node_colors.append('yellow')
            colab_info.loc[node, 'Cluster'] = 3
        else:
            node_colors.append(class2color[colab_info.loc[node, 'Class']])
            colab_info.loc[node, 'Cluster'] = colab_info.loc[node, 'Class']
        G.add_node(node, pos=colab_info.loc[node, subjects].values)

    for src, tgts in edge_info.items():
        for tgt in tgts:
            if tgt == src or colab_info.loc[src, 'Cluster'] != colab_info.loc[tgt, 'Cluster']:
                continue
            G.add_edge(src, tgt, color=edgecolormap[colab_info.loc[src, 'Cluster']])


    return G


def create_new_cluster_graph(node_list, colab_info, edge_info, subjects, edgecolormap, id2class, class2color):
    # Colab with new cluster on discourse position : Authors/Organisations who left their discoursive position

    """
    name = 'network_plot/network_with_new_cluster.html'
    title = '<br>Links to new cluster based on discoursive positioning'
    """

    G = nx.Graph()
    node_colors = []
    for node in node_list:
        if colab_info.loc[node, id2class[colab_info.loc[node, 'Class']]] < 0.4:
            node_colors.append('yellow')
            colab_info.loc[node, 'Cluster'] = 3
        else:
            node_colors.append(class2color[colab_info.loc[node, 'Class']])
            colab_info.loc[node, 'Cluster'] = colab_info.loc[node, 'Class']
        G.add_node(node, pos=colab_info.loc[node, subjects].values, size=colab_info.loc[node, 'Vocab Count'])

    for src, tgts in edge_info.items():
        for tgt in tgts:
            if colab_info.loc[src, 'Cluster'] in [0, 1, 2] and colab_info.loc[tgt, 'Cluster'] == 3:
                G.add_edge(src, tgt, color=edgecolormap[colab_info.loc[src, 'Class']])

    return G

def create_graph_with_interstitial_cluster(node_list, colab_info, edge_info, subjects, edgecolormap, class2color):
    # Interstitial : if an organisation is range (0.33+-0.02, 0.33+-0.02, 0.33+-0.02)

    """
    name = 'network_plot/network_with_interstitial_cluster.html'
    title = '<br>Interstitial Cluster based on discoursive positioning'
    """

    G = nx.Graph()
    node_colors = []
    for node in node_list:
        if all(i > 0.31 and i < 0.35 for i in colab_info.loc[node, subjects].values):
            node_colors.append('purple')
            colab_info.loc[node, 'Cluster'] = 4
        else:
            node_colors.append(class2color[colab_info.loc[node, 'Class']])
            colab_info.loc[node, 'Cluster'] = colab_info.loc[node, 'Class']
        G.add_node(node, pos=colab_info.loc[node, subjects].values)

    for src, tgts in edge_info.items():
        for tgt in tgts:
            if tgt == src or colab_info.loc[src, 'Cluster'] != 4:
                continue
            G.add_edge(src, tgt, color=edgecolormap[colab_info.loc[src, 'Cluster']])

    return G