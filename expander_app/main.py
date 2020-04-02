import numpy as np
import pandas as pd
import sympy
import networkx as nx
from expanders import *

from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, Panel, Plot, Range1d, StaticLayoutProvider
from bokeh.models.widgets import Slider, Tabs, Div
from bokeh.models.graphs import from_networkx
from bokeh.layouts import row, WidgetBox

# LAYOUT_SHAPE = 'HEART'
# LAYOUT_SHAPE = 'FISH'
LAYOUT_SHAPE = ''


def x_layout(t):
    """Parametric coordinates of the graph layout, x coordinate.
    """
    if LAYOUT_SHAPE == 'HEART':
        return (np.sin(t)*np.sqrt(np.abs(np.cos(t)))/(np.sin(t)+7/5)-2*np.sin(t)+2) * np.cos(t)
    elif LAYOUT_SHAPE == 'FISH':
        return np.cos(t) - np.sin(t) ** 2 / np.sqrt(2)
    else:
        return np.cos(t)


def y_layout(t):
    """Parametric coordinates of the graph layout, y coordinate.
    """
    if LAYOUT_SHAPE == 'HEART':
        return (np.sin(t)*np.sqrt(np.abs(np.cos(t)))/(np.sin(t)+7/5)-2*np.sin(t)+2) * np.sin(t)
    elif LAYOUT_SHAPE == 'FISH':
        return np.cos(t) * np.sin(t)
    else:
        return np.sin(t)


def make_dataset_lps3(p, div_):
    """Creates a ColumnDataSource object with data to plot.
    """
    builder = LPS3(p, remove_parallel_edges=True, remove_self_edges=True)
    builder.build()
    df = nx.to_pandas_edgelist(builder.G)

    params_text = '<b>Parameters:</b><br><ul><li>p = {}</li> <li>Number of nodes = {}</li> <li>Number of edges = {}</li></ul>'.format(p, builder.G.number_of_nodes(), builder.G.number_of_edges())
    div_.text = params_text

    # Convert dataframe to column data source#
    return ColumnDataSource(df)


def make_dataset_paley(p, div_):
    """Creates a ColumnDataSource object with data to plot.
    """
    builder = Paley(p)
    builder.build()
    df = nx.to_pandas_edgelist(builder.G)

    params_text = '<b>Parameters:</b><br><ul><li>p = {}</li> <li>Number of nodes = {}</li> <li>Number of edges = {}</li></ul>'.format(p, builder.G.number_of_nodes(), builder.G.number_of_edges())
    div_.text = params_text

    # Convert dataframe to column data source#
    return ColumnDataSource(df)


def make_dataset_margulis(m, div_):
    """Creates a ColumnDataSource object with data to plot.
    """
    builder = Margulis(m)
    builder.build()
    df = nx.to_pandas_edgelist(builder.G)

    params_text = '<b>Parameters:</b><br><ul><li>n = {}</li> <li>Number of nodes = {}</li> <li>Number of edges = {}</li></ul>'.format(m, builder.G.number_of_nodes(), builder.G.number_of_edges())
    div_.text = params_text

    # Convert dataframe to column data source#
    return ColumnDataSource(df)


def update_lps3(attr, old, new):
    """Update ColumnDataSource object.
    """
    # Change p to selected value
    pth = p_select_lps3.value
    p = sympy.prime(pth)

    # Create new graph
    new_src = make_dataset_lps3(p, div_lps3)

    # Update the data on the plot
    src_lps3.data.update(new_src.data)

    node_indices = list(range(p))

    # 1. First update layout
    circ = [i*2*np.pi/p for i in node_indices]
    x = np.array([x_layout(t) for t in circ])
    y = np.array([y_layout(t) for t in circ])
    scale = np.max([x, y])
    x = (x - np.mean(x)) / (scale - np.mean(x))
    y = (y - np.mean(y)) / (scale - np.mean(y))

    graph_layout = dict(zip(node_indices, zip(x, y)))
    graph_renderer_lps3.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)

    # 2. Then update nodes and edges
    new_data_edge = {'start': src_lps3.data['source'], 'end': src_lps3.data['target']};
    # new_data_nodes = {'index': src.data['index']};
    new_data_nodes = {'index': node_indices};
    graph_renderer_lps3.edge_renderer.data_source.data = new_data_edge;
    graph_renderer_lps3.node_renderer.data_source.data = new_data_nodes;


def update_paley(attr, old, new):
    """Update ColumnDataSource object.
    """
    # Change p to selected value
    pth = p_select_paley.value
    p = eligible_primes_paley[pth-1]

    # Create new graph
    new_src = make_dataset_paley(p, div_paley)

    # Update the data on the plot
    src_paley.data.update(new_src.data)

    node_indices = list(range(p))

    # 1. First update layout
    circ = [i*2*np.pi/p for i in node_indices]
    x = np.array([x_layout(t) for t in circ])
    y = np.array([y_layout(t) for t in circ])
    scale = np.max([x, y])
    x = (x - np.mean(x)) / (scale - np.mean(x))
    y = (y - np.mean(y)) / (scale - np.mean(y))

    graph_layout = dict(zip(node_indices, zip(x, y)))
    graph_renderer_paley.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)

    # 2. Then update nodes and edges
    new_data_edge = {'start': src_paley.data['source'], 'end': src_paley.data['target']};
    # new_data_nodes = {'index': src.data['index']};
    new_data_nodes = {'index': node_indices};
    graph_renderer_paley.edge_renderer.data_source.data = new_data_edge;
    graph_renderer_paley.node_renderer.data_source.data = new_data_nodes;


def update_margulis(attr, old, new):
    """Update ColumnDataSource object.
    """
    # Change p to selected value
    m = m_select_margulis.value

    # Create new graph
    new_src = make_dataset_margulis(m, div_margulis)

    # Update the data on the plot
    src_margulis.data.update(new_src.data)

    node_indices = list(range(m**2))

    # 1. First update layout
    circ = [i*2*np.pi/(m**2) for i in node_indices]
    x = np.array([x_layout(t) for t in circ])
    y = np.array([y_layout(t) for t in circ])
    scale = np.max([x, y])
    x = (x - np.mean(x)) / (scale - np.mean(x))
    y = (y - np.mean(y)) / (scale - np.mean(y))

    graph_layout = dict(zip(node_indices, zip(x, y)))
    graph_renderer_margulis.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)

    # 2. Then update nodes and edges
    new_data_edge = {'start': src_margulis.data['source'], 'end': src_margulis.data['target']};
    new_data_nodes = {'index': node_indices};
    graph_renderer_margulis.edge_renderer.data_source.data = new_data_edge;
    graph_renderer_margulis.node_renderer.data_source.data = new_data_nodes;

######################################################################
###
### Lubotsky-Phillips-Sarnak 3-regular
###
######################################################################

# Slider to select p
p_select_lps3 = Slider(start=2,
                       end=200,
                       step=1,
                       value=10,
                       title='n-th prime number'
                       )

# Update the plot when graph is changed
p_select_lps3.on_change('value', update_lps3)

pth = p_select_lps3.value
p_lps3 = sympy.prime(pth)

div_lps3 = Div(text='<b>Parameters:</b><br>', width=200, height=100)

src_lps3 = make_dataset_lps3(p_lps3, div_lps3)

plot_lps3 = Plot(plot_width=600,
                 plot_height=600,
                 x_range=Range1d(-1.1,1.1),
                 y_range=Range1d(-1.1,1.1)
                 )

plot_lps3.title.text = 'Lubotsky-Phillips-Sarnak 3-regular graphs'
G_lps3 = nx.from_pandas_edgelist(pd.DataFrame(src_lps3.data))
graph_renderer_lps3 = from_networkx(G_lps3, nx.circular_layout, scale=1, center=(0,0))

node_indices_lps3 = list(range(p_lps3))
circ = [i*2*np.pi/p_lps3 for i in node_indices_lps3]
x = np.array([x_layout(t) for t in circ])
y = np.array([y_layout(t) for t in circ])
scale = np.max([x, y])
x = (x - np.mean(x)) / (scale - np.mean(x))
y = (y - np.mean(y)) / (scale - np.mean(y))

graph_layout = dict(zip(node_indices_lps3, zip(x, y)))
graph_renderer_lps3.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)

plot_lps3.renderers.append(graph_renderer_lps3)

controls_lps3 = WidgetBox(p_select_lps3, div_lps3)

# Create a row layout
layout_lps3 = row(controls_lps3, plot_lps3)

# Make a tab with the layout
tab_lps3 = Panel(child=layout_lps3, title='LPS3')


######################################################################
###
### Paley graph
###
######################################################################

eligible_primes_paley = [sympy.prime(i) for i in range(3, 36) if sympy.prime(i) % 4 == 1]

# Slider to select p
p_select_paley = Slider(start=1,
                        end=len(eligible_primes_paley)+1,
                        step=1,
                        value=3,
                        title='n-th Paley prime number'
                        )

# Update the plot when graph is changed
p_select_paley.on_change('value', update_paley)

pth = p_select_paley.value
p_paley = eligible_primes_paley[pth-1]

div_paley = Div(text='<b>Parameters:</b><br>', width=200, height=100)

src_paley = make_dataset_paley(p_paley, div_paley)

plot_paley = Plot(plot_width=600,
                  plot_height=600,
                  x_range=Range1d(-1.1,1.1),
                  y_range=Range1d(-1.1,1.1)
                  )

plot_paley.title.text = 'Paley dense expander graphs'
G_paley = nx.from_pandas_edgelist(pd.DataFrame(src_paley.data))
graph_renderer_paley = from_networkx(G_paley, nx.circular_layout, scale=1, center=(0,0))

node_indices_paley = list(range(p_paley))
circ = [i*2*np.pi/p_paley for i in node_indices_paley]
x = np.array([x_layout(t) for t in circ])
y = np.array([y_layout(t) for t in circ])
scale = np.max([x, y])
x = (x - np.mean(x)) / (scale - np.mean(x))
y = (y - np.mean(y)) / (scale - np.mean(y))

graph_layout = dict(zip(node_indices_paley, zip(x, y)))
graph_renderer_paley.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)

plot_paley.renderers.append(graph_renderer_paley)

controls_paley = WidgetBox(p_select_paley, div_paley)

# Create a row layout
layout_paley = row(controls_paley, plot_paley)

# Make a tab with the layout
tab_paley = Panel(child=layout_paley, title='Paley')


######################################################################
###
### Margulis 8-regular
###
######################################################################

# Slider to select p
m_select_margulis = Slider(start=2,
                           end=25,
                           step=1,
                           value=5,
                           title='Square root of the number of nodes'
                       )

# Update the plot when graph is changed
m_select_margulis.on_change('value', update_margulis)

m_margulis = m_select_margulis.value

div_margulis = Div(text='<b>Parameters:</b><br>', width=200, height=100)

src_margulis = make_dataset_margulis(m_margulis, div_margulis)

plot_margulis = Plot(plot_width=600,
                     plot_height=600,
                     x_range=Range1d(-1.1,1.1),
                     y_range=Range1d(-1.1,1.1)
                     )

plot_margulis.title.text = 'Margulis 8-regular graphs'
G_margulis = nx.from_pandas_edgelist(pd.DataFrame(src_margulis.data))
graph_renderer_margulis = from_networkx(G_margulis, nx.circular_layout, scale=1, center=(0,0))

node_indices_margulis = list(range(m_margulis ** 2))
circ = [i*2*np.pi/(m_margulis**2) for i in node_indices_margulis]
x = np.array([x_layout(t) for t in circ])
y = np.array([y_layout(t) for t in circ])
scale = np.max([x, y])
x = (x - np.mean(x)) / (scale - np.mean(x))
y = (y - np.mean(y)) / (scale - np.mean(y))

graph_layout = dict(zip(node_indices_margulis, zip(x, y)))
graph_renderer_margulis.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)

plot_margulis.renderers.append(graph_renderer_margulis)

controls_margulis = WidgetBox(m_select_margulis, div_margulis)

# Create a row layout
layout_margulis = row(controls_margulis, plot_margulis)

# Make a tab with the layout
tab_margulis = Panel(child=layout_margulis, title='Margulis')


### ALL TABS TOGETHER
tabs = Tabs(tabs=[tab_lps3, tab_paley, tab_margulis])

curdoc().add_root(tabs)
