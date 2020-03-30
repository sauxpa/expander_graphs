import numpy as np
import pandas as pd
import sympy
import networkx as nx
from expanders import *

from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, Panel, Plot, Range1d, StaticLayoutProvider
from bokeh.models.widgets import Slider, Tabs
from bokeh.models.graphs import from_networkx
from bokeh.layouts import row, WidgetBox


def make_dataset(p):
    """Creates a ColumnDataSource object with data to plot.
    """
    builder = LPS3(p, remove_parallel_edges=True, remove_self_edges=True)
    builder.build()
    df = nx.to_pandas_edgelist(builder.G)
    # Convert dataframe to column data source#
    return ColumnDataSource(df)


def update(attr, old, new):
    """Update ColumnDataSource object.
    """
    # Change p to selected value
    pth = p_select.value
    p = sympy.prime(pth)

    # Create new graph
    new_src = make_dataset(p)

    # Update the data on the plot
    src.data.update(new_src.data)


    node_indices = list(range(p))

    # 1. First update layout
    circ = [i*2*np.pi/p for i in node_indices]
    x = [np.cos(i) for i in circ]
    y = [np.sin(i) for i in circ]
    graph_layout = dict(zip(node_indices, zip(x, y)))
    graph_renderer.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)

    # 2. Then update nodes and edges
    new_data_edge = {'start': src.data['source'], 'end': src.data['target']};
    # new_data_nodes = {'index': src.data['index']};
    new_data_nodes = {'index': node_indices};
    graph_renderer.edge_renderer.data_source.data = new_data_edge;
    graph_renderer.node_renderer.data_source.data = new_data_nodes;


# Slider to select target yields
p_select = Slider(start=2,
                  end=100,
                  step=1,
                  value=10,
                  title='n-th prime number'
                  )

# Update the plot when yields are changed
p_select.on_change('value', update)

pth = p_select.value
p = sympy.prime(pth)
src = make_dataset(p)

plot = Plot(plot_width=600,
            plot_height=600,
            x_range=Range1d(-1.1,1.1),
            y_range=Range1d(-1.1,1.1)
            )
plot.title.text = 'Lubotsky-Phillips-Sarnak 3-regular graphs'
G = nx.from_pandas_edgelist(pd.DataFrame(src.data))
graph_renderer = from_networkx(G, nx.circular_layout, scale=1, center=(0,0))
plot.renderers.append(graph_renderer)

controls = WidgetBox(p_select)

# Create a row layout
layout = row(controls, plot)

# Make a tab with the layout
tab = Panel(child=layout, title='LPS3')
### ALL TABS TOGETHER
tabs = Tabs(tabs=[tab])

curdoc().add_root(tabs)
