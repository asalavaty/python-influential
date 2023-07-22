#! usr/bin/env python3

# Import the requirements
import pandas as pd
import numpy as np
import igraph
from random import seed
from plotnine import ggplot, aes, geoms, arrow, scale_color_cmap, scale_color_cmap_d, scale_color_manual, scale_fill_gradient, scale_size_identity, scale_x_continuous, scale_y_continuous, ggtitle, labs, guides, guide_legend, facet_wrap, scale_size_continuous, themes
from plotnine.utils import to_inches
from .ranNorm import rangeNormalize
from .rank import rank_cal
from functools import reduce
from copy import deepcopy

# =============================================================================
#
#    Visualization of a graph based on centrality measures
#
# =============================================================================

def cent_network_vis(
    graph: igraph.Graph,
    cent_metric: list,
    layout = "kk",
    node_color = "viridis",
    node_size_min = 3,
    node_size_max = 15,
    dist_power = 1,
    node_shape = "circle",
    stroke_size = 1.5,
    stroke_color = "identical",
    stroke_alpha = 0.6,
    show_labels = True,
    label_cex = 0.4,
    label_color = "black",
    directed = False,
    arrow_width = 25,
    arrow_length = 0.07,
    edge_width = 0.5,
    weighted = False,
    edge_width_min = 0.2,
    edge_width_max = 1,
    edge_color = "lightgrey",
    edge_linetype = "solid",
    legend_position = "right",
    legend_direction = "vertical",
    legend_title = "Centrality\nmeasure",
    boxed_legend = True,
    show_plot_title = True,
    plot_title = "Centrality Measure-based Network",
    title_position = "center",
    show_bottom_border = True,
    show_left_border = True,
    vis_seed = 1234):

    """

    This function has been developed for the visualization of a network based on
    applying a centrality measure to the size and color of network nodes. You are
    also able to adjust the directedness and weight of connections. Some of the documentations
    of the arguments of this function have been adapted from ggplot2 and igraph packages.
    A shiny app has also been developed for the calculation of IVI as well as IVI-based network
    visualization, which is accessible using the `influential::runShinyApp("IVI")` command.
    You can also access the shiny app online at https://influential.erc.monash.edu/.

    :param graph: the graph to be evaluated.
    :type graph: a graph (network) of the igraph class (igraph.Graph).

    :param cent_metric: A numeric list/array of the desired centrality measure previously
    calculated by any means. For example, you may use the function 'influential.centrality.ivi' for the calculation of 
    the Integrated Value of Influence (IVI) of network nodes. Please note that
    if the centrality measure has been calculated by any means other than the influential package, make
    sure that the order of the values in the cent_metric list is consistent with the order of vertices
    in the network (you can get the graph vertices uisng the command 'graph.vs').
    :type cent_metric: list

    :param layout: The layout to be used for organizing network nodes. Current available layouts include
    "auto", "bipartite", "circle", "dh", "drl", "drl_3d", "fr", "fr_3d", "grid", "grid_3d", "graphopt", "kk",
    "kk_3d", "lgl", "gem", "mds", "lgl", "mds", "random", "random_3d", "rt", "rt_circular", "sphere", 
    "star" and "sugiyama" (default is set to "kk"). For a complete description of different layouts and their
    underlying algorithms please refer to the module igraph.layout.
    :type layout: str

    :param node_color: a standard Matplotlib colormap name. The default is `viridis`. For the list of names checkout the 
    output of ``matplotlib.cm.cmaps_listed.keys()`` or see the `documentation <http://matplotlib.org/users/colormaps.html>`_.
    :type node_color: str

    :param node_size_min: the size of nodes with the lowest value of the centrality measure (default is set to 3).
    :type node_size_min: int

    :param node_size_max: the size of nodes with the highest value of the centrality measure (default is set to 15).
    :type node_size_max: int

    :param dist_power: the power to be used to visualize more distinction between nodes with high and low
    centrality measure values. The higher the power, the smaller the nodes with lower values of the centrality
    measure will become. Default is set to 1, meaning the relative sizes of nodes are reflective of their
    actual centrality measure values.
    :type dist_power: float

    :param node_shape: the shape of nodes. Current available shapes include "circle",
    "square", "diamond", "triangle", and "inverted triangle" (default is set to "circle"). You can also
    set different shapes to different groups of nodes by providing a list of shapes of nodes with
    the same length and order of network nodes. This is useful when plotting a network that include different
    types of nodes (for example, up- and down-regulated features).
    :type node_shape: str or list

    :param stroke_size: the size of stroke (border) around the nodes (default is set to 1.5).
    :type stroke_size: float

    :param stroke_color: the color of stroke (border) around the nodes (default is set to "identical" meaning that the
    stroke color of a node will be identical to its corresponding node color). You can also
    set different colors to different groups of nodes by providing a character vector of colors of nodes with
    the same length and order of network nodes. This is useful when plotting a network that include different
    type of node (for example, up- and down-regulated features).
    :type stroke_color: str or list

    :param stroke_alpha: the transparency of the stroke (border) around the nodes which should
    be a number between 0 and 1 (default is set to 0.6).
    :type stroke_alpha: float

    :param show_labels: whether to show node labels or not (default is set to True).
    :type show_labels: bool

    :param label_cex: the amount by which node labels should be scaled relative to the node sizes (default is set to 0.4).
    :type label_cex: float

    :param label_color: the color of node labels (default is set to "black").
    :type label_color: str

    :param directed: whether to draw the network as directed or not (default is set to False).
    :type directed: bool

    :param arrow_width: the width of arrows in the case the network is directed (default is set to 25).
    :type arrow_width: float

    :param arrow_length: the length of arrows in inch in the case the network is directed (default is set to 0.07).
    :type arrow_length: float

    :param edge_width: the constant width of edges if the network is unweighted (default is set to 0.5).
    :type edge_width: float

    :param weighted: whether the network is a weighted network or not (default is set to False).
    :type weighted: bool
    
    :param edge_width_min: the width of edges with the lowest weight (default is set to 0.2).
    This parameter is ignored for unweighted networks.
    :type edge_width_min: float
    
    :param edge_width_max: the width of edges with the highest weight (default is set to 1).
    This parameter is ignored for unweighted networks.
    :type edge_width_min: float

    :param edge_color: the color of edges (default is set to "lightgrey").
    :type edge_color: str

    :param edge_linetype: the line type of edges. Current available linetypes include
    "twodash", "longdash", "dotdash", "dotted", "dashed", and "solid" (default is set to "solid").
    :type edge_linetype: str

    :param legend_position: The position of legends ("none", "left", "right",
    "bottom", "top", or two-element numeric vector). The default is set to "right".
    :type legend_position: str

    :param legend_direction: layout of items in legends ("horizontal" or "vertical").
    The default is set to "vertical".
    :type legend_direction: str

    :param legend_title: the legend title in the string format (default is set to "Centrality measure").
    :type legend_title: str

    :param boxed_legend: whether to draw a box around the legend or not (default is set to TRUE).
    :type boxed_legend: bool

    :param show_plot_title: whether to show the plot title or not (default is set to True).
    :type show_plot_title: bool

    :param plot_title: the plot title in the string format (default is set to "Centrality Measure-based Network").
    :type plot_title: str

    :param title_position: the position of title ("left", "center", or "right"). The default is set to "center".
    :type title_position: str

    :param show_bottom_border: whether to draw the bottom border line (default is set to True).
    :type show_bottom_border: bool

    :param show_left_border: whether to draw the left border line (default is set to True).
    :type show_left_border: bool

    :param vis_seed: a single value, interpreted as an integer to be used for random number generation for preparing
    the network layout (default is set to 1234).
    :type vis_seed: int

    :return: a plot with the class ggplot

    """

    # Preparing the layout and plotcord table
    seed(vis_seed)
    graph_layout = graph.layout(layout)
    plotcord = pd.DataFrame({
        'x': [graph_layout[i][0] for i in range(len(graph_layout))],
        'y': [graph_layout[i][1] for i in range(len(graph_layout))]
    })

    ## Correct the indices (row names)
    plotcord = plotcord.set_axis(graph.vs['name'], axis=0)
    
    ## Correct the column names
    plotcord = plotcord.set_axis(['X', 'Y'], axis=1)

    ## Add the centrality measure
    plotcord = plotcord.assign(cent_metric = cent_metric)

    ## Range normalize the node size based on the centrality measure
    plotcord = plotcord.assign(node_size = rangeNormalize(data = list(map(lambda x: x ** dist_power, list(plotcord['cent_metric']))), 
    minimum = node_size_min, maximum = node_size_max))

    ## Add the node names
    plotcord = plotcord.assign(node_name = list(plotcord.index))

    #****************************#

    # Get the edges (pairs of node IDs)
    edgelist = graph.get_edgelist()

    # Prepare a four column edge dataframe with source and destination coordinates
    edges_cord = pd.DataFrame()

    for i in edgelist:
        edges_cord = pd.concat([edges_cord, 
        pd.DataFrame({
            'X1': plotcord.iloc[i[0], 0],
            'Y1': plotcord.iloc[i[0], 1],
            'X2': plotcord.iloc[i[1], 0],
            'Y2': plotcord.iloc[i[1], 1]
        }, index= [0])], axis=0, ignore_index= True)

    #****************************#

    # Refine end positions of edges for directerd networks
    if directed:
        for i in range(edges_cord.shape[0]):

            ## correct the x coordinate of arrow
            if edges_cord['X1'][i] > edges_cord['X2'][i]:
                edges_cord['X2'][i] = edges_cord['X2'][i] + 0.115
            elif edges_cord['X1'][i] < edges_cord['X2'][i]:
                edges_cord['X2'][i] = edges_cord['X2'][i] - 0.115
            
            # correct the y coordinate of arrow
            if edges_cord['Y1'][i] > edges_cord['Y2'][i]:
                edges_cord['Y2'][i] = edges_cord['Y2'][i] + 0.115
            elif edges_cord['Y1'][i] < edges_cord['Y2'][i]:
                edges_cord['Y2'][i] = edges_cord['Y2'][i] - 0.115

    #****************************#

    # Set the edge width
    if weighted:
        edges_cord.insert(Weight = graph.es['weight'])

        ## range normalize the weight
        edges_cord['Weight'] = rangeNormalize(data = edges_cord['Weight'], minimum = edge_width_min, maximum = edge_width_max)

    #****************************#
    
    # Draw the plot
    tmp_plot = ggplot(data = plotcord) + aes(x = 'X', y = 'Y')

    ## Add the edges
    if directed:
        tmp_plot = tmp_plot + geoms.geom_segment(aes(x='X1', y='Y1', xend = 'X2', yend = 'Y2'), 
        data = edges_cord, 
        size = edges_cord['Weight'] if weighted else edge_width, 
        arrow =  arrow(angle = arrow_width, length = to_inches(arrow_length, 'in'), type = 'closed'),
        colour = edge_color,
        linetype = edge_linetype
        )
    else:
        tmp_plot = tmp_plot + geoms.geom_segment(aes(x='X1', y='Y1', xend = 'X2', yend = 'Y2'), 
        data = edges_cord, 
        size = edges_cord['Weight'] if weighted else edge_width,
        colour = edge_color,
        linetype = edge_linetype
        )

    #****************************#

    # Add nodes

    ## Define the node shape
    if node_shape == "circle":
        node_shape = 'o'
    elif node_shape == "square":
        node_shape = 's'
    elif node_shape == "diamond":
         node_shape = 'D'
    elif node_shape == "triangle":
         node_shape = '^'
    elif node_shape == "inverted triangle":
         node_shape = 'v'

    ## Add stroke color
    if stroke_color == 'identical':
        tmp_plot = tmp_plot + geoms.geom_point(aes(x = 'X', y = 'Y', color = cent_metric), 
                                               data = plotcord, 
                                               shape = node_shape,
                                               size = list(plotcord['node_size']),
                                               stroke = stroke_size,
                                               alpha = stroke_alpha,
                                               show_legend = False) + scale_color_cmap(cmap_name = node_color)
    
    else:
        tmp_plot = tmp_plot + geoms.geom_point(aes(x = 'X', y = 'Y'), 
                                               data = plotcord, 
                                               shape = node_shape,
                                               color = stroke_color,
                                               size = list(plotcord['node_size']),
                                               stroke = stroke_size,
                                               alpha = stroke_alpha,
                                               show_legend = False)
        
    ## Add node objects and their colors
    tmp_plot = tmp_plot + geoms.geom_point(aes(x = 'X', y = 'Y', fill = cent_metric),
                                           data = plotcord,
                                           shape = node_shape,
                                           stroke = 0,
                                           size = list(plotcord['node_size'])) + scale_color_cmap(cmap_name = node_color)
        
    ## Add node labels
    if show_labels:
        tmp_plot = tmp_plot + geoms.geom_text(aes(x = 'X', y = 'Y', label = list(plotcord['node_name'])),
                                              data = plotcord,
                                              size = list(plotcord['node_size']*label_cex),
                                              color = label_color)
        
    #****************************#
        
    # Expand the x and y limits
    tmp_plot = tmp_plot + scale_x_continuous(expand=(0,1)) + scale_y_continuous(expand=(0,1))

    #****************************#

    # Add theme elements

    ## Add main plot theme
    tmp_plot = tmp_plot + themes.theme_void() + themes.theme(

        ## Add legend specifications
        legend_position = legend_position,
        legend_direction = legend_direction,
        legend_title_align='center',
        panel_border = themes.element_blank()
        ) + labs(fill = legend_title)
    
    ## Add title
    if show_plot_title:

        ## Define title position
        if title_position == "left":
            title_position = 0
        elif title_position == "center":
            title_position = 0.5
        elif title_position == "right":
            title_position = 1
        
        tmp_plot = tmp_plot +  ggtitle(plot_title) + themes.theme(plot_title = themes.element_text(hjust = title_position))

    if boxed_legend:
        tmp_plot = tmp_plot + themes.theme(
            legend_position=legend_position,
            legend_direction=legend_direction,
            legend_title_align='center',
            legend_box_margin=5,
            legend_background=themes.element_rect(color='black', size=0.5),
            legend_box=legend_direction
            )
        
    
    ## Add border lines
    if show_bottom_border:
      tmp_plot = tmp_plot + themes.theme(axis_line_x = themes.element_line(color = 'black'))

    if show_left_border:
      tmp_plot = tmp_plot + themes.theme(axis_line_y = themes.element_line(color = 'black'))

    return tmp_plot

# =============================================================================
#
#    Visualization of ExIR results
#
# =============================================================================

def exir_vis(exir_results,
             synonyms_table = None,
             n = 10,
             driver_type = "combined",
             biomarker_type = "combined",
             show_drivers = True,
             show_biomarkers = True,
             show_de_mediators = True,
             show_nonDE_mediators = True,
             basis = "Rank",
             nrow = 1,
             dot_size_min = 1,
             dot_size_max = 3,
             type_color = "viridis",
             stroke_size = 1.5,
             stroke_alpha = 1,
             dot_color_low = "blue",
             dot_color_high = "red",
             legend_position = "right",
             legend_direction = "vertical",
             boxed_legend = True,
             show_plot_title = True,
             plot_title = "auto",
             title_position = "left",
             plot_title_size = 12,
             show_plot_subtitle = True,
             plot_subtitle = "auto",
             y_axis_title = "Feature",
             show_y_axis_grid = True):
    
    """

    This function has been developed for the visualization of ExIR results. Some of the documentations
    of the arguments of this function have been adapted from ggplot2 R package.
    A shiny app has also been developed for Running the ExIR model, visualization of its results as well as computational
    simulation of knockout and/or up-regulation of its top candidate outputs, which is accessible online at https://influential.erc.monash.edu/.

    :param exir_results: output of the function `exir`.
    :type exir_results: dict

    :param synonyms_table: (optional) a pandas data frame with two columns including a column for the used feature
    names in the input data of the "exir" model and the other column their synonyms. Note, the original feature names should
    always come on the first column and the synonyms on the second one. For example, if
    the original feature names used for running the "exir" model are Ensembl gene
    symbols, you can use their HGNC synonyms in the second column to be used for the visualization of the ExIR results.
    :type synonyms_table: pandas.core.frame.DataFrame

    :param n: an integer specifying the number of top candidates to be selected from each category of ExIR results (default is set to 10).
    :type n: int

    :param driver_type: a string specifying the type of drivers to be used for the selection of top N candidates. The possible types
    include "combined" (meaning both driver types), "accelerator" and "decelerator" (default is set to "combined").
    :type driver_type: str

    :param biomarker_type: A string specifying the type of biomarkers to be used for the selection of top N candidates. Possible types
    include "combined" (meaning both biomarker types), "up-regulated" and "down-regulated" (default is set to "combined").
    :type biomarker_type: str

    :param show_drivers: whether to show Drivers or not (default is set to True).
    :type show_drivers: bool

    :param show_biomarkers: whether to show Biomarkers or not (default is set to True).
    :type show_biomarkers: bool

    :param show_de_mediators: whether to show DE-mediators or not (default is set to True).
    :type show_de_mediators: bool

    :param show_nonDE_mediators: whether to show nonDE-mediators or not (default is set to True).
    :type show_nonDE_mediators: bool

    :param basis: a string specifying the basis for the selection of top N candidates from each category of the results. Possible options include
    "Rank" and "Adjusted p-value" (default is set to "Rank").
    :type basis: str

    :param nrow: number of rows of the plot (default is set to 1).
    :type nrow: int

    :param dot_size_min: the size of dots with the lowest statistical significance (default is set to 1).
    :type dot_size_min: foat

    :param dot_size_max: the size of dots with the highest statistical significance (default is set to 3).
    :type dot_size_max: foat

    :param type_color: a string indicating the color palette to be used for the visualization of
    different types of candidates. You may choose one of the Viridis palettes including 'magma', 'inferno', 'plasma', 
    'viridis', 'cividis', 'twilight', 'twilight_shifted', 'turbo', or manually specify the vector of colors for different types.
    :type type_color: str or list

    :param stroke_size: the size of stroke (border) around the dots (default is set to 1.5).
    :type stroke_size: foat

    :param stroke_alpha: the transparency of the stroke (border) around the dots which should
    be a number between 0 and 1 (default is set to 1).
    :type stroke_alpha: foat

    :param dot_color_low: the color to be used for the visualization of dots (features) with the lowest Z-score values (default is set to "blue").
    :type dot_color_low: str

    :param dot_color_high: The color to be used for the visualization of dots (features) with the highest Z-score values (default is set to "red").
    :type dot_color_high: str

    :param legend_position: the position of legends ("none", "left", "right",
    "bottom", "top"). The default is set to "bottom".
    :type legend_position: str

    :param legend_direction: layout of items in legends ("horizontal" or "vertical").
    The default is set to "vertical".
    :type legend_direction: str
    
    :param boxed_legend: whether to draw a box around the legend or not (default is set to True).
    :type boxed_legend: bool

    :param show_plot_title: whether to show the plot title or not (default is set to True).
    :type show_plot_title: bool

    :param plot_title: the plot title in the string format (default is set to "auto" which automatically generates a title for the plot).
    :type plot_title: str

    :param title_position: the position of title and subtitle ("left", "center", or "right"). The default is set to "left".
    :type title_position: str

    :param plot_title_size: the font size of the plot title (default is set to 12).
    :type plot_title_size: float

    :param show_plot_subtitle: whether to show the plot subtitle or not (default is set to True).
    :type show_plot_subtitle: bool

    :param plot_subtitle: the plot subtitle in the string format (default is set to "auto" which automatically generates a subtitle for the plot).
    :type plot_subtitle: str

    :param y_axis_title: the title of the y axis (features title). Default is set to "Features".
    :type y_axis_title: str

    :param show_y_axis_grid: whether to draw y axis grid lines (default is set to True).
    :type show_y_axis_grid: bool

    :return: a plot with the class `plotnine.ggplot.ggplot`.

    """

    # Prepare a list for storing the results
    exir_for_plot = list()

    # Select top N features

    ## for driver table
    if any(list(map(lambda x: x == "Driver table", list(exir_results.keys())))):
        top_N_driver_table = exir_results['Driver table']
        top_N_driver_table['Feature'] = list(top_N_driver_table.index)
        top_N_driver_table['Class'] = "Driver"

        # top N combined
        if driver_type == "combined" and pd.DataFrame(top_N_driver_table).shape[0] > n:

            if basis == "Rank":
                top_drivers_index = list(np.argsort(top_N_driver_table.Rank)[:n])
            elif basis == "Adjusted p-value":
                top_drivers_index = list(np.argsort(top_N_driver_table.P_adj)[:n])

            top_N_driver_table = top_N_driver_table.iloc[top_drivers_index]
            top_N_driver_table.Rank = rank_cal(top_N_driver_table.Rank)
        
        # top N accelerator
        elif driver_type == "accelerator":
            top_N_driver_table = top_N_driver_table[top_N_driver_table.Type == 'Accelerator']

            if basis == 'Rank':
                top_drivers_index = list(np.argsort(top_N_driver_table.Rank)[:n])
            elif basis == 'Adjusted p-value':
                top_drivers_index = list(np.argsort(top_N_driver_table.P_adj)[:n])

            top_N_driver_table = top_N_driver_table.iloc[top_drivers_index]
            top_N_driver_table.Rank = rank_cal(top_N_driver_table.Rank)
        
        # top N decelerator
        elif driver_type == "decelerator":
            top_N_driver_table = top_N_driver_table[top_N_driver_table.Type == 'Decelerator']

            if basis == 'Rank':
                top_drivers_index = list(np.argsort(top_N_driver_table.Rank)[:n])
            elif basis == 'Adjusted p-value':
                top_drivers_index = list(np.argsort(top_N_driver_table.P_adj)[:n])

            top_N_driver_table = top_N_driver_table.iloc[top_drivers_index]
            top_N_driver_table.Rank = rank_cal(top_N_driver_table.Rank)
        
        top_N_driver_table.Type[top_N_driver_table.Type == "Accelerator"] = "Accelerator\ndriver"
        top_N_driver_table.Type[top_N_driver_table.Type == "Decelerator"] = "Decelerator\ndriver"
        exir_for_plot.append(top_N_driver_table)

    ####********************####

    ## for biomarker table
    if any(list(map(lambda x: x == "Biomarker table", list(exir_results.keys())))):
        top_N_biomarker_table = exir_results['Biomarker table']
        top_N_biomarker_table['Feature'] = list(top_N_biomarker_table.index)
        top_N_biomarker_table['Class'] = "Biomarker"

        # top N combined
        if biomarker_type == "combined" and pd.DataFrame(top_N_biomarker_table).shape[0] > n:

            if basis == "Rank":
                top_biomarkers_index = list(np.argsort(top_N_biomarker_table.Rank)[:n])
            elif basis == "Adjusted p-value":
                top_biomarkers_index = list(np.argsort(top_N_biomarker_table.P_adj)[:n])

            top_N_biomarker_table = top_N_biomarker_table.iloc[top_biomarkers_index]
            top_N_biomarker_table.Rank = rank_cal(top_N_biomarker_table.Rank)
        
        # top N up-regulated
        elif biomarker_type == "up-regulated":
            top_N_biomarker_table = top_N_biomarker_table[top_N_biomarker_table.Type == 'Up-regulated']

            if basis == 'Rank':
                top_biomarkers_index = list(np.argsort(top_N_biomarker_table.Rank)[:n])
            elif basis == 'Adjusted p-value':
                top_biomarkers_index = list(np.argsort(top_N_biomarker_table.P_adj)[:n])

            top_N_biomarker_table = top_N_biomarker_table.iloc[top_biomarkers_index]
            top_N_biomarker_table.Rank = rank_cal(top_N_biomarker_table.Rank)
        
        # top N down-regulated
        elif biomarker_type == "down-regulated":
            top_N_biomarker_table = top_N_biomarker_table[top_N_biomarker_table.Type == 'Down-regulated']

            if basis == 'Rank':
                top_biomarkers_index = list(np.argsort(top_N_biomarker_table.Rank)[:n])
            elif basis == 'Adjusted p-value':
                top_biomarkers_index = list(np.argsort(top_N_biomarker_table.P_adj)[:n])

            top_N_biomarker_table = top_N_biomarker_table.iloc[top_biomarkers_index]
            top_N_biomarker_table.Rank = rank_cal(top_N_biomarker_table.Rank)
        
        top_N_biomarker_table.Type[top_N_biomarker_table.Type == "Up-regulated"] = "Up-regulated\nbiomarker"
        top_N_biomarker_table.Type[top_N_biomarker_table.Type == "Down-regulated"] = "Down-regulated\nbiomarker"
        exir_for_plot.append(top_N_biomarker_table)
    
    ####********************####

    ## for nonDE-mediator table
    
    if any(list(map(lambda x: x == "nonDE-mediator table", list(exir_results.keys())))):
        top_N_nonDE_mediator_table = exir_results['nonDE-mediator table']
        top_N_nonDE_mediator_table['Feature'] = list(top_N_nonDE_mediator_table.index)
        top_N_nonDE_mediator_table['Class'] = "nonDE-mediator"

        # top N
        if basis == "Rank":
            top_nonDE_mediators_index = list(np.argsort(top_N_nonDE_mediator_table.Rank)[:n])
        elif basis == "Adjusted p-value":
            top_nonDE_mediators_index = list(np.argsort(top_N_nonDE_mediator_table.P_adj)[:n])

        top_N_nonDE_mediator_table = top_N_nonDE_mediator_table.iloc[top_nonDE_mediators_index]
        top_N_nonDE_mediator_table.Rank = rank_cal(top_N_nonDE_mediator_table.Rank)

        exir_for_plot.append(top_N_nonDE_mediator_table)

    ####********************####

    ## for DE-mediator table
    if any(list(map(lambda x: x == "DE-mediator table", list(exir_results.keys())))):
        top_N_DE_mediator_table = exir_results['DE-mediator table']
        top_N_DE_mediator_table['Feature'] = list(top_N_DE_mediator_table.index)
        top_N_DE_mediator_table['Class'] = "DE-mediator"

        # top N
        if basis == "Rank":
            top_DE_mediators_index = list(np.argsort(top_N_DE_mediator_table.Rank)[:n])
        elif basis == "Adjusted p-value":
            top_DE_mediators_index = list(np.argsort(top_N_DE_mediator_table.P_adj)[:n])

        top_N_DE_mediator_table = top_N_DE_mediator_table.iloc[top_DE_mediators_index]
        top_N_DE_mediator_table.Rank = rank_cal(top_N_DE_mediator_table.Rank)

        exir_for_plot.append(top_N_DE_mediator_table)

    ####********************####

    # Combine the results for plotting
    exir_for_plot = reduce(lambda x, y: pd.merge(x, y, how= 'outer'), exir_for_plot)

    exir_for_plot.Type[exir_for_plot.Class == 'nonDE-mediator'] = 'Non-DE Mediator'
    exir_for_plot.Type[exir_for_plot.Class == 'DE-mediator'] = 'DE Mediator'

        ###################################################

    # Check which items from subject satisfy the operator function respective to the source
    def which(subject, source, operator = 'in'):

        ## For each element in subject check if that satisfies the operator function respective to the source
        if operator == 'in':
            bool_index = list(map(lambda x: x in source, subject))
        elif operator == 'not in':
            bool_index = list(map(lambda x: x not in source, subject))
        elif operator == '>=':
            bool_index =  list(map(lambda x: x >= source, subject))
        elif operator == '>':
            bool_index =  list(map(lambda x: x > source, subject))
        elif operator == '<=':
            bool_index =  list(map(lambda x: x <= source, subject))
        elif operator == '<':
            bool_index =  list(map(lambda x: x < source, subject))

        ## Get the indices of true elements
        match_index = [i for i, x in enumerate(bool_index) if x]

        ## Get the values of true elements
        match_values = list(map(lambda x: subject[x], match_index))

        ## Return a dictionary including match indices as keys and their corresponding values as the values
        return dict(zip(match_index, match_values))
    
    ###################################################

    ## correct the features names
    if synonyms_table is not None:
        synonyms_table = deepcopy(synonyms_table)
        synonyms_table.index = list(synonyms_table.iloc[:,0])
        synonyms_index = list(which(subject = list(exir_for_plot.Feature), source = list(synonyms_table.index), operator= 'in').values())
        exir_for_plot.Feature.iloc[list(which(subject = list(exir_for_plot.Feature), source = synonyms_index, operator= 'in').keys())] = list(synonyms_table.loc[synonyms_index, synonyms_table.columns[1]])

    ## remove undesired classes
    if show_drivers is False and any(exir_for_plot.Class == "Driver"):
      exir_for_plot = exir_for_plot.drop(list(i for i,j in enumerate(list(exir_for_plot.Class == "Driver")) if j), axis=0)

    if show_biomarkers is False and any(exir_for_plot.Class == "Biomarker"):
      exir_for_plot = exir_for_plot.drop(list(i for i,j in enumerate(list(exir_for_plot.Class == "Biomarker")) if j), axis=0)

    if show_de_mediators is False and any(exir_for_plot.Class == "DE-mediator"):
      exir_for_plot = exir_for_plot.drop(list(i for i,j in enumerate(list(exir_for_plot.Class == "DE-mediator")) if j), axis=0)

    if show_nonDE_mediators is False and any(exir_for_plot.Class == "nonDE-mediator"):
      exir_for_plot = exir_for_plot.drop(list(i for i,j in enumerate(list(exir_for_plot.Class == "nonDE-mediator")) if j), axis=0)

    ## correct the P_adj to be used as the dot size
    exir_for_plot.P_value[exir_for_plot.P_value == np.nan] = 1
    exir_for_plot.P_adj[exir_for_plot.P_adj == np.nan] = 1

    if min(exir_for_plot.P_adj) == 0:
      
      ### range normalize the primitive P_adj
      temp_min_P_adj = sorted(list(set(list(exir_for_plot.P_adj))))[1]
      temp_max_P_adj = max(exir_for_plot.P_adj)
      exir_for_plot.P_adj = rangeNormalize(data = exir_for_plot.P_adj, minimum = temp_max_P_adj, maximum = temp_min_P_adj)
      
    ## Set the P.adj based on min and max arguments
    if len(list(set(list(exir_for_plot.P_adj)))) == 1:
        exir_for_plot.P_adj = mean([dot_size_min, dot_size_max])
    else:
        exir_for_plot.P_adj = rangeNormalize(data = exir_for_plot.P_adj, minimum = dot_size_min, maximum = dot_size_max)

    ####*******************************####

    # Draw the plot
    temp_exir_plot = ggplot(data = exir_for_plot) + aes(x = 'Rank', y = 'Feature')

    # add node objects
    temp_exir_plot += geoms.geom_point(aes(fill = 'Z_score',
                                           colour = 'Type',
                                           size = 'P_adj'),
                                           shape = 'o',
                                           stroke = 0,
                                           alpha = 1)

    ##***********##

    # add color of Type
    temp_exir_plot += geoms.geom_point(aes(fill = 'Z_score',
                                           colour = 'Type',
                                           size = 'P_adj'),
                                           shape = 'o',
                                           stroke = stroke_size,
                                           alpha = stroke_alpha)
    
    ##***********##
    
    # add node and stroke colors
    if isinstance(type_color, str):
        if len(which(subject = ["magma", "inferno", "plasma", "viridis", "cividis", 'twilight', 'twilight_shifted', 'turbo'],
                     source = type_color, operator= 'in')) == 1:
            temp_exir_plot += scale_color_cmap_d(cmap_name = type_color)
        else:
            temp_exir_plot += scale_color_manual(type_color)
    else:
      temp_exir_plot += scale_color_manual(type_color)

    temp_exir_plot += scale_fill_gradient(name = "Z-score", low = dot_color_low, high = dot_color_high)

    # correct size identity inside of aes
    temp_exir_plot += scale_size_identity(guide = "legend")

    # add y axis title
    temp_exir_plot += labs(y = y_axis_title)

    # add facets
    temp_exir_plot += facet_wrap(facets= 'Class',
                          scales = "free_x",
                          nrow = nrow)
    
    temp_exir_plot += scale_size_continuous(name = "Statistical\nsignificance", values = 'P_adj')

    
    # add theme elements
    temp_exir_plot = temp_exir_plot + themes.theme_bw() + themes.theme(legend_position = legend_position, legend_direction = legend_direction)
    
    if boxed_legend:
        temp_exir_plot +=  themes.theme(legend_position=legend_position,
                                      legend_direction=legend_direction,
                                      legend_box_margin=5,
                                      legend_background=themes.element_rect(color='black', size=0.5),
                                      legend_box=legend_direction)
      
    # add title

    if plot_title == "auto":
      plot_title = "ExIR model-based prioritized features"

    if show_plot_title:
      temp_exir_plot += labs(title = plot_title)
    
    # add subtitle
    if plot_subtitle == "auto":
        plot_subtitle = basis + "-based selection of top " + str(n) + " candidates"
    
    if show_plot_subtitle:
        temp_exir_plot += labs(title = plot_title + "\n" + plot_subtitle)

    # define title position
    if title_position == "left":
        title_position = 0
    elif title_position == "center":
        title_position = 0.5
    elif title_position == "right":
        title_position = 1

    title_position = int(title_position)

    title_size = plot_title_size - 2

    # set title position
    if show_plot_title:
        temp_exir_plot += themes.theme(plot_title = themes.element_text(size = title_size, hjust = title_position))
    
    # Set the order of legends
    temp_exir_plot += guides(color = guide_legend(order = 1), size = guide_legend(order = 2))

    if show_y_axis_grid is False:
        temp_exir_plot += themes.theme(panel_grid_major_y = themes.element_blank())
    
    
    return temp_exir_plot
