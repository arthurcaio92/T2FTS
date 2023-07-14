"""This module is used to plot 3-dimensional graphs of type-2 fuzzy sets."""

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d


from . import global_settings as gs


def _darken_colour(hexstr, p):
    """Multiply the hex values by the percentage p."""
    hexstr = hexstr.replace('#', '')
    rgb = [(ord(c)*p)/255.0 for c in hexstr.decode('hex')]
    #rgb.append(0.5)
    return rgb


def _identify_direction_changes(l):
    """Identify the indexes of a list where the values change direction.

    i.e. changing from increasing value to decreasing value, or vice versa.
    Returns a list of indexes where each index is where the new direction
    begins in the list l.
    """
    if len(l) <= 2:
        return [len(l)]
    LOWER = 0
    HIGHER = 1

    def _check_cur_state(i1, i2):
        if l[i2] > l[i1]:
            return HIGHER
        else:
            return LOWER
    change_locations = []
    cur_state = _check_cur_state(0, 1)
    for i in range(1, len(l)-1):
        new_state = _check_cur_state(i, i+1)
        if cur_state != new_state:
            change_locations.append(i+1)
        cur_state = new_state
    return change_locations


def _append_reversed_list(li):
    """Return a new list as li.extend(li.reverse())."""
    new_li = li[:]
    li_reverse = li[:]
    li_reverse.reverse()
    new_li.extend(li_reverse)
    return new_li


def _plot_faces(ax, XL, YL, XU, YU, prev_z, cur_z, col):
    """Plot the FOU for the bottom and top of the zslice."""
    X = XL[:]
    Y = YL[:]
    X.extend(XU)
    Y.extend(YU)
    verts = [list(zip(X, Y, [prev_z for i in range(len(X))]))]
    p = Poly3DCollection(verts, edgecolors='#000000')
    p.set_facecolor(col)
    ax.add_collection3d(p)
    verts = [list(zip(X, Y, [cur_z for i in range(len(X))]))]
    p = Poly3DCollection(verts, edgecolors='#000000')
    p.set_facecolor(col)
    ax.add_collection3d(p)


def _plot_edge(ax, XL, YL, prev_z, cur_z, col):
    """Plot the inside and outside edge of the zslice.

    This joins the top and bottom parts plotted by _plot_faces.
    """
    dir_changes = _identify_direction_changes(YL)
    dir_changes.insert(0, 1)
    dir_changes.append(len(YL))
    for i in range(len(dir_changes)-1):
        start = dir_changes[i]-1
        end = dir_changes[i+1]
        X = _append_reversed_list(XL[start:end])
        Y = _append_reversed_list(YL[start:end])
        Z = [prev_z for i in range(end - start)]
        Z.extend([cur_z for i in range(end - start)])
        X.append(X[0])
        Y.append(Y[0])
        Z.append(Z[0])
        verts = [list(zip(X, Y, Z))]
        p = Poly3DCollection(verts, edgecolors='#000000')
        p.set_facecolor(col)
        ax.add_collection3d(p)


def plot_sets(fuzzy_sets, filename=None):
    """Display a 3-dimensional plot of the given list of fuzzy sets.

    If filename is None, the plot is displayed.
    If a filename is given, the plot is saved to the given location.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if gs.type_2_3d_colour_scheme == gs.UNIQUE:
        colours = gs.colours
    elif gs.type_2_3d_colour_scheme == gs.HEATMAP:
        colours = gs.get_z_level_heatmap()
    elif gs.type_2_3d_colour_scheme == gs.GREYSCALE:
        colours = gs.get_z_level_greyscale()
    colour_index = 0
    for fs in fuzzy_sets:
        zlevels = fs.zlevel_coords[:]
        zlevels.insert(0, 0)
        for zi in range(1, len(zlevels)):
            XL = []
            XU = []
            YL = []
            YU = []
            for x in gs.get_x_points():
                yl, yu = fs.calculate_membership(x, zlevels[zi])
                # Note: If you have these if statements then spikes
                # in IAA won't be as apparent
                if yl > 0:
                    XL.append(float(x))
                    YL.append(float(yl))
                if yu > 0:
                    # add in reverse order to loop around the fuzzy set in a circle
                    XU.insert(0, float(x))
                    YU.insert(0, float(yu))
            prev_z = float(zlevels[zi-1])
            cur_z = float(zlevels[zi])
            if gs.type_2_3d_colour_scheme == gs.UNIQUE:
                col = _darken_colour(gs.colours[colour_index], 1-(0.5*cur_z))
            else:
                col = colours[colour_index].get_rgb()
            _plot_faces(ax, XL, YL, XU, YU, prev_z, cur_z, col)
            if len(XL) > 0:
                _plot_edge(ax, XL, YL, prev_z, cur_z, col)
            if len(XU) > 0:
                _plot_edge(ax, XU, YU, prev_z, cur_z, col)
            if gs.type_2_3d_colour_scheme != gs.UNIQUE:
                colour_index += 1
        if gs.type_2_3d_colour_scheme == gs.UNIQUE:
            colour_index += 1
        else:
            colour_index = 0
    ax.set_xlim(fs.uod[0], fs.uod[1])
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel(gs.xlabel, fontsize=18)
    ax.set_ylabel(gs.type_2_ylabel, fontsize=18)
    ax.set_zlabel(gs.type_1_ylabel, fontsize=18)
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
