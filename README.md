# gtfs2graph

The command line application `gtfs2graph` converts a [General Transit Feed Specification (GTFS) transit feed](https://developers.google.com/transit/gtfs/reference) into different graph formats.

## Examples

[![Washington, DC, USA](https://github.com/mocnik-science/gtfs2graph/blob/master/examples/usa-dc-washington.png)](https://github.com/mocnik-science/gtfs2graph/blob/master/examples/usa-dc-washington.pdf)

```
gtfs2graph svg data/usa-dc-washington --line-width=4 --title="WASHINGTON, DC, USA"
```

Visualization: [PDF](https://github.com/mocnik-science/gtfs2graph/blob/master/examples/usa-dc-washington.pdf) /
GraphML: [GraphML](https://github.com/mocnik-science/gtfs2graph/blob/master/examples/usa-dc-washington.graphml) /
GTFS data: [www.transitfeeds.com](http://www.transitfeeds.com/p/wmata/75/20160608)

[![Albuquerque, NM, USA](https://github.com/mocnik-science/gtfs2graph/blob/master/examples/usa-nm-albuquerque.png)](https://github.com/mocnik-science/gtfs2graph/blob/master/examples/usa-nm-albuquerque.pdf)

```
gtfs2graph svg data/usa-nm-albuquerque --color=red --color=blue --color=yellow --weights-line-width=3 --weights-brighten=0.8 --weights-opacity-min=0.3 --background-color=black --title="ALBUQUERQUE, NM, USA" --title-color=#ddd --title-font=Helvetica
```

Visualization: [PDF](https://github.com/mocnik-science/gtfs2graph/blob/master/examples/usa-nm-albuquerque.pdf) /
GraphML: [GraphML](https://github.com/mocnik-science/gtfs2graph/blob/master/examples/usa-nm-albuquerque.graphml) /
GTFS data: [www.transitfeeds.com](http://www.transitfeeds.com/p/chicago-transit-authority/165/20160603)

[![San Francisco, CA, USA](https://github.com/mocnik-science/gtfs2graph/blob/master/examples/usa-ca-san-francisco.png)](https://github.com/mocnik-science/gtfs2graph/blob/master/examples/usa-ca-san-francisco.pdf)

```
gtfs2graph svg data/usa-ca-san-francisco --no-shape --line-width=0 --background-color="" --color=black --one-color-per-file --weights-line-width=0 --weights-brighten=0 --weights-opacity-min=1 --size=1
```

Visualization: [PDF](https://github.com/mocnik-science/gtfs2graph/blob/master/examples/usa-ca-san-francisco.pdf) /
GraphML: [GraphML](https://github.com/mocnik-science/gtfs2graph/blob/master/examples/usa-ca-san-francisco.graphml) /
GTFS data: [www.transitfeeds.com](http://www.transitfeeds.com/p/sfmta/60/20160526)

## Installation

To install `gtfs2graph`, [Haskell](https://www.haskell.org/platform/) needs to be installed. Then execute:
```
mkdir gtfs2graph && cd gtfs2graph
wget https://github.com/mocnik-science/gtfs2graph/archive/master.zip
unzip master.zip
cd gtfs2graph-master
cabal update
cabal install
```

## Usage

Two different types of networks can be constructed:

**Connectivity Network.** The nodes of the connectivity network represent stops and stations of the transit feed, and edges represent pairs of successive stops of the same trip.

**Shaped Network.** The shaped network represents, in addition to the connectivity network, information about the shape of the connections between the stops and stations.

The weights can be constructed in two different ways:

**Connection Weights.** The edge weights refer to the number of edges with the same start and end node, i.e. to the number of connections existing between two nodes.

**Travel Time Weights.** The weights of the edges represent the travel time (the start point in time of the travel is defined as the arithmetic mean of the preceding arrival and the departure of the travel represented by the edge, and the end point as the mean of the arrival and the succeeding departure). When more than one edge is present, the one with the smallest weight is chosen.

Dependent on the file format chosen to export the data, different types of networks and different types of weights are available.

The options can be explored by `gtfs2graph -h`:
```
gtfs2graph, (C) Copyright 2015–2016 by Franz-Benjamin Mocnik
https://github.com/mocnik-science/gtfs2graph

gtfs2graph [COMMAND] ... [OPTIONS]

Common flags:
        --handle-broken-csv         try to handle broken csv data in a GTFS
                                    transit feed
  -h -? --help                      Display help message
  -V    --version                   Print version information
        --numeric-version           Print just the version number

gtfs2graph graphml [OPTIONS] [DIR]
  convert one or more GTFS paths to a GraphML file

gtfs2graph svg [OPTIONS] [DIR]
  convert a GTFS path to a SVG file

  -n    --no-shape                  do not use the shape of the GTFS data but
                                    use instead the connectivity graph) [False]
  -l    --line-width=NUM            line width; use '0' to compute a suitable
                                    choice automatically [2]
  -c    --color=ITEM                line color [#528c8e, #f47059, #708e52,
                                    #2a3d3b, #e9be2f]
  -o    --one-color-per-file        use one color per file instead of one
                                    color per route type [False]
        --weights-line-width=NUM    adjust line width relative to weights
                                    [0.6]
        --weights-brighten=NUM      adjust brightness of the line color
                                    relative to weights [0]
        --weights-opacity-min=NUM   adjust the opacity to the given minimum
                                    relative to weights [0.15]
        --background-color=ITEM     background color [#fffcf0]
        --background-gradient=ITEM  adds a gradient to the background with
                                    the given color [None]
  -s    --size=NUM                  relative size [1.1]
        --title=ITEM                title [none]
        --title-color=ITEM          color of the title [#707a78]
        --title-font=ITEM           font of the title [Georgia]
```
Most notably, the option `--handle-broken-csv` can be used to handle GTFS data that is broken.

## Export As GraphML

A GTFS transit feed can, as a *connectivity network* with *connection weights*, be exported in the [GraphML format](http://graphml.graphdrawing.org) format:
```
gtfs2graph graphml gtfs_dir
```
The parameter `gtfs_dir` refers to the path which contains the transit feed as unpacked txt files.

## Export As SVG

A visualization of a GTFS transit feed can, as a *shaped network* with *travel time weights*, be saved in the [SVG format](http://www.w3.org/Graphics/SVG/):
```
gtfs2graph svg gtfs_dir
```
Alternatively, the also the *connectivity network* with *travel time weights* can be saved:
```
gtfs2graph svg --no-shape gtfs_dir
```
Many options are available to adjust the visualization. The options can be explored by `gtfs2graph -h`.

A generated SVG file can be converted to a PDF file, e.g. by using `svg2pdf` or `cairosvg` available from [Cairo](http://www.cairographics.org):
```
cairosvg file.svg -o file.pdf
```
Alternatively, a generated SVG file can be converted to a PNG (or JPEG) file, e.g. by using `pdftoppm` available from [Poppler](http://poppler.freedesktop.org):
```
pdftoppm -png -scale-to-x 1200 -singlefile file.pdf file
```

The colours used in the visualization refer, in case the option `--one-color-per-file` is not used, to the different modes of transport. The first given colour refers to trams, the second one to subways and metros, etc. Compare also the [GTFS documentation](https://developers.google.com/transit/gtfs/reference#routestxt):
```
0 - Tram, Streetcar, Light rail. Any light rail or street level system within a metropolitan area.
1 - Subway, Metro. Any underground rail system within a metropolitan area.
2 - Rail. Used for intercity or long-distance travel.
3 - Bus. Used for short- and long-distance bus routes.
4 - Ferry. Used for short- and long-distance boat service.
5 - Cable car. Used for street-level cable cars where the cable runs beneath the car.
6 - Gondola, Suspended cable car. Typically used for aerial cable cars where the car is suspended from the cable.
7 - Funicular. Any rail system designed for steep inclines.
```

## Author

This application is written and maintained by Franz-Benjamin Mocnik, <mail@mocnik-science.net>.

(c) by Franz-Benjamin Mocnik, 2015–2016.

The code is licensed under the [GPL-3](https://github.com/mocnik-science/gtfs2graph/blob/master/LICENSE.md).
