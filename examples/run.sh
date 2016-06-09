mkdir data

## Example: Washington, DC, USA
curl http://transitfeeds.com/p/wmata/75/20160608/download -L -o data/usa-dc-washington.zip
unzip data/usa-dc-washington.zip -d data/usa-dc-washington
gtfs2graph svg data/usa-dc-washington --line-width=4 --title="WASHINGTON, DC, USA"
cairosvg data/usa-dc-washington.svg -o usa-dc-washington.pdf
pdftoppm -png -scale-to-x 500 -scale-to-y -1 -singlefile usa-dc-washington.pdf usa-dc-washington
gzip --best -k data/usa-dc-washington.svg && mv data/usa-dc-washington.svg.gz usa-dc-washington.svgz
gtfs2graph graphml data/usa-dc-washington && mv data/usa-dc-washington.graphml .

## Example: Albuquerque, NM, USA
curl http://transitfeeds.com/p/abq-ride/52/20160429/download -L -o data/usa-nm-albuquerque.zip
unzip data/usa-nm-albuquerque.zip -d data/usa-nm-albuquerque
gtfs2graph svg data/usa-nm-albuquerque --color=red --color=blue --color=yellow --one-color-per-file --weights-line-width=3 --weights-brighten=0.8 --weights-opacity-min=0.3 --background-color=black --title="ALBUQUERQUE, NM, USA" --title-color=#ddd --title-font=Helvetica
cairosvg data/usa-nm-albuquerque.svg -o usa-nm-albuquerque.pdf
pdftoppm -png -scale-to-x 500 -scale-to-y -1 -singlefile usa-nm-albuquerque.pdf usa-nm-albuquerque
gzip --best -k data/usa-nm-albuquerque.svg && mv data/usa-nm-albuquerque.svg.gz usa-nm-albuquerque.svgz
gtfs2graph graphml data/usa-nm-albuquerque && mv data/usa-nm-albuquerque.graphml .

## Example: San Francisco, CA, USA
curl http://transitfeeds.com/p/sfmta/60/20160526/download -L -o data/usa-ca-san-francisco.zip
unzip data/usa-ca-san-francisco.zip -d data/usa-ca-san-francisco
gtfs2graph svg data/usa-ca-san-francisco --no-shape --line-width=0 --background-color="" --color=black --one-color-per-file --weights-line-width=0 --weights-brighten=0 --weights-opacity-min=1 --size=1
cairosvg data/usa-ca-san-francisco-no-shape.svg -o usa-ca-san-francisco.pdf
pdftoppm -png -scale-to-x 500 -scale-to-y -1 -singlefile usa-ca-san-francisco.pdf usa-ca-san-francisco
gzip --best -k data/usa-ca-san-francisco-no-shape.svg && mv data/usa-ca-san-francisco-no-shape.svg.gz usa-ca-san-francisco.svgz
gtfs2graph graphml data/usa-ca-san-francisco && mv data/usa-ca-san-francisco.graphml .

## Clean Up
rm -rf data/usa-dc-washington
rm -rf data/usa-nm-albuquerque
rm -rf data/usa-ca-san-francisco
