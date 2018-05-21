-----------------------------------------------------------------------------
--
-- Module      :  gtfs2graph
-- Copyright   :  2015-2018 Franz-Benjamin Mocnik
--
-- The command line application gtfs2graph converts a General Transit Feed
-- Specification (GTFS) transit feed into different graph formats.
-----------------------------------------------------------------------------

{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RankNTypes #-}
{-# OPTIONS_GHC -fno-cse #-}

import Chorale.Common
import Chorale.Geo.Coordinates

import Control.Monad
import qualified Data.ByteString.Lazy.Char8 as C
import Data.Csv hiding (lookup)
import Data.Char
import Data.Either
import Data.List
import Data.List.Utils
import Data.Maybe
import Data.Ord
import qualified Data.Text as T
import qualified Data.Text.Read as T
import qualified Data.Vector as V
import GHC.Generics
import System.Console.CmdArgs
import Text.Printf
import Text.Regex.PCRE.ByteString.Utils

data WeightType = TravelTime | NetworkDistance | DistanceInSpace deriving (Show, Data)

data Setting = GraphML { handle_broken_csv :: Bool, paths :: [String], weightType :: WeightType }
    | SVG { no_shape :: Bool, line_width :: Double, color :: [String], one_color_per_file :: Bool, weights_line_width :: Double, weights_brighten :: Double, weights_opacity_min :: Double, background_color :: String, background_gradient :: Maybe String, size :: Double, title :: Maybe String, title_color :: String, title_font :: String, handle_broken_csv :: Bool, paths :: [String] }
    deriving (Show, Typeable, Data)

{-# ANN handle_broken_csvFlags ("HLint: ignore Use camelCase" :: String) #-}
handle_broken_csvFlags :: Data val => val -> val
handle_broken_csvFlags x = x &= help "try to handle broken csv data in a GTFS transit feed"

pathsFlags :: Data val => val -> val
pathsFlags x = x &= args &= typDir

exportGraphML :: Setting
exportGraphML = GraphML {
    handle_broken_csv = handle_broken_csvFlags False,
    paths = pathsFlags [],
    weightType = TravelTime &= help "weights by TravelTime | NetworkDistance (takes very long to compute) | DistanceInSpace [TravelTime]"
} &= help "convert one or more GTFS paths to a GraphML file"

exportSVG :: Setting
exportSVG = SVG {
    no_shape = def &= help "do not use the shape of the GTFS data but use instead the connectivity graph) [False]",
    line_width = 2 &= help "line width; use '0' to compute a suitable choice automatically [2]",
    color = [] &= help "line color [#528c8e, #f47059, #708e52, #2a3d3b, #e9be2f]",
    one_color_per_file = False &= help "use one color per file instead of one color per route type [False]",
    weights_line_width = 0.6 &= help "adjust line width relative to weights [0.6]",
    weights_brighten = 0 &= help "adjust brightness of the line color relative to weights [0]",
    weights_opacity_min = 0.15 &= help "adjust the opacity to the given minimum relative to weights [0.15]",
    background_color = "#fffcf0" &= help "background color [#fffcf0]",
    background_gradient = Nothing &= help "adds a gradient to the background with the given color [None]",
    size = 1.1 &= help "relative size [1.1]",
    title = Nothing &= help "title [none]",
    title_color = "#707a78" &= help "color of the title [#707a78]",
    title_font = "Georgia" &= help "font of the title [Georgia]",
    handle_broken_csv = handle_broken_csvFlags False,
    paths = pathsFlags []
} &= help "convert a GTFS path to a SVG file"

argsMode :: Mode (CmdArgs Setting)
argsMode = cmdArgsMode $ modes [exportGraphML, exportSVG]
    &= summary "gtfs2graph, (C) Copyright 2015–2018 by Franz-Benjamin Mocnik\nhttps://github.com/mocnik-science/gtfs2graph"
    &= program "gtfs2graph"
    &= helpArg [name "h"]

modifySetting :: Setting -> Setting
modifySetting s@SVG{}
    | (== Just "None") . background_gradient $ s' = s'{ background_gradient = Nothing }
    | otherwise = s' where
        s' = if (notNull . color) s then s else s{ color = ["#528c8e", "#f47059", "#708e52", "#2a3d3b", "#e9be2f"] }
modifySetting s = s

main :: IO ()
main = do
    s <- modifySetting <$> cmdArgsRun argsMode
    case s of
        GraphML { paths = ps, weightType = TravelTime } -> testForPaths ps $ writeGraphML (head ps) =<< concat <$> mapM (\p -> uncurryM3 makeEdgesWeightedByTravelTime (gtfsRead s stopTimes p, gtfsRead s trips p, gtfsRead s routes p)) ps
        GraphML { paths = ps, weightType = NetworkDistance } -> testForPaths ps $ writeGraphML (head ps) =<< concat <$> mapM (\p -> uncurryM5 makeEdgesWeightedByNetworkDistance (gtfsRead s stopTimes p, gtfsRead s stops p, gtfsRead s trips p, gtfsRead s routes p, gtfsRead s shapes p)) ps
        GraphML { paths = ps, weightType = DistanceInSpace } -> testForPaths ps $ writeGraphML (head ps) =<< concat <$> mapM (\p -> uncurryM4 makeEdgesWeightedByDistanceInSpace (gtfsRead s stopTimes p, gtfsRead s stops p, gtfsRead s trips p, gtfsRead s routes p)) ps
        SVG { paths = ps, no_shape = True } -> testForPaths ps $ writeSvg s (head ps ++ "-no-shape") =<< mapM (\p -> uncurryM2 (edgesWithLocalCoordinates .* edgesWithCoordinates) (gtfsRead s stops p, uncurryM3 makeEdgesWeightedByTravelTime (gtfsRead s stopTimes p, gtfsRead s trips p, gtfsRead s routes p))) ps
        SVG { paths = ps, no_shape = False } -> testForPaths ps $ writeSvg s (head ps) =<< mapM (\p -> edgesWithLocalCoordinates <$> uncurryM5 makeEdgesShape (gtfsRead s stopTimes p, gtfsRead s stops p, gtfsRead s trips p, gtfsRead s routes p, gtfsRead s shapes p)) ps

testForPaths :: [String] -> IO () -> IO ()
testForPaths [] = const . putStrLn $ "Nothing to do. No GTFS paths provided."
testForPaths _ = id

-- --== READ GTFS

type RouteType = Int

{-# ANN Route ("HLint: ignore Use camelCase" :: String) #-}
data Route = Route { route_id :: !T.Text, route_type :: !RouteType } deriving (Generic, Show)
instance FromNamedRecord Route

{-# ANN Shape ("HLint: ignore Use camelCase" :: String) #-}
data Shape = Shape { shape_id :: !T.Text, shape_pt_lat :: !Double, shape_pt_lon :: !Double, shape_pt_sequence :: !Int } deriving (Generic, Show)
instance FromNamedRecord Shape

{-# ANN shape_coordinates ("HLint: ignore Use camelCase" :: String) #-}
shape_coordinates :: Shape -> CoordinatesWGS84
shape_coordinates = CoordinatesWGS84 . map21 (shape_pt_lat, shape_pt_lon)

{-# ANN shape_pt_coordinates ("HLint: ignore Use camelCase" :: String) #-}
shape_pt_coordinates :: Shape -> CoordinatesWGS84
shape_pt_coordinates = CoordinatesWGS84 . map21 (shape_pt_lat, shape_pt_lon)

{-# ANN Stop ("HLint: ignore Use camelCase" :: String) #-}
data Stop = Stop { stop_id' :: !T.Text, stop_lat :: !Double, stop_lon :: !Double } deriving (Generic, Show)
instance FromNamedRecord Stop where
    parseNamedRecord m = Stop <$>
        m .: "stop_id" <*>
        m .: "stop_lat" <*>
        m .: "stop_lon"

{-# ANN stop_coordinates ("HLint: ignore Use camelCase" :: String) #-}
stop_coordinates :: Stop -> CoordinatesWGS84
stop_coordinates = CoordinatesWGS84 . map21 (stop_lat, stop_lon)

{-# ANN StopTime ("HLint: ignore Use camelCase" :: String) #-}
data StopTime = StopTime { trip_id :: !T.Text, stop_id :: !T.Text, stop_sequence :: !Int, arrival_time :: !T.Text, departure_time :: !T.Text } deriving (Generic, Show)
instance FromNamedRecord StopTime

{-# ANN Trip ("HLint: ignore Use camelCase" :: String) #-}
data Trip = Trip { route_id' :: !T.Text, trip_id' :: !T.Text, shape_id' :: !T.Text } deriving (Generic, Show)
instance FromNamedRecord Trip where
    parseNamedRecord m = Trip <$>
        m .: "route_id" <*>
        m .: "trip_id" <*>
        m .: "shape_id"

handleBrokenCSV :: IO C.ByteString -> IO C.ByteString
handleBrokenCSV x = fmap (C.intercalate "\n") . mapM (fmap (C.fromStrict . fromRight) . flip (substituteCompile "(?<![\\r\\n,])\"(?![\\r\\n,])") "_" . C.toStrict) . C.split '\n' =<< x

gtfsRead :: FromNamedRecord a => Setting -> String -> String -> IO [a]
gtfsRead setting f p = do
    e <- mapRight (mapSnd V.toList) . decodeByName <$> (applyIf (handle_broken_csv setting) handleBrokenCSV . C.readFile) (p ++ "/" ++ f)
    when (isLeft e) . error $ "Error: could not read GTFS file\n" ++ fromLeft e
    return . snd . fromRight $ e

routes :: String
routes = "routes.txt"

shapes :: String
shapes = "shapes.txt"

stops :: String
stops = "stops.txt"

stopTimes :: String
stopTimes = "stop_times.txt"

trips :: String
trips = "trips.txt"

-- --== EDGES

newtype Edge a = Edge (a, a, Double, RouteType) deriving (Ord, Eq, Show)
type LocalCoordinates = CoordinatesCartesian

weight :: Edge a -> Double
weight (Edge x) = thd4 x

replaceWeight :: Double -> Edge a -> Edge a
replaceWeight w (Edge (n1, n2, _, rt)) = Edge (n1, n2, w, rt)

routeType :: Edge a -> RouteType
routeType (Edge x) = fth4 x

nodes :: Edge a -> (a, a)
nodes (Edge (n1, n2, _, _)) = (n1, n2)

mapEdge :: (n -> m) -> Edge n -> Edge m
mapEdge f (Edge (n1, n2, w, rt)) = Edge (f n1, f n2, w, rt)

routeTypeByTripId :: [(T.Text, RouteType)] -> T.Text -> RouteType
routeTypeByTripId rtbtilt tripId = fromMaybe 0 . lookup tripId $ rtbtilt

routeTypeByTripIdLookupTable :: [Trip] -> [Route] -> [(T.Text, RouteType)]
routeTypeByTripIdLookupTable ts rs = map (map21 (trip_id', \t -> maybe 0 route_type . find ((== route_id' t) . route_id) $ rs)) ts

shapeCoordinatesPerTripLookupTable :: [Trip] -> [Shape] -> [(T.Text, [CoordinatesWGS84])]
shapeCoordinatesPerTripLookupTable ts shs = map (mapSnd fromJust) . filter (isJust . snd) . map (map21 (trip_id', flip lookup shapeCoordinatesPerShape . shape_id')) $ ts where
    shapeCoordinatesPerShape = map (mapSnd $ map shape_pt_coordinates . sortBy (comparing shape_pt_sequence)) . sortAndGroupLookupBy shape_id $ shs

makeEdgesWeightedByFunction :: (T.Text -> [(StopTime, StopTime)] -> [((StopTime, StopTime), Double)]) -> [StopTime] -> [Trip] -> [Route] -> [Edge T.Text]
makeEdgesWeightedByFunction dists sts ts rs = concatMap makeEdges' . sortAndGroupLookupBy trip_id $ sts where
    makeEdges' (t, es) = map (makeEdge' t) . dists t . uncurry zip . map21 (init, tail) . sortBy (comparing stop_sequence) $ es
    makeEdge' t ((s1, s2), w) = Edge (stop_id s1, stop_id s2, w, routeTypeByTripId rtbtilt t)
    rtbtilt = routeTypeByTripIdLookupTable ts rs

makeEdgesWeightedByFunction2 :: (StopTime -> StopTime -> Double) -> [StopTime] -> [Trip] -> [Route] -> [Edge T.Text]
makeEdgesWeightedByFunction2 dist = makeEdgesWeightedByFunction dists where
    dists _ = map (\x -> (x, uncurry dist x))

makeEdgesWeightedByTravelTime :: [StopTime] -> [Trip] -> [Route] -> [Edge T.Text]
makeEdgesWeightedByTravelTime = makeEdgesWeightedByFunction2 dist where
    dist s1 s2 = (toSeconds arrival_time s2 + toSeconds departure_time s2 - toSeconds arrival_time s1 - toSeconds departure_time s1) / 2
    toSeconds f = (\[h, m, s] -> s + 60 * (m + 60 * h)) . map (fst . fromRight . T.double) . T.splitOn ":" . f

makeEdgesWeightedByDistanceInSpace :: [StopTime] -> [Stop] -> [Trip] -> [Route] -> [Edge T.Text]
makeEdgesWeightedByDistanceInSpace sts ss = makeEdgesWeightedByFunction2 dist sts where
    dist s1 s2 = uncurry distance . fromJust . listToTuple2 . map (stopIdToCoordinates ss . stop_id) $ [s1, s2]

makeEdgesWeightedByNetworkDistance :: [StopTime] -> [Stop] -> [Trip] -> [Route] -> [Shape] -> [Edge T.Text]
makeEdgesWeightedByNetworkDistance sts ss ts rs shs = makeEdgesWeightedByFunction dists sts ts rs where
    dists t = case lookup t scptlt of
        Nothing -> const []
        Just shs' -> map (\x -> (x, dist x)) where
            dist (s1, s2) = lengthOfShape affectedShapePairCoordinates where
                (c1, c2) = map12 (stopIdToCoordinates ss . stop_id) (s1, s2)
                lengthOfShape [_] = distance c1 c2
                lengthOfShape sps = distance c1 (snd . head $ sps) + (sum . map (uncurry distance) . tail . init $ sps) + distance (fst . last $ sps) c2
                affectedShapePairCoordinates = flip sublistByIndex shapePairs . fromJust . listToTuple2 . sort . map indexOfShapePairForStopTime $ [c1, c2]
                indexOfShapePairForStopTime c = minimumIndex . map (maximumDistance c) $ shapePairs
            shapePairs = filter (uncurry (/=)) . uncurry zip . map21 (init, tail) $ shs'
    -- works only for short distances
    maximumDistance c (c1, c2)
        | isWithin1 && isWithin2 = max ((* dist1) . abs . sin $ delta1) ((* dist2) . abs . sin $ delta2)
        | isWithin1 = dist2
        | isWithin2 = dist1
        | otherwise = max dist1 dist2 where
            isWithin1 = (abs . roundTwoPi $ delta1) < pi / 2
            isWithin2 = (abs . roundTwoPi $ delta2) < pi / 2
            dist1 = distance c1 c
            dist2 = distance c2 c
            delta1 = az1 - az12
            delta2 = az2 - az21
            az1 = azimuth c1 c
            az2 = azimuth c2 c
            az12 = azimuth c1 c2
            az21 = az12 - pi
            roundTwoPi x = if x < pi then roundTwoPi (x + 2 * pi) else roundTwoPi' x where
                roundTwoPi' x' = if x' > pi then roundTwoPi' (x' - 2 * pi) else x'
    scptlt = shapeCoordinatesPerTripLookupTable ts shs

makeEdgesShape :: [StopTime] -> [Stop] -> [Trip] -> [Route] -> [Shape] -> [Edge CoordinatesWGS84]
makeEdgesShape sts ss ts rs shs = map (Edge . appendThd4 1) . concatMap findEdges $ endpointsPerTrip where
    endpointsPerTrip = map (mapSnd $ map12 (stopIdToCoordinates ss . stop_id) . map21 (head, last) . sortBy (comparing stop_sequence)) . sortAndGroupLookupBy trip_id $ sts
    scptlt = shapeCoordinatesPerTripLookupTable ts shs
    rtbtilt = routeTypeByTripIdLookupTable ts rs
    findEdges (t, (c1, c2)) = case lookup t scptlt of
        Nothing -> []
        Just shs' -> map (appendThd3 . routeTypeByTripId rtbtilt $ t) . uncurry zip . map21 (init, tail) . sublistByIndex (i, j) $ shs' where
            (i, j) = map12 minimumIndex . unzip . map (map21 (distance c1, distance c2)) $ shs'

edgesWithCoordinates :: [Stop] -> [Edge T.Text] -> [Edge CoordinatesWGS84]
edgesWithCoordinates ss = map (mapEdge $ stopIdToCoordinates ss)

stopIdToCoordinates :: [Stop] -> T.Text -> CoordinatesWGS84
stopIdToCoordinates = stop_coordinates . fromJust .* flip (lookupBy stop_id')

edgesWithLocalCoordinates :: [Edge CoordinatesWGS84] -> [Edge LocalCoordinates]
edgesWithLocalCoordinates = map (mapEdge $ transformWGS84toCartesian 12)

-- --== GRAPML

writeGraphML :: FilePath -> [Edge T.Text] -> IO ()
writeGraphML p es = writeFile (p ++ ".graphml") . unlines $ cs where
    es' = map (minimumBy (comparing weight)) . sortAndGroupBy nodes $ es
    ns = nubOrd . concatMap (tupleToList2 . nodes) $ es'
    cs = ["<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
        "<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd\">",
        "<key id=\"weight\" for=\"edge\" attr.name=\"weight\" attr.type=\"double\"/>",
        "<graph id=\"G\" edgedefault=\"directed\">"]
        ++ map (\n -> "<node id=" ++ show n ++ "/>") ns
        ++ map (\(Edge (n1, n2, w, _)) -> "<edge source=" ++ show n1 ++ " target=" ++ show n2 ++ "><data key=\"weight\">" ++ show w ++ "</data></edge>") es'
        ++ ["</graph>", "</graphml>"]

-- --== SVG

data Group = UseRouteType | Group Int

writeSvg :: Setting -> FilePath -> [[Edge LocalCoordinates]] -> IO ()
writeSvg setting p esPerFile = writeFile (p ++ ".svg") . unlines $ cs where
    -- result
    cs = ["<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"" ++ show (yMax - yMin + yTitle) ++ "\" width=\"" ++ show (xMax - xMin) ++ "\">"]
        ++ background
        ++ [maybe "" (\t -> "<text x=\"" ++ show ((xMax - xMin) / 2) ++ "\" y=\"" ++ show yTitle ++ "\" text-anchor=\"middle\" alignment-baseline=\"hanging\" style=\"font-size: " ++ show ((xMax - xMin) / 25) ++ "; font-family: " ++ title_font setting ++ "; fill: " ++ title_color setting ++ ";\">" ++ t ++ "</text>") . title $ setting]
        ++ ["<g stroke-linecap=\"round\">"]
        ++ concatMap convertGroup gs
        ++ ["</g>"]
        ++ ["</svg>"]
    -- background
    background
        | background_color setting /= "" =
            if isNothing . background_gradient $ setting
            then ["<rect width=\"100%\" height=\"100%\" fill=\"" ++ background_color setting ++ "\"/>"]
            else [
                "<radialGradient id=\"backgroundGradient\" gradientUnits=\"userSpaceOnUse\" cx=\"50%\" cy=\"50%\" r=\"100%\">",
                "<stop stop-color=\"" ++ (fromJust . background_gradient $ setting) ++ "\" offset=\"0\"/>",
                "<stop stop-color=\"" ++ background_color setting ++ "\" offset=\"1\"/>",
                "</radialGradient>",
                "<rect x=\"-50%\" y=\"-50%\" width=\"200%\" height=\"200%\" fill=\"url(#backgroundGradient)\"/>"]
        | otherwise = []
    -- groups of edges
    convertGroup ((x, j), es) = ["<g " ++ groupProperties ((colorToRgb . (!!(min j . flip (-) 1 . length . color $ setting)) . cycle . color) setting) x ++ ">"] ++ map convertEdge es ++ ["</g>"]
    groupProperties color' (w, (wMin, wMax)) = unwords [
        "stroke-width=\"" ++ show (lineWidth * (1 + wRel * weights_line_width setting)) ++ "\"",
        "stroke=\"" ++ (rgbToColor . brighten (wRel * weights_brighten setting * (1 - (minimum . tupleToList3 $ color') / 3)) $ color') ++ "\"",
        "opacity=\"" ++ printf "%f" (realToFrac (weights_opacity_min setting + (1 - weights_opacity_min setting) * wRel) :: Float) ++ "\""] where
            wRel = (w - wMin) / (wMax - wMin)
    -- edge
    convertEdge (Edge (CoordinatesCartesian (x1, y1), CoordinatesCartesian (x2, y2), _, _)) = "<line x1=\"" ++ show (x1 - xMin) ++ "\" y1=\"" ++ show (y1 - yMin + yTitle) ++ "\" x2=\"" ++ show (x2 - xMin) ++ "\" y2=\"" ++ show (y2 - yMin + yTitle) ++ "\"/>"
    -- collect edges
    gs = if one_color_per_file setting
        then concatMap (uncurry collectEdges . mapFst Group) . zip [0..] $ esPerFile
        else collectEdges UseRouteType . concat $ esPerFile
    collectEdges (Group j) es = map (mapFst $ appendSnd j . appendSnd (wMin, wMax)) . sortAndGroupLookupBy weight $ es' where
        es' = map (\ns -> replaceWeight (fromIntegral . length $ ns) . head $ ns) . sortAndGroupBy nodes $ es
        (wMin, wMax) = map21 (minimum, maximum) . map weight $ es'
    collectEdges UseRouteType es = map (\(w, es'') -> (((w, (wMin, wMax)), routeType . head $ es''), es'')) . sortAndGroupLookupBy weight $ es' where
        es' = map (\ns -> replaceWeight (fromIntegral . length $ ns) . head $ ns) . sortAndGroupBy (map21 (nodes, routeType)) $ es
        (wMin, wMax) = map21 (minimum, maximum) . map weight $ es'
    -- computations
    ((xMin', xMax'), (yMin', yMax')) = map12 (map21 (minimum, maximum)) . unzip . concatMap (map toTuple . tupleToList2 . nodes) . concatMap snd $ gs
    xDelta = 0.5 * (size setting - 1) * (xMax' - xMin')
    yDelta = 0.5 * (size setting - 1) * (yMax' - yMin')
    xMin = xMin' - xDelta
    yMin = yMin' - yDelta
    xMax = xMax' + xDelta
    yMax = yMax' + yDelta
    yTitle = maybe 0 (const $ (xMax - xMin) / 20) . title $ setting
    lineWidth
        | line_width setting == 0 = (xMax - xMin + yMax - yMin) / 2000.0
        | otherwise = line_width setting

-- --== SVG COLORS

type RGB = (Double, Double, Double)

colorToRgb :: String -> RGB
colorToRgb x
    | map toLower x == "black" = (0, 0, 0)
    | map toLower x == "white" = (1, 1, 1)
    | map toLower x == "red" = (1, 0, 0)
    | map toLower x == "green" = (0, 0.5, 0)
    | map toLower x == "blue" = (0, 0, 1)
    | map toLower x == "yellow" = (1, 1, 0)
    | startswith "#" x && length x == 4 = let ['#', r, g, b] = x in map13 ((/ 15) . hexToDouble . map toLower) ([r], [g], [b])
    | startswith "#" x && length x == 7 = let ['#', r1, r2, g1, g2, b1, b2] = x in map13 ((/ 255) . hexToDouble . map toLower) ([r1, r2], [g1, g2], [b1, b2])
    | otherwise = error "Color could not be parsed. Please use 'black', 'white', 'red', 'green', 'blue', 'yellow', or a color code."

rgbToColor :: RGB -> String
rgbToColor = concat . ("#":) . map (reverse . take 2 . reverse . ("00" ++) . doubleToHex . (* 255)) . tupleToList3

{-# ANN hexNumbers ("HLint: ignore Use String" :: String) #-}
hexNumbers :: [Char]
hexNumbers = "0123456789abcdef"

hexToDouble :: String -> Double
hexToDouble = hexToDouble' 0.0 where
    hexToDouble' n [] = n
    hexToDouble' n (x:xs) = case elemIndex x hexNumbers of
        Just j -> hexToDouble' (n * (fromIntegral . length) hexNumbers + fromIntegral j) xs
        Nothing -> error "Color must contain only a hexidecimal numbers."

doubleToHex :: Double -> String
doubleToHex = doubleToHex' "" . round where
    doubleToHex' "" 0 = "0"
    doubleToHex' xs 0 = xs
    doubleToHex' xs n = doubleToHex' (hexNumbers!!j : xs) n' where
        (n', j) = divMod n (length hexNumbers)

brighten :: Double -> RGB -> RGB
brighten x = map13 (min 1 . (+ x))
