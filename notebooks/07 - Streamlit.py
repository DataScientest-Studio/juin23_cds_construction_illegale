################################################################################
# 00 - Tests Streamlit
################################################################################


################################################################################
# Imports & Déclarations
################################################################################

# Imports
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import streamlit as st
import folium
import requests
import shapely as sh
from streamlit_folium import st_folium
from PIL import Image, ImageOps
from io import BytesIO
from shapely.geometry import Point
from ultralytics import YOLO
from pyproj import Proj, transform

# Constantes
HOME = '/Users/davidroch/Programmation/python/constructions/'
MODEL_PATH = HOME + 'yolo/' + 'yolov8l-seg-GPUT100-11h.pt'
MAP_STARTING_LOCATION = [48.836674, 2.239333]
LAMBERT_IMAGE_SIDE = 200
PIXELS_IMAGE_SIDE = 640
AREA_2_SQM_FACTOR = (200 ** 2) # surface normalisée -> mètres-carrés
COLOR_TRUE = '#00FF00'
COLOR_PRED = '#0080FF'
COLOR_SUSPECTS = '#FF0000'

# Variables globales
VAR_IMAGE = np.zeros((PIXELS_IMAGE_SIDE, PIXELS_IMAGE_SIDE, 3))
VAR_TRUE_POLYGONS = gpd.GeoSeries()
VAR_PRED_POLYGONS = gpd.GeoSeries()

################################################################################
# Fonctions de conversion et de normalisation
################################################################################

lambert93Proj = Proj(init='epsg:9794')
wgsProj = Proj(init='epsg:4326')

def lambert2wgs(x, y):
    return transform(lambert93Proj, wgsProj, x, y)

def wgs2lambert(x, y):
    return transform(wgsProj, lambert93Proj, x, y)

def normalize_band(band, lower_percentile=20, upper_percentile=80):
    lower_value, upper_value = np.percentile(band, (lower_percentile, upper_percentile))
    return np.clip((band - lower_value) / (upper_value - lower_value), 0, 1)

def normalize_image(value, lower_percentile=20, upper_percentile=80):
    value[...,0]=normalize_band(value[...,0], lower_percentile, upper_percentile)*255
    value[...,1]=normalize_band(value[...,1], lower_percentile, upper_percentile)*255
    value[...,2]=normalize_band(value[...,2], lower_percentile, upper_percentile)*255
    return value

################################################################################
# req_ortho()
# Récupère l'orthophoto dans un cadre rectangulaire
#    bounds : en coordonnées Lambert 93
################################################################################
@st.cache_data
def req_ortho(bounds): # bounds doit être en Lambert93 (9794 mais seul 2154 fonctionne pour la requête)
    request = 'https://data.geopf.fr/wms-r?LAYERS=HR.ORTHOIMAGERY.ORTHOPHOTOS&FORMAT=image/tiff&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&STYLES=&CRS=EPSG:2154&BBOX='
    request += str(bounds[0][0]) + ',' + str(bounds[0][1]) + ',' + str(bounds[1][0]) + ',' + str(bounds[1][1])
    request += '&WIDTH=' + str(PIXELS_IMAGE_SIDE) + '&HEIGHT=' + str(PIXELS_IMAGE_SIDE)
    response = requests.get(request).content
    orthophoto = Image.open(BytesIO(response))
    return np.array(orthophoto)

################################################################################
# multiPolygons2Polygons()
################################################################################

def multiPolygons2Polygons(polygons):
    select_multi = (polygons.type == 'MultiPolygon') | (polygons.type == 'GeometryCollection')
    polygons_multi = polygons[select_multi]
    polygons = polygons[~select_multi]
    for p in polygons_multi:
        polygons = pd.concat([polygons, gpd.GeoSeries(list(p.geoms))])
    return polygons

################################################################################
# cropPolygonsOnBounds()
################################################################################

def cropPolygonsOnBounds(polygons, bounds):
    rect_poly = sh.Polygon([(bounds[0][0], bounds[0][1]),
                            (bounds[1][0], bounds[0][1]),
                            (bounds[1][0], bounds[1][1]),
                            (bounds[0][0], bounds[1][1])])
    ret = gpd.GeoSeries(polygons.values.intersection(rect_poly))
    return ret

################################################################################
# req_batiments()
# Récupère tous les bâtiments dans la zone
#    bounds : en coordonnées WGS
################################################################################

def req_batiments(bounds): # bounds doit être en WGS
    request = 'https://data.geopf.fr/wfs/ows?SERVICE=WFS&REQUEST=GetFeature&VERSION=2.0.0&outputFormat=json&TYPENAMES=BDTOPO_V3:batiment&bbox='
    request += str(bounds[0][1])+','+str(bounds[0][0])+','+str(bounds[1][1])+','+str(bounds[1][0])
    response = requests.get(request)
    polygons = gpd.GeoDataFrame.from_features(response.json()['features'])
    polygons = polygons['geometry'].transform(lambda p : sh.wkb.loads(sh.wkb.dumps(p, output_dimension=2))) # Supprimer l'altitude
    return polygons

################################################################################
# resizePolygons()
# Permet de mettre les coordonnées des polygones :
#   - à la taille de l'image
#   - origine en haut à gauche (d'où reverseY)
################################################################################

def resizePolygon(xy, xrange, yrange, xtarget, ytarget, reverseY=False):
    if (reverseY==True):
        ret = (xy - [xrange[0], yrange[1]]) * (xtarget-1, ytarget-1) / [xrange[1]-xrange[0], yrange[0]-yrange[1]]
    else:
        ret = (xy - [xrange[0], yrange[0]]) * (xtarget-1, ytarget-1) / [xrange[1]-xrange[0], yrange[1]-yrange[0]]
    return ret

################################################################################
# getBounds()
# Retoune 2 points délimitant l'image à afficher (en WGS)
################################################################################

def getBounds():
    x, y = wgs2lambert(st_data['center']['lng'], st_data['center']['lat'])
    bounds = [(x-LAMBERT_IMAGE_SIDE/2, y-LAMBERT_IMAGE_SIDE/2),
                (x+LAMBERT_IMAGE_SIDE/2, y+LAMBERT_IMAGE_SIDE/2)]
    return bounds

################################################################################
# loadModel()
################################################################################
@st.cache_resource
def loadModel():
    return YOLO(MODEL_PATH)

################################################################################
# getPredPolygons()
# Création des polygones prédits par le modèle
################################################################################

def getPredPolygons(model, img):
    all_polygons = []
    result = model(img)[0] # on a un batch d'une seule image
    # parcours des différents polygones de l'image
    if result.masks != None:
        for mask in result.masks:
            points = np.array(mask.xyn).squeeze() # passer de (1, nbpoints, 2) à (nbpoints, 2)
            if len(points) > 2:
                poly = sh.geometry.Polygon([[p[0], p[1]] for p in points])
                all_polygons.append(poly) 
    ret = gpd.GeoSeries(all_polygons)
    #ret = multiPolygons2Polygons(ret) # PATCH
    return ret

################################################################################
# geSuspectPolygons()
# Calcule les poygones prédits n'apparaissant pas dans les réels
################################################################################

def getSuspectPolygons(true, pred):
    # Calcul de l'union des vrais polygones par image : truepoly
    truepoly = true.unary_union
    # Calcul des polygones suspects
    suspectpolys, areas = [], []
    nberr = 0
    for predpoly in pred:
        try:
            suspectpoly = predpoly - truepoly
            suspectpolys.append(suspectpoly)
            areas.append(suspectpoly.area)
        except:
            nberr +=1
    if (nberr > 0):
        st.write(f'{nberr=}')

    ret = gpd.GeoDataFrame({'geometry' : suspectpolys, 'suspectArea' : areas})
    #ret['suspectArea'] = (ret['suspectArea'] / (PIXELS_IMAGE_SIDE ** 2)) * AREA_2_SQM_FACTOR
    ret['suspectArea'] = ret['suspectArea'] * ((LAMBERT_IMAGE_SIDE / PIXELS_IMAGE_SIDE) ** 2)
    return ret


######################################################################################################################
# Page
######################################################################################################################

# Configuration de la page
st.set_page_config(
    page_title='Détection de constructions illégales',
    layout='wide')

# Checkboxes
st.sidebar.write('Contours')
cb_contours_true = st.sidebar.checkbox('Recensés (BDTOPO)')
cb_contours_pred = st.sidebar.checkbox('Prédits (YOLO)')
# Radio buttons
st.sidebar.divider()
st.sidebar.write('\n')
rd_fill = st.sidebar.radio(label='Surfaces', options=['Recensés (BDTOPO)', 'Prédits (YOLO)', 'Suspects'])
# Slider
st.sidebar.divider()
sl_surface = st.sidebar.select_slider(label='Surface illégale minimale', options=range(101), value=20)
# Colonnes à afficher
st.title('Détection de constructions illégales')
st.divider()
left, medium, right = st.columns(3)

with left:
    m = folium.Map(location=MAP_STARTING_LOCATION, zoom_start=18, width=100, height=100)  
    st_data = st_folium(m, width=500, height=500)
    lambert_bounds = getBounds()
    wgs_bounds = lambert2wgs(*lambert_bounds[0]), lambert2wgs(*lambert_bounds[1])

with medium:
    # Chargement de l'image
    VAR_IMAGE = req_ortho(lambert_bounds)
    VAR_IMAGE = normalize_image(VAR_IMAGE, lower_percentile=5, upper_percentile=95)

    # Récupération de VAR_TRUE_POLYGONS
    VAR_TRUE_POLYGONS = req_batiments(wgs_bounds) # en WGS
    VAR_TRUE_POLYGONS = VAR_TRUE_POLYGONS.set_crs('EPSG:4326', allow_override=True).to_crs('EPSG:9794') # en LAMB93
    VAR_TRUE_POLYGONS = cropPolygonsOnBounds(VAR_TRUE_POLYGONS, lambert_bounds)
    xrange, yrange = (lambert_bounds[0][0], lambert_bounds[1][0]), (lambert_bounds[0][1], lambert_bounds[1][1])
    VAR_TRUE_POLYGONS = gpd.GeoSeries(sh.transform(VAR_TRUE_POLYGONS, lambda p : resizePolygon(
        p, xrange, yrange, PIXELS_IMAGE_SIDE, PIXELS_IMAGE_SIDE, reverseY=True))) # à la taille de l'image

    # Calcul de VAR_PRED_POLYGONS
    model = loadModel()
    VAR_PRED_POLYGONS = getPredPolygons(model, VAR_IMAGE)
    VAR_PRED_POLYGONS = VAR_PRED_POLYGONS.scale(xfact=PIXELS_IMAGE_SIDE, yfact=PIXELS_IMAGE_SIDE, origin=(0, 0))

    # Calcul de VAR_SUSPECTS
    VAR_SUSPECTS = getSuspectPolygons(VAR_TRUE_POLYGONS, VAR_PRED_POLYGONS)
    VAR_SUSPECTS = VAR_SUSPECTS[VAR_SUSPECTS['suspectArea'] >= sl_surface]

    # Affichages
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(VAR_IMAGE)
    plt.axis('off')
    if (cb_contours_true):
        VAR_TRUE_POLYGONS.plot(linewidth=1, facecolor='none', edgecolor=COLOR_TRUE, ax=ax)
        dfcoords = VAR_TRUE_POLYGONS.get_coordinates()
        dfcoords.to_excel('coords.xlsx')
    if (cb_contours_pred):
        VAR_PRED_POLYGONS.plot(linewidth=1, facecolor='none', edgecolor=COLOR_PRED, ax=ax)
    st.pyplot(fig)

with right:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(VAR_IMAGE, alpha=0.5)
    plt.axis('off')
    match rd_fill:
        case 'Recensés (BDTOPO)':
            VAR_TRUE_POLYGONS.plot(linewidth=1, facecolor=COLOR_TRUE, edgecolor='none', ax=ax)
        case 'Prédits (YOLO)':
            VAR_PRED_POLYGONS.plot(linewidth=1, facecolor=COLOR_PRED, edgecolor='none', ax=ax)
        case 'Suspects':
            VAR_SUSPECTS.plot(linewidth=1, facecolor=COLOR_SUSPECTS, edgecolor='none', ax=ax)
    st.pyplot(fig)

