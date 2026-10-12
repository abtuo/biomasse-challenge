var roi = ee.Geometry.Rectangle([-7.85, 4.95, -6.35, 6.45]);
Map.centerObject(roi, 8);

var startDate = '2020-01-01';
var endDate = '2025-01-01';
var QA_BAND = 'cs_cdf';
var CLEAR_THRESHOLD = 0.60;

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(roi)
  .filterDate(startDate, endDate);

var cloudScore = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');

var linked = s2.linkCollection(cloudScore, [QA_BAND]);

function maskClouds(image) {
  return image.updateMask(image.select(QA_BAND).gte(CLEAR_THRESHOLD));
}

var bands = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'];

var composite = linked
  .map(maskClouds)
  .select(bands)
  .median()
  .clip(roi)
  .divide(10000);

var ndvi = composite.normalizedDifference(['B8','B4']).rename('NDVI');
var ndmi = composite.normalizedDifference(['B8','B11']).rename('NDMI');
var nbr = composite.normalizedDifference(['B8','B12']).rename('NBR');
var ndre = composite.normalizedDifference(['B8A','B5']).rename('NDRE');

var stack = composite.addBands([ndvi, ndmi, nbr, ndre]);

Export.image.toDrive({
  image: stack,
  description: 'Sentinel2_Tai_features_2020_2024',
  folder: 'GEDI_Biomass_Challenge',
  fileNamePrefix: 'sentinel2_tai_features_2020_2024',
  region: roi,
  scale: 20,
  crs: 'EPSG:32629',
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});
