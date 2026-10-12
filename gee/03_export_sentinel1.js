var roi = ee.Geometry.Rectangle([-7.85, 4.95, -6.35, 6.45]);
Map.centerObject(roi, 8);

var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(roi)
  .filterDate('2020-01-01', '2025-01-01')
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .select(['VV', 'VH']);

var median = s1.median().clip(roi);
var ratio = median.select('VV').subtract(median.select('VH')).rename('VV_minus_VH');
var stack = median.addBands(ratio);

Export.image.toDrive({
  image: stack,
  description: 'Sentinel1_Tai_features_2020_2024',
  folder: 'GEDI_Biomass_Challenge',
  fileNamePrefix: 'sentinel1_tai_features_2020_2024',
  region: roi,
  scale: 20,
  crs: 'EPSG:32629',
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});
