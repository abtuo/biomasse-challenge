// Annual Sentinel-2 features for GEDI temporal alignment, Taï ROI.
// Run in the Earth Engine Code Editor. Exports are one GeoTIFF per available year.
var roi = ee.Geometry.Rectangle([-7.85, 4.95, -6.35, 6.45]);
var years = [2019, 2020, 2021, 2022, 2023, 2024, 2025];
var bands = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'];
var qaBand = 'cs_cdf';
var clearThreshold = 0.60;
var source = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi);
var cloudScore = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');

function maskClouds(image) {
  return image.updateMask(image.select(qaBand).gte(clearThreshold));
}

years.forEach(function(year) {
  var start = ee.Date.fromYMD(year, 1, 1);
  var end = start.advance(1, 'year');
  var collection = source.filterDate(start, end).linkCollection(cloudScore, [qaBand]).map(maskClouds);
  var count = collection.size().getInfo();
  print('S2 ' + year + ' image count', count);
  if (count > 0) {
    var composite = collection.select(bands).median().clip(roi).divide(10000);
    var stack = composite
      .addBands(composite.normalizedDifference(['B8','B4']).rename('NDVI'))
      .addBands(composite.normalizedDifference(['B8','B11']).rename('NDMI'))
      .addBands(composite.normalizedDifference(['B8','B12']).rename('NBR'))
      .addBands(composite.normalizedDifference(['B8A','B5']).rename('NDRE'))
      .float();
    Export.image.toDrive({image: stack, description: 'S2_Tai_' + year,
      folder: 'GEDI_Biomass_Challenge_temporal', fileNamePrefix: 's2_' + year,
      region: roi, scale: 20, crs: 'EPSG:32629', maxPixels: 1e13,
      fileFormat: 'GeoTIFF', formatOptions: {cloudOptimized: true}});
  }
});
