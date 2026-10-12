// Annual Sentinel-1 features for GEDI temporal alignment, Taï ROI.
var roi = ee.Geometry.Rectangle([-7.85, 4.95, -6.35, 6.45]);
var years = [2019, 2020, 2021, 2022, 2023, 2024, 2025];
var source = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(roi)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .select(['VV', 'VH']);

years.forEach(function(year) {
  var start = ee.Date.fromYMD(year, 1, 1);
  var collection = source.filterDate(start, start.advance(1, 'year'));
  var count = collection.size().getInfo();
  print('S1 ' + year + ' image count', count);
  if (count > 0) {
    var median = collection.median().clip(roi);
    var stack = median.addBands(median.select('VV').subtract(median.select('VH')).rename('VV_minus_VH')).float();
    Export.image.toDrive({image: stack, description: 'S1_Tai_' + year,
      folder: 'GEDI_Biomass_Challenge_temporal', fileNamePrefix: 's1_' + year,
      region: roi, scale: 20, crs: 'EPSG:32629', maxPixels: 1e13,
      fileFormat: 'GeoTIFF', formatOptions: {cloudOptimized: true}});
  }
});
