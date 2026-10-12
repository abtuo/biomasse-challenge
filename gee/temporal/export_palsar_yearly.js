// Annual PALSAR-2 ScanSAR features. Annual matching is the finest supported
// strategy here; it is not an exact +/- acquisition-date window.
var roi = ee.Geometry.Rectangle([-7.85, 4.95, -6.35, 6.45]);
var years = [2019, 2020, 2021, 2022, 2023, 2024, 2025];
var source = ee.ImageCollection('JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR')
  .filterBounds(roi)
  .filter(ee.Filter.listContains('Polarizations', 'HH'))
  .filter(ee.Filter.listContains('Polarizations', 'HV'));

function toDb(image) {
  var hh = image.select('HH').float().max(1).log10().multiply(20).subtract(83).rename('HH_dB');
  var hv = image.select('HV').float().max(1).log10().multiply(20).subtract(83).rename('HV_dB');
  return ee.Image.cat([hh, hv]).copyProperties(image, ['system:time_start']);
}

years.forEach(function(year) {
  var start = ee.Date.fromYMD(year, 1, 1);
  var collection = source.filterDate(start, start.advance(1, 'year'));
  var count = collection.size().getInfo();
  print('PALSAR HH+HV ' + year + ' image count', count);
  if (count > 0) {
    var composite = collection.map(toDb).median().clip(roi).float();
    var hh = composite.select('HH_dB').rename('PALSAR_HH_dB');
    var hv = composite.select('HV_dB').rename('PALSAR_HV_dB');
    var stack = ee.Image.cat([
      hh, hv, hh.subtract(hv).rename('PALSAR_HH_minus_HV_dB'),
      ee.Image(10).pow(hv.subtract(hh).divide(10)).rename('PALSAR_HV_HH_ratio'),
      ee.Image(10).pow(hh.subtract(hv).divide(10)).rename('PALSAR_HH_HV_ratio')
    ]).float();
    Export.image.toDrive({image: stack, description: 'PALSAR_Tai_' + year,
      folder: 'GEDI_Biomass_Challenge_temporal', fileNamePrefix: 'palsar_' + year,
      region: roi, scale: 25, crs: 'EPSG:32629', maxPixels: 1e13,
      fileFormat: 'GeoTIFF', formatOptions: {cloudOptimized: true}});
  }
});
