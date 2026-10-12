var roi = ee.Geometry.Rectangle([-7.85, 4.95, -6.35, 6.45]);
Map.centerObject(roi, 8);

var idx = ee.FeatureCollection('LARSE/GEDI/GEDI04_A_002_INDEX').filterBounds(roi);
var tableIds = idx.aggregate_array('table_id');

tableIds.evaluate(function(ids) {
  var merged = ee.FeatureCollection([]);

  ids.forEach(function(id) {
    merged = merged.merge(ee.FeatureCollection(id).filterBounds(roi));
  });

  var filtered = merged
    .filter(ee.Filter.eq('l4_quality_flag', 1))
    .filter(ee.Filter.eq('degrade_flag', 0))
    .filter(ee.Filter.gt('agbd', 0))
    .filter(ee.Filter.gt('agbd_se', 0))
    .map(function(f) {
      var rel = ee.Number(f.get('agbd_se')).divide(ee.Number(f.get('agbd')));
      var xy = f.geometry().coordinates();
      return f.set({
        relative_se: rel,
        longitude: xy.get(0),
        latitude: xy.get(1)
      });
    })
    .filter(ee.Filter.lte('relative_se', 0.5));

  print('Filtered GEDI footprints', filtered.size());

  Export.table.toDrive({
    collection: filtered,
    description: 'GEDI_L4A_Tai_filtered',
    folder: 'GEDI_Biomass_Challenge',
    fileNamePrefix: 'gedi_l4a_tai_filtered',
    fileFormat: 'CSV',
    selectors: [
      'longitude', 'latitude', 'agbd', 'agbd_se', 'relative_se',
      'l4_quality_flag', 'degrade_flag', 'sensitivity',
      'predictor_limit_flag', 'response_limit_flag',
      'beam', 'shot_number', 'delta_time'
    ]
  });
});
