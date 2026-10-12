// ========================================================
// ALOS PALSAR-2 — FEATURES BIOMASSE
// Zone : Sud-Ouest Côte d'Ivoire / Taï
// Période : 2020-2024
// ========================================================


// ========================================================
// 1. Zone d'étude
// ========================================================

var roi = ee.Geometry.Rectangle([
  -7.85, 4.95,
  -6.35, 6.45
]);

Map.centerObject(roi, 8);


// ========================================================
// 2. Charger PALSAR-2 ScanSAR
// ========================================================

var palsar = ee.ImageCollection(
  'JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR'
)
.filterBounds(roi)
.filterDate('2020-01-01', '2025-01-01')


// ========================================================
// 3. Garder uniquement les acquisitions HH + HV
// ========================================================

// La collection est hétérogène : certaines images
// ne contiennent pas HV.
// On ne garde donc que les acquisitions double polarisation.

palsar = palsar
  .filter(
    ee.Filter.listContains(
      'Polarizations',
      'HH'
    )
  )
  .filter(
    ee.Filter.listContains(
      'Polarizations',
      'HV'
    )
  );

print(
  'Nombre d’images PALSAR-2 HH+HV 2020-2024:',
  palsar.size()
);

print(
  'Première image:',
  palsar.first()
);

print(
  'Bandes disponibles:',
  palsar.first().bandNames()
);


// ========================================================
// 4. Fonction de conversion DN -> gamma0 dB
// ========================================================

// Documentation JAXA / Earth Engine :
//
// gamma0 [dB] = 10 * log10(DN^2) - 83
//
// soit :
// gamma0 [dB] = 20 * log10(DN) - 83

function toDb(image) {

  var hh = image
    .select('HH')
    .float();

  var hv = image
    .select('HV')
    .float();


  var hhDb = hh
    .max(1)
    .log10()
    .multiply(20)
    .subtract(83)
    .rename('HH_dB');


  var hvDb = hv
    .max(1)
    .log10()
    .multiply(20)
    .subtract(83)
    .rename('HV_dB');


  return image
    .addBands(hhDb)
    .addBands(hvDb);
}


// ========================================================
// 5. Conversion de toute la collection
// ========================================================

var palsarDb = palsar.map(toDb);


// ========================================================
// 6. Composite médian 2020-2024
// ========================================================

// On prend la médiane temporelle pour réduire
// le bruit acquisition par acquisition.

var composite = palsarDb
  .select([
    'HH_dB',
    'HV_dB'
  ])
  .median()
  .clip(roi)
  .float();


// ========================================================
// 7. Features radar dérivées
// ========================================================

var hhDb = composite
  .select('HH_dB')
  .rename('PALSAR_HH_dB');


var hvDb = composite
  .select('HV_dB')
  .rename('PALSAR_HV_dB');


// --------------------------------------------------------
// Différence HH - HV en dB
// --------------------------------------------------------

var hhMinusHv = hhDb
  .subtract(hvDb)
  .rename('PALSAR_HH_minus_HV_dB');


// --------------------------------------------------------
// Ratio linéaire HV / HH
// --------------------------------------------------------

// Puisque les valeurs sont en dB :
//
// HV / HH en linéaire
// = 10 ^ ((HV_dB - HH_dB) / 10)

var hvHhRatio = ee.Image(10)
  .pow(
    hvDb
      .subtract(hhDb)
      .divide(10)
  )
  .rename('PALSAR_HV_HH_ratio');


// --------------------------------------------------------
// Ratio HH / HV
// --------------------------------------------------------

var hhHvRatio = ee.Image(10)
  .pow(
    hhDb
      .subtract(hvDb)
      .divide(10)
  )
  .rename('PALSAR_HH_HV_ratio');


// ========================================================
// 8. Stack final
// ========================================================

var palsarFeatures = ee.Image.cat([

  hhDb,
  hvDb,
  hhMinusHv,
  hvHhRatio,
  hhHvRatio

])
.float();

print(
  'Features PALSAR finales:',
  palsarFeatures.bandNames()
);

print(
  'Types de bandes:',
  palsarFeatures.bandTypes()
);


// ========================================================
// 9. Visualisation
// ========================================================

// HV est particulièrement intéressant pour la végétation.

Map.addLayer(
  hvDb,
  {
    min: -25,
    max: -5
  },
  'PALSAR HV dB'
);


// HH

Map.addLayer(
  hhDb,
  {
    min: -20,
    max: -3
  },
  'PALSAR HH dB',
  false
);


// Ratio HH-HV

Map.addLayer(
  hhMinusHv,
  {
    min: 0,
    max: 15
  },
  'PALSAR HH-HV dB',
  false
);


// ========================================================
// 10. Export GeoTIFF
// ========================================================

Export.image.toDrive({

  image: palsarFeatures,

  description:
    'PALSAR_Tai_features_2020_2024',

  folder:
    'GEDI_Biomass_Challenge',

  fileNamePrefix:
    'palsar_tai_features_2020_2024',

  region: roi,

  scale: 25,

  crs: 'EPSG:32629',

  maxPixels: 1e13,

  fileFormat: 'GeoTIFF',

  formatOptions: {
    cloudOptimized: true
  }

});