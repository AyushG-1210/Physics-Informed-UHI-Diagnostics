/**
 * PHASE 2 — MODIS LST EXTRACTION  (v3, 60-day windows)
 *
 * WHY THE WINDOW GREW
 * A 14-day window yielded only 13-26 usable night passes per config.
 * After requiring nodes to be observed on at least 60% of passes,
 * Bengaluru April fell from 973 nodes to 211 with 34% of its target
 * matrix interpolated, and training ran on 9 overlapping windows over
 * 111 nodes. Any k-sweep on that is confounded: with 211 nodes, k=16
 * is close to a global average, so "more neighbours" and "more data"
 * cannot be told apart.
 *
 * 60 days gives roughly 4x the passes -- Terra and Aqua each provide
 * one night overpass per day, so ~120 before cloud, and 40-90 after.
 * Nodes then clear a coverage threshold on REAL observations instead
 * of the threshold being lowered to admit interpolated ones.
 *
 * WHAT DOES NOT CHANGE
 * The Landsat validation scene stays anchored to the original target
 * dates. It is a single held-out snapshot; widening its search would
 * only move it further from the study period.
 *
 * NO2 NOW MATCHES THE TARGET WINDOW. In v2 a one-month NO2 mean was
 * attached to a 14-day window; leaving that unchanged would have
 * spread a one-month mean across two months of observations.
 *
 * LIMITATION TO STATE IN THE PAPER
 * NDVI, NDBI and the Landsat scene remain SINGLE SNAPSHOTS while the
 * target now spans 60 days. Vegetation drifts over that period,
 * particularly across the pre-monsoon transition, so the morphological
 * predictors are a fixed description of a surface that is changing.
 * That is a real limitation. It is smaller than training on 9 windows,
 * but it belongs in the text rather than being left for a reviewer.
 */

// ---- CONFIG ------------------------------------------------------
var ROOT = 'projects/uhi-paper-488715/assets/';

var CITIES = {
  Blr: {
    region: ee.Geometry.Rectangle([77.45, 12.85, 77.75, 13.10]),
    nodesAsset: ROOT + 'Blr_Nodes_1km', lon: 77.60
  },
  Hyd: {
    region: ee.Geometry.Rectangle([78.30, 17.30, 78.60, 17.50]),
    nodesAsset: ROOT + 'Hyd_Nodes_1km', lon: 78.45
  },
  Dlh: {
    region: ee.Geometry.Rectangle([77.05, 28.50, 77.35, 28.75]),
    nodesAsset: ROOT + 'Dlh_Nodes_1km', lon: 77.20
  }
};

var REQUIRED_NODE_PROPS = ['bld_cover', 'bld_height'];
var MAX_CLOUD = 30, MIN_CLEAR_FRAC = 0.90, SCORE_SCALE = 90;

var SELECTORS = [
  'id', 'timestamp', 'pass', 'hour_utc', 'local_hour', 'view_time',
  'lst_modis', 'qc_ok', 'qc_strict',
  'bld_cover', 'bld_height',
  'NDVI', 'NDBI', 'lst_landsat',
  'traffic_no2_proxy',
  't2m', 'u10', 'v10', 'ssr', 'str', 'slhf', 'lai_hi', 'lai_lo'
];

// ---- BLOCKING ASSET CHECK ----------------------------------------
var assetsOk = true;
print('=== NODE ASSET CHECK ===');
Object.keys(CITIES).forEach(function (k) {
  var fc = ee.FeatureCollection(CITIES[k].nodesAsset);
  var props, n;
  try {
    props = ee.Feature(fc.first()).propertyNames().getInfo();
    n = fc.size().getInfo();
  } catch (e) {
    print('  ' + k + ': CANNOT READ ' + CITIES[k].nodesAsset + ' — ' + e);
    assetsOk = false; return;
  }
  var missing = REQUIRED_NODE_PROPS.filter(function (p) {
    return props.indexOf(p) < 0;
  });
  print('  ' + k + ': ' + n + ' nodes — ' + props.join(', '));
  if (missing.length) { print('    !! MISSING ' + missing.join(', ')); assetsOk = false; }
});
if (!assetsOk) { print('!! ABORTED — fix asset ids. No tasks created.'); }

// ---- MODIS -------------------------------------------------------
function modisPass(col, lstBand, qcBand, vtBand, passName, region, lon) {
  return col.map(function (img) {
    var qc = img.select(qcBand);
    var mandatory = qc.bitwiseAnd(3);            // bits 0-1
    var usable = mandatory.lte(1);               // good OR other quality
    var strict = mandatory.eq(0);                // good only

    var lst = img.select(lstBand).multiply(0.02).subtract(273.15)
      .updateMask(usable).rename('lst_modis');
    var vt = img.select(vtBand).multiply(0.1).rename('view_time'); // local solar h

    // MOD11A1/MYD11A1 are DAILY COMPOSITES: system:time_start is
    // midnight, not the overpass. Matching ERA5 to it would give every
    // pass the same midnight forcing, so the real view time is used.
    var meanLocal = ee.Number(vt.reduceRegion({
      reducer: ee.Reducer.mean(), geometry: region,
      scale: 1000, maxPixels: 1e9, bestEffort: true
    }).get('view_time'));

    var utcHour = ee.Algorithms.If(
      meanLocal,
      ee.Number(meanLocal).subtract(ee.Number(lon).divide(15)).mod(24),
      null);

    return lst.addBands(vt)
      .addBands(strict.rename('qc_strict').toByte())
      .addBands(usable.rename('qc_ok').toByte())
      .set('pass', passName)
      .set('local_hour', meanLocal)
      .set('hour_utc', utcHour)
      .set('system:time_start', img.get('system:time_start'));
  }).filter(ee.Filter.notNull(['hour_utc']));
}

function modisCollection(region, start, end, lon) {
  var terra = ee.ImageCollection('MODIS/061/MOD11A1')
    .filterBounds(region).filterDate(start, end);
  var aqua = ee.ImageCollection('MODIS/061/MYD11A1')
    .filterBounds(region).filterDate(start, end);
  return ee.ImageCollection(
    modisPass(terra, 'LST_Day_1km',   'QC_Day',   'Day_view_time',   'terra_day',   region, lon)
      .merge(modisPass(terra, 'LST_Night_1km', 'QC_Night', 'Night_view_time', 'terra_night', region, lon))
      .merge(modisPass(aqua,  'LST_Day_1km',   'QC_Day',   'Day_view_time',   'aqua_day',    region, lon))
      .merge(modisPass(aqua,  'LST_Night_1km', 'QC_Night', 'Night_view_time', 'aqua_night',  region, lon))
  ).sort('system:time_start');
}

// ---- Landsat (held-out validation only) --------------------------
function maskL2(img) {
  var qa = img.select('QA_PIXEL');
  return img.updateMask(
    qa.bitwiseAnd(1 << 1).eq(0).and(qa.bitwiseAnd(1 << 2).eq(0))
      .and(qa.bitwiseAnd(1 << 3).eq(0)).and(qa.bitwiseAnd(1 << 4).eq(0)));
}

function landsatLayer(region, start, end, target, tag) {
  var refPixels = region.area(1).divide(SCORE_SCALE * SCORE_SCALE);
  var t = ee.Date(target).millis();
  var cands = ee.ImageCollection(
      ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
        .merge(ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')))
    .filterBounds(region).filterDate(start, end)
    .filter(ee.Filter.lt('CLOUD_COVER', MAX_CLOUD))
    .map(function (img) {
      var px = maskL2(img).select('ST_B10').reduceRegion({
        reducer: ee.Reducer.count(), geometry: region,
        scale: SCORE_SCALE, maxPixels: 1e10, bestEffort: true
      }).get('ST_B10');
      return img.set({
        'clear_frac': ee.Number(px).divide(refPixels),
        'dt_days': ee.Number(img.get('system:time_start'))
                     .subtract(t).abs().divide(86400000)
      });
    });
  var usable = cands.filter(ee.Filter.gte('clear_frac', MIN_CLEAR_FRAC))
                    .sort('dt_days');
  if (usable.size().getInfo() === 0) {
    print('  !! ' + tag + ': no Landsat scene clears ' + MIN_CLEAR_FRAC);
    return null;
  }
  var scene = maskL2(ee.Image(usable.first()));
  print('  ' + tag + ' Landsat (validation only):',
        scene.get('DATE_ACQUIRED'), 'clear:', scene.get('clear_frac'));
  // rho = 0.0000275*DN - 0.2 BEFORE the normalised differences: the
  // offset cancels in the numerator but not the denominator.
  var sr = scene.select(['SR_B4', 'SR_B5', 'SR_B6'])
    .multiply(0.0000275).add(-0.2);
  return ee.Image.cat([
    scene.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15)
      .rename('lst_landsat'),
    sr.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI'),
    sr.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI')
  ]);
}

// ---- Extraction --------------------------------------------------
function extract(cfg, win, no2win, l8win, l8target, exportName) {
  var region = cfg.region;
  var nodes = ee.FeatureCollection(cfg.nodesAsset);

  print('--- ' + exportName + ' ---');
  var modis = modisCollection(region, win[0], win[1], cfg.lon);
  print('  MODIS observations (expect ~200-240 over 60 days,',
        'roughly half of them night):', modis.size());

  var statics = landsatLayer(region, l8win[0], l8win[1], l8target, exportName);
  if (statics === null) { return; }

  var no2 = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2')
    .filterBounds(region).filterDate(no2win[0], no2win[1])
    .select('NO2_column_number_density').mean().rename('traffic_no2_proxy');

  var era5 = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
    .filterBounds(region)
    .filterDate(ee.Date(win[0]).advance(-1, 'day'),
                ee.Date(win[1]).advance(1, 'day'))
    .select(['temperature_2m', 'u_component_of_wind_10m',
             'v_component_of_wind_10m', 'surface_net_solar_radiation_hourly',
             'surface_net_thermal_radiation_hourly',
             'surface_latent_heat_flux_hourly',
             'leaf_area_index_high_vegetation',
             'leaf_area_index_low_vegetation'],
            ['t2m', 'u10', 'v10', 'ssr', 'str', 'slhf', 'lai_hi', 'lai_lo']);

  var table = modis.map(function (img) {
    var day = ee.Date(img.get('system:time_start'));
    var obsTime = day.advance(ee.Number(img.get('hour_utc')), 'hour');
    var atm = ee.Image(
      era5.map(function (e) {
        return e.set('dt', ee.Number(e.get('system:time_start'))
                             .subtract(obsTime.millis()).abs());
      }).sort('dt').first());

    return img.addBands(atm).addBands(statics).addBands(no2)
      .reduceRegions({
        collection: nodes,
        reducer: ee.Reducer.mean(),
        scale: 1000                 // MODIS native; do NOT oversample
      })
      .map(function (f) {
        return f.set('timestamp', obsTime.millis())
                .set('pass', img.get('pass'))
                .set('hour_utc', img.get('hour_utc'))
                .set('local_hour', img.get('local_hour'));
      });
  }).flatten();

  Export.table.toDrive({
    collection: table,
    description: exportName,
    fileFormat: 'CSV',
    selectors: SELECTORS
  });
}

// ---- RUNS --------------------------------------------------------
// 60-day target windows. NO2 spans the same period. Landsat target
// dates are unchanged, so the held-out validation scene stays put.
var runs = [
  { city: 'Blr', tag: 'Blr_April_MODIS_60d',
    win: ['2025-03-01', '2025-04-30'], no2: ['2025-03-01', '2025-04-30'],
    l8: ['2025-03-08', '2025-05-08'], l8target: '2025-04-08' },
  { city: 'Blr', tag: 'Blr_December_MODIS_60d',
    win: ['2025-11-15', '2026-01-15'], no2: ['2025-11-15', '2026-01-15'],
    l8: ['2025-11-09', '2026-01-08'], l8target: '2025-12-09' },
  { city: 'Hyd', tag: 'Hyd_April_MODIS_60d',
    win: ['2025-03-01', '2025-04-30'], no2: ['2025-03-01', '2025-04-30'],
    l8: ['2025-03-08', '2025-05-08'], l8target: '2025-04-08' },
  { city: 'Hyd', tag: 'Hyd_December_MODIS_60d',
    win: ['2025-11-15', '2026-01-15'], no2: ['2025-11-15', '2026-01-15'],
    l8: ['2025-11-09', '2026-01-08'], l8target: '2025-12-09' },
  { city: 'Dlh', tag: 'Dlh_April_MODIS_60d',
    win: ['2025-03-01', '2025-04-30'], no2: ['2025-03-01', '2025-04-30'],
    l8: ['2025-03-08', '2025-05-08'], l8target: '2025-04-08' },
  { city: 'Dlh', tag: 'Dlh_December_MODIS_60d',
    win: ['2025-11-15', '2026-01-15'], no2: ['2025-11-15', '2026-01-15'],
    l8: ['2025-11-09', '2026-01-08'], l8target: '2025-12-09' }
];

if (assetsOk) {
  runs.forEach(function (r) {
    extract(CITIES[r.city], r.win, r.no2, r.l8, r.l8target, r.tag);
  });
  print('');
  print('Six tasks, ~4x the rows of the 14-day version. Expect a');
  print('noticeably longer export -- the row count scales with passes.');
  print('');
  print('AFTER EXTRACTION, the number to check is night passes per');
  print('config: 40-90 rather than 13-26. If a config still comes back');
  print('under ~30, its sky is the limit, not the window, and that');
  print('config should be reported with its coverage stated.');
  print('');
  print('The season labels now cover Mar-Apr and mid-Nov to mid-Jan.');
  print('Describe them that way in the paper rather than as "April"');
  print('and "December".');
}