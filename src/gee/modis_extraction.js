/**
 * PHASE 2 — MODIS LST EXTRACTION  (v2, overpass-time corrected)
 *
 * THE BUG IN v1
 * MOD11A1/MYD11A1 are DAILY COMPOSITES. Their system:time_start is
 * midnight of the compositing day, not the moment of overpass. v1
 * matched ERA5 to that value, so EVERY observation -- day and night,
 * Terra and Aqua -- received the midnight ERA5 hour. The physics head
 * would have seen identical atmospheric forcing for a 13:30 daytime
 * retrieval and a 01:30 nighttime one. The evidence was visible in
 * the v1 output: 14 timestamps per pass, all at 00:00:00, identical
 * across day and night.
 *
 * THE FIX
 * MODIS carries the real overpass time per pixel in Day_view_time and
 * Night_view_time: LOCAL SOLAR hours, scale factor 0.1. These are
 * converted to UTC using the window's central longitude
 * (UTC = local_solar - lon/15) and rounded to the nearest hour, which
 * is ERA5's resolution. Each observation then gets the atmospheric
 * state that actually accompanied it, and the diurnal cycle the
 * physics head needs is present.
 *
 * The view time is a per-pixel band. Its mean over the region is used
 * for the ERA5 match -- overpass time varies by only a few minutes
 * across a 30 km window, well inside ERA5's hourly resolution -- while
 * the per-node value is exported as `view_time` so the assumption is
 * checkable rather than assumed.
 *
 * ALSO NEW
 *  - `qc_strict` alongside `qc_ok`. v1 kept mandatory QA <= 1, but
 *    only 0.6-15.6% of retrievals were flagged GOOD; the rest are
 *    "other quality", which for MODIS LST often means error above
 *    2 K. Both flags are exported so the strict subset can be tested
 *    without re-extracting.
 *  - `hour_utc` and `local_hour`, so the diurnal structure can be
 *    inspected directly.
 *
 * UNCHANGED: 1 km nodes matched to MODIS native resolution, Landsat
 * held out for validation only, reflectance scaling before normalised
 * differences, blocking asset check, explicit selectors.
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
/**
 * One image per (satellite, pass). The view-time band gives the real
 * overpass hour in LOCAL SOLAR time; converting to UTC needs the
 * longitude, since local solar time is defined by the sun's position.
 */
function modisPass(col, lstBand, qcBand, vtBand, passName, region, lon) {
  return col.map(function (img) {
    var qc = img.select(qcBand);
    var mandatory = qc.bitwiseAnd(3);           // bits 0-1
    var usable = mandatory.lte(1);              // good OR other quality
    var strict = mandatory.eq(0);               // good only

    var lst = img.select(lstBand).multiply(0.02).subtract(273.15)
      .updateMask(usable).rename('lst_modis');

    var vt = img.select(vtBand).multiply(0.1).rename('view_time');  // local solar h

    // Region-mean overpass hour -> UTC. Across a ~30 km window the
    // overpass time varies by only minutes, far inside ERA5's hourly
    // step, so a single scalar per image is adequate. The per-node
    // value is still exported so this can be verified.
    var meanLocal = ee.Number(vt.reduceRegion({
      reducer: ee.Reducer.mean(), geometry: region,
      scale: 1000, maxPixels: 1e9, bestEffort: true
    }).get('view_time'));

    var utcHour = ee.Algorithms.If(
      meanLocal,
      ee.Number(meanLocal).subtract(ee.Number(lon).divide(15)).mod(24),
      null);

    return lst
      .addBands(vt)
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
  print('  MODIS observations:', modis.size());
  print('  overpass hours (LOCAL SOLAR) by pass — expect roughly',
        'terra_day 10.5, aqua_day 13.5, terra_night 22.5, aqua_night 1.5:');
  print('   ', modis.aggregate_array('pass'));
  print('   ', modis.aggregate_array('local_hour'));

  var statics = landsatLayer(region, l8win[0], l8win[1], l8target, exportName);
  if (statics === null) { return; }

  var no2 = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2')
    .filterBounds(region).filterDate(no2win[0], no2win[1])
    .select('NO2_column_number_density').mean().rename('traffic_no2_proxy');

  // ERA5 is a PREDICTOR of the regional baseline here, not the target.
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
    // Composite date at midnight, plus the REAL overpass hour.
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
        scale: 1000                   // MODIS native; do NOT oversample
      })
      .map(function (f) {
        return f.set('timestamp', obsTime.millis())   // real overpass time
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
var runs = [
  { city: 'Blr', tag: 'Blr_April_MODIS_v2',
    win: ['2025-04-01', '2025-04-15'], no2: ['2025-03-01', '2025-04-15'],
    l8: ['2025-03-08', '2025-05-08'], l8target: '2025-04-08' },
  { city: 'Blr', tag: 'Blr_December_MODIS_v2',
    win: ['2025-12-02', '2025-12-16'], no2: ['2025-11-01', '2025-12-16'],
    l8: ['2025-11-09', '2026-01-08'], l8target: '2025-12-09' },
  { city: 'Hyd', tag: 'Hyd_April_MODIS_v2',
    win: ['2025-04-01', '2025-04-15'], no2: ['2025-03-01', '2025-04-15'],
    l8: ['2025-03-08', '2025-05-08'], l8target: '2025-04-08' },
  { city: 'Hyd', tag: 'Hyd_December_MODIS_v2',
    win: ['2025-12-02', '2025-12-16'], no2: ['2025-11-01', '2025-12-16'],
    l8: ['2025-11-09', '2026-01-08'], l8target: '2025-12-09' },
  { city: 'Dlh', tag: 'Dlh_April_MODIS_v2',
    win: ['2025-04-01', '2025-04-15'], no2: ['2025-03-01', '2025-04-15'],
    l8: ['2025-03-08', '2025-05-08'], l8target: '2025-04-08' },
  { city: 'Dlh', tag: 'Dlh_December_MODIS_v2',
    win: ['2025-12-02', '2025-12-16'], no2: ['2025-11-01', '2025-12-16'],
    l8: ['2025-11-09', '2026-01-08'], l8target: '2025-12-09' }
];

if (assetsOk) {
  runs.forEach(function (r) {
    extract(CITIES[r.city], r.win, r.no2, r.l8, r.l8target, r.tag);
  });
  print('');
  print('BEFORE RUNNING TASKS — check the printed local_hour arrays.');
  print('Expected, roughly: terra_day ~10.5, aqua_day ~13.5,');
  print('terra_night ~22.5, aqua_night ~1.5 (local solar hours).');
  print('If they all read the same value, the view-time band is not');
  print('being picked up and the fix has not taken.');
  print('');
  print('AFTER EXTRACTION, verify the fix worked: t2m should now VARY');
  print('by pass. Group by pass and check its mean -- daytime passes');
  print('must be warmer than nighttime ones. In v1 they were identical,');
  print('because every observation received the midnight ERA5 hour.');
}