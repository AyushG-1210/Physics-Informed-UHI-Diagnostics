/**
 * PHASE 1 — NODE GENERATION AT 1 km  (MODIS-matched)
 *
 * WHY 1 km AND NOT 500 m
 * The 500 m grid was chosen to be fine. It was finer than any target
 * available, and that is what produced the illusion of spatial skill:
 * ERA5 at ~11 km gave ~7 distinct values across 3,679 nodes, so the
 * graph was recovering which coarse cell a node sat in, not urban
 * structure. Sampling a target BELOW its native resolution
 * manufactures structure that is not there.
 *
 * MODIS LST is 1 km native. Matching the node grid to it means every
 * node carries one genuinely independent observation. Roughly 900
 * nodes per city instead of 3,300-3,700 -- smaller, and honest.
 *
 * WHAT CARRIES OVER UNCHANGED
 *  - grid built in EPSG:32643 so cells are equal-area (bld_cover is
 *    an area ratio and must not be latitude-distorted)
 *  - bld_cover as the mean of a painted 0/1 footprint raster
 *  - property names <= 10 chars so nothing is truncated on export
 *  - zip-based ids rather than O(n^2) indexOf
 *
 * WHAT IS DROPPED
 *  - the ee.Join.saveAll producing bld_count. It is a spatial join
 *    over every building centroid, it dominated runtime (Delhi took
 *    ~2.5 h), and bld_count was never in FEATURE_COLS. Removing it
 *    should cut this to minutes.
 *
 * Run once per city by editing CITY and region.
 */

// ---- CONFIG ------------------------------------------------------
var CITY = 'Blr';
var region = ee.Geometry.Rectangle([77.45, 12.85, 77.75, 13.10]);
// Hyd: ee.Geometry.Rectangle([78.30, 17.30, 78.60, 17.50]);
// Dlh: ee.Geometry.Rectangle([77.05, 28.50, 77.35, 28.75]);

var CELL_M  = 1000;   // matches MODIS LST native resolution
var COVER_M = 10;     // rasterisation scale for coverage fraction
var UTM     = 'EPSG:32643';

Map.centerObject(region, 12);

// ---- 1. 1 km grid in a PROJECTED CRS -----------------------------
var grid = region.coveringGrid(ee.Projection(UTM).atScale(CELL_M));

// ---- 2. Buildings ------------------------------------------------
var buildings = ee.FeatureCollection('GOOGLE/Research/open-buildings/v3/polygons')
  .filterBounds(region)
  .filter(ee.Filter.gte('confidence', 0.70));

// Coverage fraction: mean of a painted 0/1 raster IS the area
// fraction covered, provided the raster is reprojected first.
var builtMask = ee.Image(0).byte()
  .paint(buildings, 1)
  .rename('bld_cover')
  .reproject({crs: UTM, scale: COVER_M});

var heightImg = ee.ImageCollection('GOOGLE/Research/open-buildings-temporal/v1')
  .select('building_height')
  .filterDate('2023-01-01', '2023-12-31')
  .median()
  .rename('bld_height')
  .reproject({crs: UTM, scale: COVER_M});

// ---- 3. Aggregate onto the grid ----------------------------------
var withCover = builtMask.reduceRegions({
  collection: grid,
  reducer: ee.Reducer.mean().setOutputs(['bld_cover']),
  scale: COVER_M,
  crs: UTM
});

var withHeight = heightImg.reduceRegions({
  collection: withCover,
  reducer: ee.Reducer.mean().setOutputs(['bld_height']),
  scale: COVER_M,
  crs: UTM
});

// ---- 4. Stable integer ids ---------------------------------------
var lst = withHeight.toList(withHeight.size());
var idx = ee.List.sequence(1, withHeight.size());

var nodes = ee.FeatureCollection(
  idx.zip(lst).map(function (pair) {
    pair = ee.List(pair);
    var f = ee.Feature(pair.get(1));
    return f.set({
      'id':         ee.Number(pair.get(0)).format('%d'),
      'bld_cover':  ee.Number(ee.Algorithms.If(f.get('bld_cover'), f.get('bld_cover'), 0)),
      'bld_height': ee.Number(ee.Algorithms.If(f.get('bld_height'), f.get('bld_height'), 0))
    });
  })
);

// ---- 5. SANITY CHECKS -- read before exporting -------------------
print('node count (expect ~800-1000 for a 0.30 x 0.25 deg window):',
      nodes.size());
print('bld_cover stats — expect max ~0.4-0.7, nonzero median.',
      'A max near 0.01 means the paint/reproject step failed:',
      nodes.aggregate_stats('bld_cover'));
print('bld_height stats (metres):', nodes.aggregate_stats('bld_height'));

// At 1 km, coverage is averaged over a larger area than at 500 m, so
// the MAXIMUM will be lower than the 500 m grid's 0.59-0.69 -- dense
// cores get diluted by their surroundings. The MEDIAN should be
// similar. A much lower median means something is wrong.

Map.addLayer(nodes, {color: 'red'}, 'nodes', false);
Map.addLayer(builtMask.selfMask(), {palette: ['black']}, 'building mask', false);

// ---- 6. Export ---------------------------------------------------
// toAsset will NOT overwrite an existing assetId. Use a fresh name;
// a silent failure here is what caused Phase 2 to read a stale asset
// twice already.
Export.table.toDrive({
  collection: nodes,
  description: CITY + '_Nodes_1km',
  fileFormat: 'GeoJSON'
});

Export.table.toAsset({
  collection: nodes,
  description: CITY + '_Nodes_1km_Asset',
  assetId: 'projects/uhi-paper-488715/assets/' + CITY + '_Nodes_1km'
});

print('Exports queued. Confirm in the Assets tab that the new asset');
print('shows bld_cover and bld_height before running Phase 2.');