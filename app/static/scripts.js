// ======================================================
// Kolkata Traffic AI — Frontend Logic
// ======================================================

// ── Map init ────────────────────────────────────────────────────────────────
const map = L.map('map', { center: [22.5726, 88.3639], zoom: 13, zoomControl: false });
L.control.zoom({ position: 'bottomleft' }).addTo(map);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '© OpenStreetMap', maxZoom: 19
}).addTo(map);

// ── Constants ────────────────────────────────────────────────────────────────
const ROUTE_PALETTE = ['#1a73e8', '#ea4335', '#fbbc04', '#34a853', '#9c27b0'];
const RISK_COLOR = {
  CLEAR: '#34a853', LOW: '#34a853',
  MODERATE: '#e37400', HIGH: '#c5221f', CRITICAL: '#7c4dff'
};
const MODE_COLOR = {
  drive: '#1a73e8', walk: '#34a853', bike: '#e37400', metro: '#7c4dff', bus: '#FF5722'
};
const MODE_LABEL = {
  drive: 'Drive', walk: 'Walk', bike: 'Bike', metro: 'Metro', bus: 'Bus'
};

// Disruption zone colors by severity
const ZONE_COLOR = {
  high:   { fill: '#c5221f', stroke: '#8b0000' },
  medium: { fill: '#e37400', stroke: '#b35900' },
  low:    { fill: '#fbbc04', stroke: '#c49800' },
};
const ZONE_RADIUS = { high: 420, medium: 320, low: 220 };  // metres

// ── State ────────────────────────────────────────────────────────────────────
let routeLayers     = [];
let pinLayers       = [];
let markerLayers    = [];
let zoneLayers      = [];   // disruption area zones
let overlayLayers   = [];   // metro line/station overlay
let busOverlayLayers = [];  // bus stop markers + route polylines
let currentRoutes   = [];
let activeIdx       = 0;
let currentMode     = 'drive';
let currentComboMode = '';
let disruptionsLoaded   = false;
let lastDisruptionData  = null;
let _busNetwork         = null;  // cached {routes, stops} from /api/bus-network

// ── Mode selector ────────────────────────────────────────────────────────────
function setMode(mode) {
  const prevMode = currentMode;

  if (mode === 'metro+walk' || mode === 'metro+bike' || mode === 'metro+drive' || mode === 'metro+walk+drive') {
    currentMode = 'metro';
    currentComboMode = mode;
  } else {
    currentMode = mode;
    currentComboMode = '';
  }

  document.querySelectorAll('.mode-tab').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.mode === mode);
  });

  // Update input placeholders for the new mode
  _syncPlaceholders(mode);

  // ── Bus overlay: show all stops + routes when bus tab is selected ──────────
  if (mode === 'bus') {
    loadBusOverlay();
    document.getElementById('bus-legend').style.display  = '';
    document.getElementById('map-legend').style.display  = 'none';
  } else if (prevMode === 'bus') {
    // Leaving bus mode — remove bus overlay, restore empty-state panel
    clearBusOverlay();
    document.getElementById('bus-legend').style.display  = 'none';
    document.getElementById('map-legend').style.display  = '';
    document.getElementById('panel').innerHTML = `
      <div class="empty-state">
        <div class="icon">🗺️</div>
        <p>Choose your start and end points.<br>AI traffic routes appear here.</p>
      </div>`;
  }

  // Metro overlay is always visible — no-op here, loaded once on boot
}

// ── Load locations ───────────────────────────────────────────────────────────
let _allLocations  = [];   // {id, name, desc}
let _metroStations = [];   // {id, name, desc, line, color}
let _busStops      = [];   // {id, name, desc, lat, lon}

// Current confirmed values (what was actually selected, not just typed)
let _srcValue = '';
let _dstValue = '';

async function loadLocations() {
  try {
    const data = await (await fetch('/api/locations')).json();
    _allLocations  = data.locations      || [];
    _metroStations = data.metro_stations || [];
    _busStops      = data.bus_stops      || [];
    console.log(`[locations] ${_allLocations.length} localities, ${_metroStations.length} metro stations, ${_busStops.length} bus stops loaded`);
    _syncPlaceholders(currentMode);
  } catch(e) {
    console.error('[locations] fetch failed:', e);
    toast('Unable to load locations');
  }
}

function _syncPlaceholders(mode) {
  const isCombo = mode === 'metro+walk' || mode === 'metro+bike' || mode === 'metro+drive' || mode === 'metro+walk+drive';
  const hint = mode === 'metro'
    ? 'metro station'
    : isCombo
      ? 'locality or metro station'
      : mode === 'bus'
        ? 'bus stop or locality'
        : 'location';
  document.getElementById('src-input').placeholder = `Source ${hint}`;
  document.getElementById('dst-input').placeholder = `Destination ${hint}`;
}

// Build the flat option list for a given mode
function _buildOptions(mode) {
  const opts = [];
  // For pure metro: only stations. For combo (metro+walk/bike/drive) and
  // all other modes: show localities first, then all metro stations.
  const isCombo = mode === 'metro+walk' || mode === 'metro+bike' || mode === 'metro+drive' || mode === 'metro+walk+drive';
  const isBus   = mode === 'bus';

  if (mode !== 'metro') {
    // Localities group
    const localityLabel = isBus
      ? '📍 Kolkata Localities & Bus Stops'
      : isCombo
        ? '📍 Kolkata Localities & Metro Stations'
        : '📍 Kolkata Localities';
    opts.push({ _group: localityLabel });
    _allLocations.forEach(l => opts.push({ name: l.name, desc: l.desc, type: 'locality' }));
  }

  // Bus stops — only shown in bus mode
  if (isBus && _busStops.length) {
    opts.push({ _group: '🚌 Bus Stops' });
    _busStops.forEach(s => opts.push({
      name:  s.name,
      desc:  s.desc || 'Bus stop',
      type:  'bus',
      color: '#FF5722',
    }));
  }

  // Metro stations grouped by line — shown for metro / combo modes
  if (!isBus) {
    const LINE_META = {
      blue:   { label: '🔵 Blue Line (North-South)',              color: '#2196F3' },
      green:  { label: '🟢 Green Line (East-West)',               color: '#4CAF50' },
      purple: { label: '🟣 Purple Line (Joka-Majerhat)',          color: '#9C27B0' },
      orange: { label: '🟠 Orange Line (Kavi Subhash-Beleghata)', color: '#FF9800' },
      yellow: { label: '🟡 Yellow Line (Noapara-Jai Hind)',       color: '#FFC107' },
    };
    const lineOrder = ['blue', 'green', 'purple', 'orange', 'yellow'];

    lineOrder.forEach(line => {
      const stns = _metroStations.filter(s => s.line === line);
      if (!stns.length) return;
      opts.push({ _group: LINE_META[line]?.label || line });
      stns.forEach(s => opts.push({
        name:  s.name,
        desc:  s.desc,
        type:  'metro',
        line:  s.line,
        color: s.color || LINE_META[line]?.color || '#888',
      }));
    });
  }

  return opts;
}

// Render the dropdown list, optionally filtered by query
function _renderDropdown(which, query) {
  const dd   = document.getElementById(`${which}-dropdown`);
  // For combo modes (metro+walk/bike/drive) use the full combo mode so
  // _buildOptions knows to include both localities AND metro stations.
  const mode = currentComboMode || currentMode;
  const opts = _buildOptions(mode);
  const q    = (query || '').toLowerCase().trim();

  dd.innerHTML = '';

  // Show loading state if data hasn't arrived yet
  if (_allLocations.length === 0 && _metroStations.length === 0) {
    const loading = document.createElement('div');
    loading.className = 'loc-opt';
    loading.style.color = '#9aa0a6';
    loading.innerHTML = '<span class="loc-opt-pin">⏳</span><span>Loading locations…</span>';
    dd.appendChild(loading);
    return;
  }

  let anyResult = false;
  let pendingGroup = null;

  opts.forEach(opt => {
    if (opt._group) {
      pendingGroup = opt._group;
      return;
    }
    if (q && !opt.name.toLowerCase().includes(q)) return;

    // Flush the group header if this is the first item in the group
    if (pendingGroup) {
      const hdr = document.createElement('div');
      hdr.className = 'loc-grp-label';
      hdr.textContent = pendingGroup;
      dd.appendChild(hdr);
      pendingGroup = null;
    }

    anyResult = true;
    const row = document.createElement('div');
    row.className = 'loc-opt';
    row.dataset.value = opt.name;

    if (opt.type === 'metro') {
      row.innerHTML = `
        <span class="loc-opt-dot" style="background:${opt.color}"></span>
        <span>${opt.name}</span>`;
    } else if (opt.type === 'bus') {
      row.innerHTML = `
        <span class="loc-opt-dot" style="background:#FF5722"></span>
        <span>${opt.name}</span>`;
    } else {
      row.innerHTML = `
        <span class="loc-opt-pin">📍</span>
        <span>${opt.name}</span>`;
    }

    row.addEventListener('mousedown', (e) => {
      e.preventDefault();   // prevent blur firing before click
      selectOption(which, opt.name);
    });

    dd.appendChild(row);
  });

  if (!anyResult) {
    const empty = document.createElement('div');
    empty.className = 'loc-opt';
    empty.style.color = '#9aa0a6';
    empty.textContent = q ? `No matches for "${q}"` : 'No locations available';
    dd.appendChild(empty);
  }
}

function openDropdown(which) {
  const input = document.getElementById(`${which}-input`);
  const dd    = document.getElementById(`${which}-dropdown`);

  // Position the fixed dropdown under the input
  const rect = input.getBoundingClientRect();
  dd.style.top   = (rect.bottom + 6) + 'px';
  dd.style.left  = rect.left + 'px';
  dd.style.width = Math.max(rect.width + 28, 340) + 'px';  // a bit wider than input

  _renderDropdown(which, input.value);
  dd.classList.add('open');
}

function closeDropdown(which, delay = 0) {
  setTimeout(() => {
    const dd = document.getElementById(`${which}-dropdown`);
    dd.classList.remove('open');
    // If input text doesn't match a confirmed value, revert it
    const input = document.getElementById(`${which}-input`);
    const val   = which === 'src' ? _srcValue : _dstValue;
    if (input.value.trim() !== val) input.value = val;
  }, delay);
}

function filterDropdown(which) {
  const input = document.getElementById(`${which}-input`);
  const dd    = document.getElementById(`${which}-dropdown`);

  // Reposition in case the topbar shifted
  const rect = input.getBoundingClientRect();
  dd.style.top   = (rect.bottom + 6) + 'px';
  dd.style.left  = rect.left + 'px';
  dd.style.width = Math.max(rect.width + 28, 340) + 'px';

  _renderDropdown(which, input.value);
  dd.classList.add('open');
  // Clear the confirmed value while typing
  if (which === 'src') _srcValue = '';
  else _dstValue = '';
}

function selectOption(which, name) {
  document.getElementById(`${which}-input`).value = name;
  document.getElementById(`${which}-dropdown`).classList.remove('open');
  if (which === 'src') _srcValue = name;
  else _dstValue = name;
}

function swapLocations() {
  const tmp = _srcValue;
  _srcValue = _dstValue;
  _dstValue = tmp;
  document.getElementById('src-input').value = _srcValue;
  document.getElementById('dst-input').value = _dstValue;
}

// ── Metro overlay ────────────────────────────────────────────────────────────
async function loadMetroOverlay() {
  clearOverlay();
  try {
    const data = await (await fetch('/api/metro-overlay')).json();
    drawMetroOverlay(data);
    document.getElementById('map-legend').classList.add('visible');
  } catch (e) {
    console.warn('Metro overlay failed:', e);
  }
}

function drawMetroOverlay(geojson) {
  clearOverlay();
  geojson.features.forEach(f => {
    if (f.geometry.type === 'LineString') {
      const color = f.properties.color || '#9c27b0';
      const layer = L.geoJSON(f, {
        style: { color, weight: 4, opacity: 0.85, dashArray: null }
      }).addTo(map);
      layer.bindTooltip(f.properties.name, { permanent: false });
      overlayLayers.push(layer);
    } else if (f.geometry.type === 'Point') {
      const [lon, lat] = f.geometry.coordinates;
      const color = f.properties.color || '#9c27b0';
      const mk = L.circleMarker([lat, lon], {
        radius: 5, color: '#fff', weight: 1.5,
        fillColor: color, fillOpacity: 1,
      }).addTo(map);

      // Click station → fetch next trains
      const stnName = f.properties.name;
      const lineName = f.properties.line;
      mk.bindPopup(`<div style="min-width:200px">
        <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px">
          <div style="background:${color};width:10px;height:10px;border-radius:50%"></div>
          <b style="font-size:13px">${stnName}</b>
        </div>
        <div style="font-size:11px;color:${color};font-weight:600;margin-bottom:8px">
          ${lineName.charAt(0).toUpperCase() + lineName.slice(1)} Line
        </div>
        <div id="nt_${stnName.replace(/\s+/g,'_')}" style="font-size:12px;color:#555">
          <div style="display:flex;align-items:center;gap:6px;color:#9aa0a6">
            <div class="spinner" style="width:12px;height:12px;border-width:2px"></div>
            Loading next trains…
          </div>
        </div></div>`);

      mk.on('popupopen', async () => {
        const divId = `nt_${stnName.replace(/\s+/g,'_')}`;
        try {
          const res  = await fetch(`/api/next-metro/${encodeURIComponent(stnName)}?n=4`);
          const div  = document.getElementById(divId);
          if (!div) return;

          if (!res.ok) {
            // 404 = no trains at this station right now
            const err = await res.json().catch(() => ({}));
            div.innerHTML = `<div style="color:#e37400;font-size:11px">⚠ ${err.detail || 'No service now'}</div>`;
            return;
          }

          const data = await res.json();
          if (!data.trains || data.trains.length === 0) {
            div.innerHTML = '<div style="color:#e37400;font-size:11px">⚠ No trains running now</div>';
            return;
          }

          // Group by line+direction for cleaner display
          const rows = data.trains.slice(0, 4).map(t => {
            const away = t.minutes_away === 0
              ? '<span style="color:#34a853;font-weight:700">Now</span>'
              : `<span style="font-weight:700">${t.minutes_away} min</span>`;
            const dir  = t.direction === 'a_to_b' ? '→' : '←';
            return `
            <div style="display:flex;align-items:center;gap:6px;margin:4px 0;font-size:12px">
              <span style="background:${t.color};color:#fff;border-radius:4px;padding:1px 5px;
                font-size:10px;font-weight:700;min-width:16px;text-align:center">
                ${t.line.charAt(0).toUpperCase()}
              </span>
              <span style="flex:1;color:#202124">${dir} ${t.terminus}</span>
              <span style="color:#5f6368">${t.departure_time}</span>
              ${away}
            </div>`;
          }).join('');

          div.innerHTML = `
            <div style="color:#80868b;font-size:10px;margin-bottom:4px">
              Next trains · ${data.queried_at}
            </div>
            ${rows}`;
        } catch (e) {
          const div = document.getElementById(divId);
          if (div) div.innerHTML = '<i style="color:#9aa0a6">Could not load</i>';
        }
      });

      overlayLayers.push(mk);
    }
  });
}

function clearOverlay() {
  overlayLayers.forEach(l => map.removeLayer(l));
  overlayLayers = [];
}

// ── Bus overlay ──────────────────────────────────────────────────────────────
function clearBusOverlay() {
  busOverlayLayers.forEach(l => map.removeLayer(l));
  busOverlayLayers = [];
}

async function loadBusOverlay() {
  // Show the panel directory immediately (skeleton while loading)
  renderBusNetworkPanel(null);

  // Use cached data if available
  if (_busNetwork) {
    drawBusOverlay(_busNetwork);
    renderBusNetworkPanel(_busNetwork);
    return;
  }

  try {
    const data = await (await fetch('/api/bus-network')).json();
    _busNetwork = data;
    drawBusOverlay(data);
    renderBusNetworkPanel(data);
  } catch (e) {
    console.warn('[bus] overlay failed:', e);
    renderBusNetworkPanel({ routes: [], stops: [], error: true });
  }
}

// Route-type colour palette for bus lines on the map
const BUS_TYPE_COLOR = {
  'AC':         '#1a73e8',   // blue
  'Government': '#34a853',   // green
  'Non-AC':     '#FF5722',   // deep orange
  'Mini':       '#9c27b0',   // purple
};

function drawBusOverlay(data) {
  clearBusOverlay();

  const routes = data.routes || [];
  const stops  = data.stops  || [];

  // ── Draw route polylines (thin, semi-transparent, colour by type) ──────────
  routes.forEach(route => {
    if (!route.coords || route.coords.length < 2) return;
    const color = BUS_TYPE_COLOR[route.route_type] || '#FF5722';
    // coords are [lon, lat] — L.geoJSON expects [lat, lon], convert
    const latlngs = route.coords.map(([lon, lat]) => [lat, lon]);
    const line = L.polyline(latlngs, {
      color,
      weight:    3,
      opacity:   0.55,
      dashArray: '6 4',
    }).addTo(map);
    line.bindTooltip(
      `<b>🚌 ${route.route_name}</b><br>${route.num_stops} stops`,
      { sticky: true }
    );
    busOverlayLayers.push(line);
  });

  // ── Draw stop circle-markers (small orange dots with popup) ───────────────
  stops.forEach(stop => {
    const mk = L.circleMarker([stop.lat, stop.lon], {
      radius:      5,
      color:       '#fff',
      weight:      1.5,
      fillColor:   '#FF5722',
      fillOpacity: 0.9,
    }).addTo(map);

    // Find which routes serve this stop for the popup
    const servingRoutes = routes
      .filter(r => r.stops.some(s => s.stop_id === stop.stop_id))
      .map(r => `<div style="font-size:11px;color:#555;margin-top:2px">
          <span style="background:${BUS_TYPE_COLOR[r.route_type] || '#FF5722'}22;
            color:${BUS_TYPE_COLOR[r.route_type] || '#FF5722'};
            border-radius:4px;padding:1px 5px;font-size:10px;font-weight:600">
            ${r.route_type}
          </span>
          ${r.route_name}
        </div>`)
      .join('');

    mk.bindPopup(`
      <div style="min-width:180px">
        <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px">
          <div style="background:#FF5722;width:10px;height:10px;border-radius:50%"></div>
          <b style="font-size:13px">${stop.name}</b>
        </div>
        <div style="font-size:10px;color:#9aa0a6;margin-bottom:4px">Bus Stop</div>
        ${servingRoutes || '<div style="font-size:11px;color:#9aa0a6">No route data</div>'}
      </div>`);

    busOverlayLayers.push(mk);
  });
}

// ── Bus network panel (shown when bus mode is active but no route searched yet)
function renderBusNetworkPanel(data) {
  const panel = document.getElementById('panel');

  if (!data) {
    // Loading skeleton
    panel.innerHTML = `
      <div class="sec-lbl">🚌 Bus Network</div>
      <div class="loading-row">
        <div class="spinner"></div>
        <span>Loading bus routes &amp; stops…</span>
      </div>`;
    return;
  }

  if (data.error || !data.routes) {
    panel.innerHTML = `
      <div class="sec-lbl">🚌 Bus Network</div>
      <div class="empty-state"><div class="icon">⚠️</div>
        <p>Could not load bus network data.</p></div>`;
    return;
  }

  const routes = data.routes || [];
  const stops  = data.stops  || [];

  // Type colour helper
  const typeColor  = t => BUS_TYPE_COLOR[t] || '#FF5722';
  const typeBadge  = t => `
    <span style="background:${typeColor(t)}22;color:${typeColor(t)};
      border:1px solid ${typeColor(t)}44;
      border-radius:5px;padding:1px 6px;font-size:10px;font-weight:600">
      ${t || '—'}
    </span>`;

  let html = `
  <div class="sec-lbl">🚌 Kolkata Bus Network</div>
  <div style="font-size:11px;color:#9aa0a6;padding:0 12px 10px">
    ${routes.length} routes · ${stops.length} stops — tap a route to see stops
  </div>`;

  routes.forEach((route, ri) => {
    const color     = typeColor(route.route_type);
    const stopNames = route.stops.map(s => s.name).join(' → ');
    const firstStop = route.stops[0]?.name  || '?';
    const lastStop  = route.stops[route.stops.length - 1]?.name || '?';
    const detailId  = `bus-route-${ri}`;

    html += `
    <div class="route-card" style="cursor:pointer" onclick="toggleBusRouteDetail('${detailId}', ${ri})">
      <div class="rc-top">
        <div class="rc-num" style="background:${color};color:#fff;font-size:10px;min-width:28px">🚌</div>
        <div class="rc-title">
          <div class="rc-label" style="font-size:12px">${route.route_name}</div>
          <div class="rc-subtitle">${firstStop} → ${lastStop}</div>
        </div>
        ${typeBadge(route.route_type)}
      </div>
      <div class="rc-stats">
        <div class="stat-chip">${route.num_stops}<span class="lbl"> stops</span></div>
      </div>

      <!-- Expandable stop list -->
      <div id="${detailId}" style="display:none;margin-top:6px">
        <div class="expand-sec">All Stops</div>
        <div class="road-node-list">`;

    route.stops.forEach((stop, si) => {
      const isFirst = si === 0;
      const isLast  = si === route.stops.length - 1;
      const dot     = isFirst ? 'node-dot-start' : isLast ? 'node-dot-end' : 'node-dot-mid';
      html += `
          <div class="road-node-row">
            <div class="node-track">
              <div class="node-dot ${dot}" style="${isFirst||isLast ? `background:${color}` : ''}"></div>
              ${!isLast ? `<div class="node-line" style="background:${color}44"></div>` : ''}
            </div>
            <div class="node-label">
              ${isFirst ? '<b>' : ''}${stop.name}${isFirst ? '</b>' : ''}
              ${isFirst ? `<span style="font-size:10px;color:${color};font-weight:600;margin-left:4px">Start</span>` : ''}
              ${isLast  ? `<span style="font-size:10px;color:${color};font-weight:600;margin-left:4px">End</span>`   : ''}
            </div>
          </div>`;
    });

    html += `
        </div>
      </div>
    </div>`;
  });

  panel.innerHTML = html;
}

// Draw only stop markers (no route polylines) — used after a route search
// so stops remain visible alongside the computed route line.
function drawBusStopsOnly(stops) {
  clearBusOverlay();
  if (!stops || !_busNetwork) return;
  const routes = _busNetwork.routes || [];
  stops.forEach(stop => {
    const mk = L.circleMarker([stop.lat, stop.lon], {
      radius: 4, color: '#fff', weight: 1.2,
      fillColor: '#FF5722', fillOpacity: 0.85,
    }).addTo(map);
    const servingRoutes = routes
      .filter(r => r.stops.some(s => s.stop_id === stop.stop_id))
      .map(r => `<div style="font-size:11px;color:#555;margin-top:2px">
        <span style="background:${(BUS_TYPE_COLOR[r.route_type]||'#FF5722')}22;
          color:${BUS_TYPE_COLOR[r.route_type]||'#FF5722'};
          border-radius:4px;padding:1px 5px;font-size:10px;font-weight:600">${r.route_type}</span>
        ${r.route_name}</div>`)
      .join('');
    mk.bindPopup(`<div style="min-width:160px">
      <b style="font-size:12px">${stop.name}</b>
      <div style="font-size:10px;color:#9aa0a6;margin:2px 0 4px">Bus Stop</div>
      ${servingRoutes || '<div style="font-size:11px;color:#9aa0a6">No route data</div>'}
    </div>`);
    busOverlayLayers.push(mk);
  });
}

function toggleBusRouteDetail(id, routeIdx) {  const el = document.getElementById(id);
  if (!el) return;
  const isOpen = el.style.display !== 'none';
  el.style.display = isOpen ? 'none' : 'block';

  // Pan map to highlight this route's polyline
  if (!isOpen && _busNetwork && busOverlayLayers.length) {
    const route = _busNetwork.routes[routeIdx];
    if (route && route.coords && route.coords.length > 1) {
      const latlngs = route.coords.map(([lon, lat]) => [lat, lon]);
      const bounds  = L.latLngBounds(latlngs);
      map.fitBounds(bounds, { paddingTopLeft: [380, 60], paddingBottomRight: [360, 60] });
    }
  }
}

// ── Main entry ───────────────────────────────────────────────────────────────
async function go() {
  const src = _srcValue.trim();
  const dst = _dstValue.trim();
  if (!src || !dst) { toast('Select source and destination'); return; }
  if (src === dst)  { toast('Source and destination cannot be the same'); return; }

  setBtn(true, 'Loading…');
  setStatus(`Computing ${currentMode} routes…`);
  clearMap();
  disruptionsLoaded  = false;
  lastDisruptionData = null;

  renderLoading(src, dst, currentMode);

  // ── Step 1: routes ───────────────────────────────────────────────────────
  let routeData;
  try {
    const payload = { source: src, destination: dst };
    if (currentComboMode) {
      payload.modes = currentComboMode.split('+');
    } else {
      payload.mode = currentMode;
    }
    const res = await fetch('/api/route', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Routing failed'); }
    routeData = await res.json();
  } catch (e) {
    toast(e.message); setBtn(false, 'Get Routes'); renderError(e.message); return;
  }

  currentRoutes = routeData.routes;
  activeIdx     = Math.max(0, currentRoutes.findIndex(r => r.is_best));

  drawRoutes(routeData);
  renderPanel(currentRoutes, [], false, null, src, dst);
  setBtn(false, 'Get Routes');

  const isMetroCombo = currentComboMode.startsWith('metro+');

  // Metro-only: no disruption analysis needed — show journey detail immediately
  if (currentMode === 'metro' && !isMetroCombo) {
    const comboLabels = {
      'metro+walk':       'Metro+Walk',
      'metro+bike':       'Metro+Bike',
      'metro+drive':      'Metro+Drive',
      'metro+walk+drive': 'Walk+Metro+Cab',
    };
    const modeStr = comboLabels[currentComboMode] || 'Metro';
    setStatus(`${src} → ${dst} · ${modeStr} route`);
    return;
  }

  // Bus: no disruption analysis — routes are stop-based, show immediately
  if (currentMode === 'bus') {
    setStatus(`${src} → ${dst} · Bus route`);
    renderPanel(currentRoutes, [], true, null, src, dst);
    // Restore stop markers so user can see nearby stops alongside their route
    if (_busNetwork) drawBusStopsOnly(_busNetwork.stops);
    return;
  }

  // ── Step 2: disruptions (drive/walk/bike or metro+walk/bike) ───────────
  setStatus('Fetching disruptions…');
  try {
    const res = await fetch('/api/disruptions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        source: src, destination: dst,
        mode: currentComboMode || currentMode,
        road_names: routeData.all_road_names || [],
        routes:     routeData.routes,
      }),
    });
    if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Disruption fetch failed'); }
    const data = await res.json();

    currentRoutes      = data.routes || currentRoutes;
    activeIdx          = Math.max(0, currentRoutes.findIndex(r => r.is_best));
    lastDisruptionData = data;
    disruptionsLoaded  = true;

    updateRouteColors(currentRoutes);
    drawMarkers(data.markers || []);
    renderPanel(currentRoutes, data.markers || [], true, data.city_weather, src, dst);
    setStatus(`${src} → ${dst} · ${data.recent_events || 0} disruptions`);
  } catch (e) {
    toast('Disruption analysis failed: ' + e.message);
    setStatus(`${src} → ${dst} · Routes shown`);
  }
}

// ── Map drawing ──────────────────────────────────────────────────────────────
function drawRoutes(data) {
  // Clear only route/pin layers — keep overlay
  routeLayers.forEach(l => map.removeLayer(l));
  pinLayers.forEach(l => map.removeLayer(l));
  markerLayers.forEach(l => map.removeLayer(l));
  routeLayers = []; pinLayers = []; markerLayers = [];

  const mode = data.mode || currentMode;

  data.routes.forEach((route, i) => {
    const isActive = i === activeIdx;
    const baseColor = route.risk_color || MODE_COLOR[mode] || ROUTE_PALETTE[i % 5];
    const w    = isActive ? 7 : 4;
    const op   = isActive ? 1 : 0.45;

    // Multi-segment GeoJSON (metro+feeder): each feature has its own colour
    const hasSegmentColors = route.geojson &&
      route.geojson.features &&
      route.geojson.features.some(f => f.properties && f.properties.color);

    let layer;
    if (hasSegmentColors) {
      // Render each feature with its own segment colour
      const featureGroup = L.featureGroup();
      route.geojson.features.forEach(f => {
        const segColor = f.properties.color || baseColor;
        const segDash  = f.properties.segment === 'metro' ? null : (isActive ? '6 4' : '8 5');
        const segW     = f.properties.segment === 'metro' ? w + 1 : w - 1;
        const segOp    = isActive ? 1 : 0.45;
        const segLayer = L.geoJSON(f, {
          style: { color: segColor, weight: segW, opacity: segOp,
                   dashArray: segDash, lineJoin: 'round', lineCap: 'round' }
        });
        featureGroup.addLayer(segLayer);
      });
      featureGroup.addTo(map);
      featureGroup.on('click', () => selectRoute(i));
      featureGroup.bindTooltip(
        `<b>${route.label}</b><br>${route.distance_km} km · ${route.travel_time_min} min`,
        { sticky: true }
      );
      layer = featureGroup;
    } else {
      // Standard single-colour route
      const dash = (mode === 'metro') ? '6 4' : (isActive ? null : '8 5');
      layer = L.geoJSON(route.geojson, {
        style: { color: baseColor, weight: w, opacity: op,
                 dashArray: dash, lineJoin: 'round', lineCap: 'round' }
      }).addTo(map);
      layer.on('click', () => selectRoute(i));
      layer.bindTooltip(
        `<b>${route.label}</b><br>${route.distance_km} km · ${route.travel_time_min} min`,
        { sticky: true }
      );
    }

    routeLayers.push(layer);
  });

  const sc = data.src_coords, dc = data.dst_coords;
  if (sc) pinLayers.push(L.marker([sc[0], sc[1]], { icon: pin('🟢') }).addTo(map).bindPopup(`<b>Start</b><br>${data.source}`));
  if (dc) pinLayers.push(L.marker([dc[0], dc[1]], { icon: pin('🔴') }).addTo(map).bindPopup(`<b>End</b><br>${data.destination}`));

  if (routeLayers[activeIdx]) {
    // paddingTopLeft: leave room for search card (left) + some top margin
    // paddingBottomRight: leave room for route panel (right) + status bar
    map.fitBounds(routeLayers[activeIdx].getBounds(), {
      paddingTopLeft:     [380, 60],
      paddingBottomRight: [360, 60],
    });
    routeLayers[activeIdx].bringToFront();
  }
}

function updateRouteColors(routes) {
  const mode = currentComboMode || currentMode;
  routes.forEach((r, i) => {
    if (!routeLayers[i]) return;
    const isActive = i === activeIdx;
    const baseColor = r.risk_color || MODE_COLOR[currentMode] || ROUTE_PALETTE[i % 5];

    // FeatureGroup (multi-segment) — restyle each child layer
    if (routeLayers[i].eachLayer) {
      routeLayers[i].eachLayer(child => {
        if (child.setStyle) {
          const segColor = child.feature?.properties?.color || baseColor;
          child.setStyle({
            color:     segColor,
            weight:    isActive ? 7 : 4,
            opacity:   isActive ? 1 : 0.4,
            dashArray: isActive ? null : '8 5',
          });
        } else if (child.eachLayer) {
          // nested feature group
          child.eachLayer(grandchild => {
            if (grandchild.setStyle) {
              const sc = grandchild.feature?.properties?.color || baseColor;
              grandchild.setStyle({
                color:     sc,
                weight:    isActive ? 7 : 4,
                opacity:   isActive ? 1 : 0.4,
                dashArray: isActive ? null : '8 5',
              });
            }
          });
        }
      });
    } else if (routeLayers[i].setStyle) {
      routeLayers[i].setStyle({
        color:     baseColor,
        weight:    isActive ? 7 : 4,
        opacity:   isActive ? 1 : 0.4,
        dashArray: isActive ? null : '8 5',
      });
    }
    if (isActive && routeLayers[i].bringToFront) routeLayers[i].bringToFront();
  });
}

function drawMarkers(markers) {
  markerLayers.forEach(l => map.removeLayer(l));
  zoneLayers.forEach(l => map.removeLayer(l));
  markerLayers = [];
  zoneLayers   = [];

  // ── Group markers by severity for zone drawing ────────────────────────────
  const withCoords = markers.filter(m => m.lat && m.lon);

  // Draw area zones first (underneath markers)
  withCoords.forEach(m => {
    const sev   = m.severity || 'low';
    const zc    = ZONE_COLOR[sev] || ZONE_COLOR.low;
    const zr    = ZONE_RADIUS[sev] || 220;

    // Outer glow circle (large, very transparent)
    const glow = L.circle([m.lat, m.lon], {
      radius:      zr * 2.2,
      color:       zc.fill,
      weight:      0,
      fillColor:   zc.fill,
      fillOpacity: 0.06,
      interactive: false,
    }).addTo(map);
    zoneLayers.push(glow);

    // Main zone circle
    const zone = L.circle([m.lat, m.lon], {
      radius:      zr,
      color:       zc.stroke,
      weight:      1.2,
      opacity:     0.5,
      fillColor:   zc.fill,
      fillOpacity: 0.18,
      dashArray:   sev === 'high' ? null : '4 3',
    }).addTo(map);

    zone.bindTooltip(
      `<b style="color:${zc.fill}">${sev.toUpperCase()} — ${m.event_type.replace(/_/g,' ')}</b><br>` +
      `📍 ${m.location}`,
      { sticky: true, opacity: 0.95 }
    );
    zoneLayers.push(zone);
  });

  // Draw point markers on top of zones
  withCoords.forEach(m => {
    const sev  = m.severity || 'low';
    const zc   = ZONE_COLOR[sev] || ZONE_COLOR.low;
    const icon = { accident:'💥', congestion:'🚗', road_closure:'🚧',
                   construction:'🏗️', protest:'✊', weather:'🌧️',
                   waterlogging:'💧', vip_movement:'🚨', metro_disruption:'🚇',
                   train_delay:'🚂', transport_strike:'✋', diversion:'↪️' };
    const emoji = icon[m.event_type] || '⚠️';

    // Pulse ring for high severity
    if (sev === 'high') {
      const pulse = L.circleMarker([m.lat, m.lon], {
        radius: 14, color: zc.fill, weight: 2,
        fillColor: 'transparent', fillOpacity: 0, opacity: 0.4,
        className: 'pulse-ring',
      }).addTo(map);
      zoneLayers.push(pulse);
    }

    // Emoji icon marker
    const mk = L.marker([m.lat, m.lon], {
      icon: L.divIcon({
        html: `
          <div style="
            background:${zc.fill};
            border:2px solid ${zc.stroke};
            border-radius:50%;
            width:28px;height:28px;
            display:flex;align-items:center;justify-content:center;
            font-size:14px;
            box-shadow:0 2px 6px rgba(0,0,0,0.35);
            cursor:pointer;
          ">${emoji}</div>`,
        className: '',
        iconSize:   [28, 28],
        iconAnchor: [14, 14],
      }),
    }).addTo(map);

    const srcLabel = m.source === 'tomtom_traffic' ? '📡 TomTom Live'
                   : m.source === 'newsapi'         ? '📰 NewsAPI'
                   :                                  '📡 RSS';
    const dur = m.duration ? ` · ${m.duration}` : '';
    const urlLink = m.tomtom_url
      ? `<br><a href="${m.tomtom_url}" target="_blank" style="font-size:10px;color:#1a73e8">View on TomTom ↗</a>`
      : '';

    mk.bindPopup(`
      <div style="min-width:180px">
        <div style="font-weight:700;color:${zc.fill};margin-bottom:4px">
          ${emoji} ${m.event_type.replace(/_/g,' ').toUpperCase()}
          ${m.is_future ? ' <span style="background:#e8f0fe;color:#1a73e8;padding:1px 5px;border-radius:8px;font-size:10px">Upcoming</span>' : ''}
        </div>
        <div style="font-size:12px;margin-bottom:3px">📍 <b>${m.location}</b></div>
        <div style="font-size:12px;color:#3c4043;margin-bottom:5px">${m.reason}</div>
        <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap">
          <span style="background:${zc.fill}22;color:${zc.fill};padding:2px 7px;border-radius:10px;font-size:10px;font-weight:600">
            ${sev.toUpperCase()}
          </span>
          <span style="font-size:10px;color:#5f6368">${m.age_label}${dur}</span>
        </div>
        <div style="font-size:10px;color:#5f6368;margin-top:4px">${srcLabel}${urlLink}</div>
      </div>
    `, { maxWidth: 260 });

    markerLayers.push(mk);
  });

  // ── Draw disruption zone legend ───────────────────────────────────────────
  _updateZoneLegend(withCoords);
}

// ── Panel rendering ───────────────────────────────────────────────────────────
function renderLoading(src, dst, mode) {
  const comboLabels = {
    'metro+walk':       '🚇+🚶 Metro+Walk',
    'metro+bike':       '🚇+🚲 Metro+Bike',
    'metro+drive':      '🚇+🚗 Metro+Drive',
    'metro+walk+drive': '🚶+🚇+🚕 Walk+Metro+Cab',
    'bus':              '🚌 Bus',
  };
  const modeLabel = comboLabels[currentComboMode] || comboLabels[mode] || MODE_LABEL[mode] || mode;
  document.getElementById('panel').innerHTML = `
    <div class="sec-lbl">${modeLabel} Routes</div>
    <div class="loading-row">
      <div class="spinner"></div>
      <span>Finding ${modeLabel.toLowerCase()} routes from <strong>${src}</strong> to <strong>${dst}</strong>…</span>
    </div>`;
}

function renderError(msg) {
  document.getElementById('panel').innerHTML = `
    <div class="empty-state"><div class="icon">⚠️</div><p>${msg}</p></div>`;
}

function renderPanel(routes, markers, ready, weather, src, dst) {
  const mode  = currentComboMode || currentMode;
  const comboLabels = {
    'metro+walk':       '🚇+🚶 Metro+Walk',
    'metro+bike':       '🚇+🚲 Metro+Bike',
    'metro+drive':      '🚇+🚗 Metro+Drive',
    'metro+walk+drive': '🚶+🚇+🚕 Walk+Metro+Cab',
    'bus':              '🚌 Bus',
  };
  const mLabel = comboLabels[mode] || MODE_LABEL[currentMode] || currentMode;

  let html = '';
  const isPureBus   = mode === 'bus';
  const isPureMetro = mode === 'metro';
  const note = (ready || isPureMetro || isPureBus)
    ? ''
    : ' <span style="font-size:10px;font-weight:400;text-transform:none;letter-spacing:0;color:#9aa0a6">— analysing…</span>';

  html += `<div class="sec-lbl">${mLabel} Routes (${routes.length})${note}</div>`;

  routes.forEach((r, i) => {
    html += buildRouteCard(r, i, ready, markers, mode);
  });

  // Weather banner — only relevant for drive/bike (not metro/walk/bus)
  if (ready && weather && (mode === 'drive' || mode === 'bike')) {
    html += buildWeatherBanner(weather);
  }

  // Disruption loading row — skip for pure metro and bus (no disruption analysis)
  if (!ready && mode !== 'metro' && mode !== 'bus') {
    html += `
    <div class="divider"></div>
    <div class="loading-row">
      <div class="spinner"></div>
      <span>Fetching TomTom, news &amp; running AI analysis…</span>
    </div>`;
  }

  document.getElementById('panel').innerHTML = html;
}

// ── Route card ───────────────────────────────────────────────────────────────
function buildRouteCard(r, i, ready, markers, mode) {
  const isActive  = i === activeIdx;
  const isBest    = !!r.is_best;
  const riskLevel = r.risk_level || '';
  const riskColor = RISK_COLOR[riskLevel] || '#80868b';
  const evtCount  = r.event_count || 0;
  const modeColor = MODE_COLOR[currentMode] || '#1a73e8';

  // Event count strip
  // Pure metro: no disruption analysis — just label it
  // Metro combos (metro+walk/bike/drive): DO run disruption analysis, show results
  // Walk: show as walking route (low relevance for disruptions)
  // Bus: no disruption analysis — label it
  let evtClass = '', evtIcon = '', evtText = '';
  const isPureMetro  = mode === 'metro';
  const isPureBus    = mode === 'bus';
  const isMetroCombo = mode === 'metro+walk' || mode === 'metro+bike' || mode === 'metro+drive' || mode === 'metro+walk+drive';
  if (isPureMetro) {
    evtClass = 'ok'; evtIcon = '✓';
    evtText  = 'Metro route';
  } else if (isPureBus) {
    evtClass = 'ok'; evtIcon = '✓';
    evtText  = 'Bus route';
  } else if (mode === 'walk') {
    evtClass = 'ok'; evtIcon = '✓'; evtText = 'Walking route';
  } else if (!ready) {
    evtIcon = '⏳'; evtText = 'Analysing…';
  } else if (evtCount === 0) {
    evtClass = 'ok'; evtIcon = '✓'; evtText = 'No disruptions';
  } else if (riskLevel === 'HIGH' || riskLevel === 'CRITICAL') {
    const specific = r.route_specific_events || 0;
    const area     = r.area_wide_events || 0;
    evtClass = 'critical'; evtIcon = '⛔';
    evtText  = specific > 0
      ? `${specific} on route · ${area} area-wide`
      : `${evtCount} disruption${evtCount > 1 ? 's' : ''}`;
  } else {
    const specific = r.route_specific_events || 0;
    const area     = r.area_wide_events || 0;
    evtClass = 'warn'; evtIcon = '⚠';
    evtText  = specific > 0
      ? `${specific} on route · ${area} area-wide`
      : `${evtCount} disruption${evtCount > 1 ? 's' : ''}`;
  }

  // Subtitle
  let subtitle = '';
  if ((isPureMetro || isMetroCombo) && r.segments) {
    const metro = r.segments.find(s => s.type === 'metro');
    let comboSubtitle;
    if (mode === 'metro+walk+drive') {
      comboSubtitle = 'Walk → Metro → Cab';
    } else {
      const feederPart = mode.split('+')[1] || 'walk';
      const feederLabel = feederPart.charAt(0).toUpperCase() + feederPart.slice(1);
      comboSubtitle = `Metro + ${feederLabel}`;
    }
    subtitle = metro ? `${metro.from} → ${metro.to}` : comboSubtitle;
  } else if (isPureBus) {
    subtitle = r.src_stop && r.dst_stop
      ? `${r.src_stop} → ${r.dst_stop}`
      : 'via bus stops';
  } else {
    subtitle = 'via ' + ((r.road_names || []).slice(0, 2).join(', ') || 'city roads');
  }

  const roadsText = (r.road_names || []).slice(0, 4).join(' · ') || '—';

  let html = `
  <div class="route-card ${isActive ? 'active' : ''} ${isBest ? 'is-best' : ''}"
       id="rc-${i}" onclick="selectRoute(${i})">

    <div class="rc-top">
      <div class="rc-num" style="${isActive ? `background:${modeColor}` : ''}">${i + 1}</div>
      <div class="rc-title">
        <div class="rc-label">${r.label}</div>
        <div class="rc-subtitle">${subtitle}</div>
      </div>
      ${isBest ? '<div class="best-badge">Best</div>' : ''}
      ${(ready && riskLevel && !isPureMetro && !isPureBus)
        ? `<div class="risk-pill" style="background:${riskColor}18;color:${riskColor}">${riskLevel}</div>`
        : ''}
    </div>

    <div class="rc-stats">
      <div class="stat-chip">${r.travel_time_min}<span class="lbl"> min</span></div>
      <div class="stat-chip">${r.distance_km}<span class="lbl"> km</span></div>
      ${(ready && riskLevel && !isPureMetro && !isPureBus)
        ? `<div class="stat-chip" style="color:${riskColor}">${r.risk_score || 0}<span class="lbl" style="color:${riskColor}99"> risk</span></div>`
        : ''}
      ${isPureBus && r.num_stops != null
        ? `<div class="stat-chip">${r.num_stops}<span class="lbl"> stops</span></div>`
        : ''}
    </div>

    ${(!isPureMetro && !isPureBus) ? `<div class="rc-roads">${roadsText}</div>` : ''}
    <div class="rc-events ${evtClass}">${evtIcon} ${evtText}</div>`;

  // Expanded detail for the active card
  if (isActive) {
    if (isPureMetro || isMetroCombo) {
      html += buildMetroJourney(r);
    } else if (isPureBus) {
      html += buildBusJourney(r);
    } else {
      html += buildExpandedDetail(r, ready);
    }
  }

  html += '</div>';
  return html;
}

// ── Bus journey breakdown ─────────────────────────────────────────────────────
function buildBusJourney(r) {
  const segments  = r.segments  || [];
  const busStops  = r.bus_stops || [];
  const matchingRoutes = r.matching_routes || [];

  let html = '<div class="metro-journey">';
  html += '<div class="expand-sec">Journey Breakdown</div>';

  // Matching bus route numbers badge row
  if (matchingRoutes.length) {
    const badges = matchingRoutes.slice(0, 3).map(rt =>
      `<span style="
        background:#FF572222;color:#FF5722;
        border:1px solid #FF572244;
        border-radius:6px;padding:2px 8px;
        font-size:11px;font-weight:600;
        margin-right:4px">
        🚌 ${rt.route_name}
      </span>`
    ).join('');
    html += `<div style="margin:4px 0 8px;display:flex;flex-wrap:wrap;gap:2px">${badges}</div>`;
  }

  // Stop-by-stop list
  if (segments.length) {
    const total = segments.length;
    segments.forEach((seg, i) => {
      const isFirst = seg.is_first || i === 0;
      const isLast  = seg.is_last  || i === total - 1;
      const dot     = isFirst ? 'node-dot-start' : isLast ? 'node-dot-end' : 'node-dot-mid';
      html += `
      <div class="road-node-row">
        <div class="node-track">
          <div class="node-dot ${dot}" style="${isFirst || isLast ? 'background:#FF5722' : ''}"></div>
          ${!isLast ? '<div class="node-line" style="background:#FF572244"></div>' : ''}
        </div>
        <div class="node-label">
          ${isFirst ? '<b>' : ''}${seg.name}${isFirst ? '</b>' : ''}
          ${isFirst  ? ' <span style="font-size:10px;color:#FF5722;font-weight:600">Board</span>'  : ''}
          ${isLast   ? ' <span style="font-size:10px;color:#FF5722;font-weight:600">Alight</span>' : ''}
        </div>
      </div>`;
    });
  } else if (busStops.length) {
    // Fallback: plain stop-name list
    busStops.forEach((name, i) => {
      const isFirst = i === 0, isLast = i === busStops.length - 1;
      const dot = isFirst ? 'node-dot-start' : isLast ? 'node-dot-end' : 'node-dot-mid';
      html += `
      <div class="road-node-row">
        <div class="node-track">
          <div class="node-dot ${dot}" style="${isFirst || isLast ? 'background:#FF5722' : ''}"></div>
          ${!isLast ? '<div class="node-line"></div>' : ''}
        </div>
        <div class="node-label">${name}</div>
      </div>`;
    });
  }

  // bus_note (e.g. "no direct route" fallback message)
  if (r.bus_note) {
    html += `<div class="metro-note-banner">${r.bus_note}</div>`;
  }

  html += '</div>';
  return html;
}

// ── Metro journey breakdown ───────────────────────────────────────────────────
function buildMetroJourney(r) {
  const segments = r.segments || [];
  if (segments.length === 0) {
    // Fallback: just show station list
    return buildExpandedDetail(r, true);
  }

  let html = '<div class="metro-journey">';
  html += '<div class="expand-sec">Journey Breakdown</div>';

  const isLast = (i) => i === segments.length - 1;

  segments.forEach((seg, i) => {
    const isMetro = seg.type === 'metro';
    const icon = isMetro ? '🚇' : (seg.type === 'bike' ? '🚲' : seg.type === 'drive' ? '🚕' : '🚶');
    const lineClass = isMetro
      ? `<span class="metro-line-pill ${seg.line || 'blue'}">${(seg.line || 'blue').toUpperCase()} LINE</span>`
      : '';

    let detail = '';
    if (isMetro && seg.stations) {
      const stops = seg.stations;
      const count = seg.num_stops || stops.length - 1;
      detail = `${stops[0]} → ${stops[stops.length - 1]} · ${count} stop${count !== 1 ? 's' : ''}`;
      if (seg.stations.length > 2) {
        const midStops = stops.slice(1, -1).join(', ');
        detail += `<br><small style="color:#9aa0a6">${midStops}</small>`;
      }
      // Next train chip
      if (seg.next_train) {
        const nt = seg.next_train;
        const away = nt.minutes_away === 0
          ? '<span style="color:#34a853">Departing now</span>'
          : `<span style="color:#1a73e8">Boards in ${nt.minutes_away} min</span>`;
        detail += `<br><div style="margin-top:4px;font-size:11px">
          🚇 Next: <b>${nt.departure_time}</b> · ${away}
        </div>`;
      }
    } else {
      detail = `${seg.from} → ${seg.to} · ${seg.distance_km} km`;
      if (seg.cab_note) {
        detail += `<br><small style="color:#1a73e8">🚕 ${seg.cab_note}</small>`;
      }
    }

    const segTypeLabel = isMetro ? 'Metro'
      : seg.type === 'bike'  ? 'Bike'
      : seg.type === 'drive' ? 'Cab / Drive'
      : 'Walk';

    html += `
    <div class="journey-seg">
      <div class="seg-track">
        <div class="seg-icon ${seg.type}">${icon}</div>
        ${!isLast(i) ? `<div class="seg-connector ${seg.type}"></div>` : ''}
      </div>
      <div class="seg-body">
        <div class="seg-title">${segTypeLabel} ${lineClass}</div>
        <div class="seg-detail">${detail}</div>
        <div class="seg-time-chip">${seg.time_min} min · ${seg.distance_km} km</div>
      </div>
    </div>`;
  });

  // Interchange note — uses the dynamic note from the backend (e.g. "Change at Esplanade")
  if (r.interchange && r.interchange_note) {
    html += `
    <div class="metro-note-banner">
      🔄 ${r.interchange_note}
    </div>`;
  }

  // Disruption flags from context-aware routing
  if (r.disruption_flags && r.disruption_flags.length > 0) {
    r.disruption_flags.forEach(flag => {
      html += `<div class="metro-note-banner" style="background:#fce8e6;border-color:#ea4335;color:#c5221f">
        ⚠️ ${flag}
      </div>`;
    });
  }

  // Metro note
  if (r.metro_note) {
    html += `<div class="metro-note-banner">${r.metro_note}</div>`;
  }

  html += '</div>';
  return html;
}

// ── Expanded road + disruption detail ────────────────────────────────────────
function buildExpandedDetail(r, ready) {
  let html = '<div class="card-expand">';

  const allRoads = r.road_names || [];
  if (allRoads.length > 0) {
    html += `<div class="expand-sec">Road Segments (${allRoads.length})</div>`;
    html += '<div class="road-node-list">';
    allRoads.forEach((road, idx) => {
      const first = idx === 0, last = idx === allRoads.length - 1;
      const dot   = first ? 'node-dot-start' : last ? 'node-dot-end' : 'node-dot-mid';
      html += `
      <div class="road-node-row">
        <div class="node-track">
          <div class="node-dot ${dot}"></div>
          ${!last ? '<div class="node-line"></div>' : ''}
        </div>
        <div class="node-label">${road}</div>
      </div>`;
    });
    html += '</div>';
  }

  if (ready) {
    const allActive = (r.matched_events || []).filter(e => !e.is_future_event);
    const future    = (r.matched_events || []).filter(e =>  e.is_future_event);

    // Split into route-specific and area-wide
    const specific  = allActive.filter(e => e.route_specific);
    const areaWide  = allActive.filter(e => !e.route_specific);

    if (allActive.length === 0) {
      html += '<div class="expand-sec">Disruptions on this Route (0)</div>';
      html += '<div class="expand-clear">✓ No active disruptions</div>';
    } else {
      // Route-specific disruptions
      if (specific.length > 0) {
        html += `
        <div class="expand-sec">
          On This Route
          <span class="sec-badge specific">${specific.length}</span>
        </div>`;
        specific.forEach(ev => { html += buildEventCard(ev); });
      }

      // Area-wide disruptions (collapsible)
      if (areaWide.length > 0) {
        const areaId = `area-${r.id || Math.random()}`;
        html += `
        <div class="expand-sec area-sec" onclick="toggleAreaEvents('${areaId}')">
          Area-wide (affect all routes)
          <span class="sec-badge area">${areaWide.length}</span>
          <span class="area-toggle" id="tog-${areaId}">▼</span>
        </div>
        <div id="${areaId}" class="area-events-list">`;
        areaWide.forEach(ev => { html += buildEventCard(ev, false, true); });
        html += '</div>';
      }
    }

    if (future.length > 0) {
      html += `<div class="expand-sec">Upcoming (${future.length})</div>`;
      future.forEach(ev => { html += buildEventCard(ev, true); });
    }
  } else {
    html += `
    <div class="loading-row" style="padding:8px 0 4px">
      <div class="spinner"></div>
      <span style="font-size:12px">Analysing disruptions…</span>
    </div>`;
  }

  html += '</div>';
  return html;
}

function toggleAreaEvents(id) {
  const el  = document.getElementById(id);
  const tog = document.getElementById('tog-' + id);
  if (!el) return;
  const hidden = el.style.display === 'none' || el.style.display === '';
  el.style.display   = hidden ? 'block' : 'none';
  if (tog) tog.textContent = hidden ? '▲' : '▼';
}

function buildEventCard(ev, isFuture = false, isAreaWide = false) {
  const isLive = ev.source === 'tomtom_traffic';
  const dur    = ev.impact_duration_label ? ` · ${ev.impact_duration_label}` : '';
  const src    = isLive ? '📡 TomTom Live' : ev.source === 'newsapi' ? '📰 NewsAPI' : '📡 RSS';
  const ftag   = isFuture  ? '<span class="future-tag">Upcoming</span>' : '';
  const atag   = isAreaWide ? '<span class="area-tag">Area-wide</span>'  : '';
  const corrTag = ev.severity_corrected ? '<span class="hgnn-tag">HGNN ✓</span>' : '';
  return `
  <div class="ev-item ${ev.severity} ${isAreaWide ? 'area-wide-ev' : ''}">
    <div class="ev-head">
      <span class="ev-type" style="color:${ev.color || '#5f6368'}">${ev.event_type.replace(/_/g, ' ')}${ftag}${atag}${corrTag}</span>
      <span class="ev-age">${ev.age_label || ''}</span>
    </div>
    <div class="ev-loc">📍 ${ev.location}</div>
    <div class="ev-reason">${ev.reason}</div>
    <div class="ev-meta">
      <span class="${isLive ? 'live' : ''}">${src}</span>
      ${dur ? `<span>${dur}</span>` : ''}
    </div>
  </div>`;
}

// ── Weather banner ────────────────────────────────────────────────────────────
function buildWeatherBanner(weather) {
  if (!weather || !weather.success) return '';
  const sev    = (weather.severity || 'unknown').toLowerCase();
  const wsi    = weather.avg_wsi != null ? weather.avg_wsi.toFixed(2) : 'N/A';
  const score  = weather.score || 0;
  const raw    = weather.raw || {};
  const temp   = raw.temp   != null ? `${raw.temp}°C` : '';
  const cond   = raw.condition || '';
  const humid  = raw.humidity  != null ? `${raw.humidity}% humidity` : '';
  const wind   = raw.wind_speed != null ? `${raw.wind_speed} m/s wind` : '';
  const rain   = raw.rain_1h > 0 ? `${raw.rain_1h} mm/h rain` : '';
  const icon   = sev === 'high' ? '🌧️' : sev === 'medium' ? '🌦️'
               : cond === 'Thunderstorm' ? '⛈️' : cond === 'Fog' || cond === 'Haze' ? '🌫️' : '☀️';
  const detail = [temp, cond, rain, wind, humid].filter(Boolean).join(' · ');
  return `
  <div class="divider"></div>
  <div class="weather-banner">
    <div class="weather-icon">${icon}</div>
    <div class="weather-body">
      <div class="weather-title">City-wide Weather · Kolkata</div>
      <div class="weather-detail">${detail || 'Conditions normal'}</div>
      <div style="display:flex;align-items:center;gap:8px;margin-top:5px;flex-wrap:wrap">
        <div class="weather-sev ${sev}">${sev.toUpperCase()}</div>
        <div style="font-size:11px;color:#5f6368">WSI ${wsi} · +${score} pts on all routes</div>
      </div>
      <div class="weather-note">City-wide — applied equally to every route</div>
    </div>
  </div>`;
}

// ── Route selection ──────────────────────────────────────────────────────────
function selectRoute(i) {
  activeIdx = i;
  updateRouteColors(currentRoutes);
  if (routeLayers[i]) {
    map.fitBounds(routeLayers[i].getBounds(), {
      paddingTopLeft:     [380, 60],
      paddingBottomRight: [360, 60],
    });
    routeLayers[i].bringToFront();
  }
  renderPanel(
    currentRoutes,
    lastDisruptionData?.markers || [],
    disruptionsLoaded,
    lastDisruptionData?.city_weather || null,
    _srcValue,
    _dstValue,
  );
}

// ── Helpers ──────────────────────────────────────────────────────────────────
function clearMap() {
  routeLayers.forEach(l => map.removeLayer(l));
  pinLayers.forEach(l => map.removeLayer(l));
  markerLayers.forEach(l => map.removeLayer(l));
  zoneLayers.forEach(l => map.removeLayer(l));
  routeLayers = []; pinLayers = []; markerLayers = []; zoneLayers = [];
  // Also clear bus stop overlay when a route search starts
  clearBusOverlay();
  // Remove zone legend
  const leg = document.getElementById('zone-legend');
  if (leg) leg.remove();
}

function pin(emoji) {
  return L.divIcon({
    html: `<div style="font-size:22px;line-height:1">${emoji}</div>`,
    className: '', iconSize: [26, 26], iconAnchor: [13, 13],
  });
}

function setBtn(disabled, text) {
  const b = document.getElementById('btn-go');
  b.disabled = disabled; b.innerText = text;
}

function setStatus(text) {
  document.getElementById('status-l').innerText = text;
}

function toast(msg) {
  const t = document.getElementById('toast');
  t.innerText = msg; t.style.display = 'block';
  setTimeout(() => t.style.display = 'none', 3500);
}

// ── Disruption zone legend ────────────────────────────────────────────────────
function _updateZoneLegend(markers) {
  // Remove old legend
  const old = document.getElementById('zone-legend');
  if (old) old.remove();

  if (!markers || markers.length === 0) return;

  const counts = { high: 0, medium: 0, low: 0 };
  markers.forEach(m => {
    const s = m.severity || 'low';
    if (counts[s] !== undefined) counts[s]++;
  });

  const rows = Object.entries(counts)
    .filter(([, c]) => c > 0)
    .map(([sev, cnt]) => {
      const zc = ZONE_COLOR[sev] || ZONE_COLOR.low;
      return `
      <div class="zleg-row">
        <div class="zleg-swatch" style="background:${zc.fill};border:1.5px solid ${zc.stroke}"></div>
        <span>${sev.charAt(0).toUpperCase()+sev.slice(1)}</span>
        <span class="zleg-count">${cnt}</span>
      </div>`;
    }).join('');

  const leg = document.createElement('div');
  leg.id        = 'zone-legend';
  leg.className = 'zone-legend';
  leg.innerHTML = `
    <div class="zleg-title">Disruption Zones</div>
    ${rows}
    <div class="zleg-note">Radius = severity</div>
  `;
  document.body.appendChild(leg);
}

// ── Boot ─────────────────────────────────────────────────────────────────────
loadLocations();
loadMetroOverlay();
document.getElementById('map-legend').classList.add('visible');
