// ============================================================
//  MOTOR DE MAQUILLAJE — 100% en el navegador
//  Usa MediaPipe Face Landmarker (468 puntos de la cara).
//  No genera una cara nueva: pinta encima de la tuya.
// ============================================================

import { FaceLandmarker, FilesetResolver }
  from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.mjs'

let landmarker = null

// --- Indices de la malla facial de MediaPipe ---
const LABIOS_FUERA = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146]
const LABIOS_DENTRO = [78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95]

const OVALO = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,
               377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]

const OJO_IZQ  = [33,246,161,160,159,158,157,173,133]
const CEJA_IZQ = [55,65,52,53,46]
const OJO_DER  = [263,466,388,387,386,385,384,398,362]
const CEJA_DER = [285,295,282,283,276]

// Cejas completas (contorno superior + inferior)
const CEJA_IZQ_FULL = [70,63,105,66,107,55,65,52,53,46]
const CEJA_DER_FULL = [300,293,334,296,336,285,295,282,283,276]

const MEJILLA_IZQ = 50
const MEJILLA_DER = 280

// --- Intensidades ---
const NIVELES = {
  suave:     { labios:.38, sombra:.28, rubor:.20, brillo:.5, purpurina:70,  arcoiris:.10 },
  fabuloso:  { labios:.58, sombra:.45, rubor:.32, brillo:.8, purpurina:150, arcoiris:.18 },
  extra:     { labios:.78, sombra:.62, rubor:.45, brillo:1,  purpurina:260, arcoiris:.28 },
  drag:      { labios:.95, sombra:.88, rubor:.55, brillo:1.15, purpurina:320, arcoiris:.22,
               delineado:1, pestanas:1, contorno:.55, sobredibujo:1.05, cutCrease:1 }
}

// ------------------------------------------------------------
//  Carga del modelo (una sola vez)
// ------------------------------------------------------------
async function preparar () {
  if (landmarker) return landmarker
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm'
  )
  landmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
      delegate: 'GPU'
    },
    runningMode: 'IMAGE',
    numFaces: 4
  })
  return landmarker
}

// ------------------------------------------------------------
//  Utilidades de dibujo
// ------------------------------------------------------------
function trazar (ctx, puntos, indices, w, h) {
  indices.forEach((idx, i) => {
    const p = puntos[idx]
    const x = p.x * w, y = p.y * h
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
  })
  ctx.closePath()
}

function lienzo (w, h) {
  const c = document.createElement('canvas')
  c.width = w; c.height = h
  return c
}

function fundir (destino, capa, modo, alfa, desenfoque) {
  destino.save()
  destino.globalCompositeOperation = modo
  destino.globalAlpha = alfa
  if (desenfoque) destino.filter = `blur(${desenfoque}px)`
  destino.drawImage(capa, 0, 0)
  destino.restore()
}

// ------------------------------------------------------------
//  Capas de maquillaje
// ------------------------------------------------------------

function pintarLabios (ctx, puntos, w, h, nivel) {
  const capa = lienzo(w, h)
  const c = capa.getContext('2d')

  const xs = LABIOS_FUERA.map(i => puntos[i].x * w)
  const ys = LABIOS_FUERA.map(i => puntos[i].y * h)
  const g = c.createLinearGradient(Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys))
  g.addColorStop(0,   '#ff1f8f')
  g.addColorStop(0.5, '#e0155f')
  g.addColorStop(1,   '#ff4fb0')

  c.fillStyle = g
  c.beginPath()
  trazar(c, puntos, LABIOS_FUERA, w, h)
  trazar(c, puntos, LABIOS_DENTRO, w, h)
  c.fill('evenodd')

  const radio = Math.max(...xs) - Math.min(...xs)
  fundir(ctx, capa, 'multiply', nivel.labios, radio * 0.012)

  // Brillo humedo en el centro del labio inferior
  const brillo = lienzo(w, h)
  const b = brillo.getContext('2d')
  const cx = puntos[14].x * w, cy = puntos[17].y * h
  const rg = b.createRadialGradient(cx, cy, 0, cx, cy, radio * 0.22)
  rg.addColorStop(0, 'rgba(255,255,255,.85)')
  rg.addColorStop(1, 'rgba(255,255,255,0)')
  b.fillStyle = rg
  b.fillRect(cx - radio * .25, cy - radio * .25, radio * .5, radio * .5)
  fundir(ctx, brillo, 'screen', nivel.brillo * .5, radio * .02)
}

function pintarSombra (ctx, puntos, w, h, nivel) {
  const capa = lienzo(w, h)
  const c = capa.getContext('2d')

  const pares = [[OJO_IZQ, CEJA_IZQ], [OJO_DER, CEJA_DER]]
  for (const [ojo, ceja] of pares) {
    const xs = ojo.map(i => puntos[i].x * w)
    const ys = ojo.map(i => puntos[i].y * h)
    const g = c.createLinearGradient(Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys))
    g.addColorStop(0,   '#22d3ee')
    g.addColorStop(0.5, '#a855f7')
    g.addColorStop(1,   '#ff2ea6')
    c.fillStyle = g
    c.beginPath()
    trazar(c, puntos, [...ojo, ...ceja], w, h)
    c.fill()
  }

  const anchoCara = Math.abs(puntos[454].x - puntos[234].x) * w
  fundir(ctx, capa, 'source-over', nivel.sombra, anchoCara * 0.018)
}

function pintarRubor (ctx, puntos, w, h, nivel) {
  const capa = lienzo(w, h)
  const c = capa.getContext('2d')
  const anchoCara = Math.abs(puntos[454].x - puntos[234].x) * w
  const r = anchoCara * 0.17

  for (const idx of [MEJILLA_IZQ, MEJILLA_DER]) {
    const x = puntos[idx].x * w, y = puntos[idx].y * h
    const g = c.createRadialGradient(x, y, 0, x, y, r)
    g.addColorStop(0,   'rgba(255,60,150,.9)')
    g.addColorStop(0.6, 'rgba(255,90,170,.35)')
    g.addColorStop(1,   'rgba(255,120,190,0)')
    c.fillStyle = g
    c.fillRect(x - r, y - r, r * 2, r * 2)
  }

  fundir(ctx, capa, 'soft-light', nivel.rubor * 1.6, anchoCara * 0.02)
  fundir(ctx, capa, 'source-over', nivel.rubor * .45, anchoCara * 0.03)
}

function pintarPurpurina (ctx, puntos, w, h, nivel) {
  const capa = lienzo(w, h)
  const c = capa.getContext('2d')

  // Region valida: el ovalo de la cara
  const mascara = c
  mascara.beginPath()
  trazar(mascara, puntos, OVALO, w, h)

  const xs = OVALO.map(i => puntos[i].x * w)
  const ys = OVALO.map(i => puntos[i].y * h)
  const x0 = Math.min(...xs), x1 = Math.max(...xs)
  const y0 = Math.min(...ys), y1 = Math.max(...ys)

  const colores = ['#ffffff', '#ffd93d', '#22d3ee', '#ff2ea6', '#7cf03d']
  let puestos = 0, intentos = 0

  while (puestos < nivel.purpurina && intentos < nivel.purpurina * 12) {
    intentos++
    const x = x0 + Math.random() * (x1 - x0)
    const y = y0 + Math.random() * (y1 - y0)
    if (!c.isPointInPath(x, y)) continue

    const r = (1 + Math.random() * 2.4) * (w / 700)
    c.fillStyle = colores[(Math.random() * colores.length) | 0]
    c.globalAlpha = 0.35 + Math.random() * 0.65
    c.beginPath()
    c.arc(x, y, r, 0, Math.PI * 2)
    c.fill()
    puestos++
  }
  c.globalAlpha = 1

  fundir(ctx, capa, 'screen', nivel.brillo, 0.4)
}

function pintarAmbiente (ctx, w, h, nivel) {
  // Luz arcoiris sobre toda la foto
  const capa = lienzo(w, h)
  const c = capa.getContext('2d')
  const g = c.createLinearGradient(0, 0, w, h)
  g.addColorStop(0,    '#ff2ea6')
  g.addColorStop(0.2,  '#ff7a1a')
  g.addColorStop(0.4,  '#ffd93d')
  g.addColorStop(0.6,  '#7cf03d')
  g.addColorStop(0.8,  '#22d3ee')
  g.addColorStop(1,    '#a855f7')
  c.fillStyle = g
  c.fillRect(0, 0, w, h)
  fundir(ctx, capa, 'overlay', nivel.arcoiris, 0)

  // Confeti en las esquinas
  const conf = lienzo(w, h)
  const k = conf.getContext('2d')
  const colores = ['#ff2ea6', '#ffd93d', '#22d3ee', '#7cf03d', '#a855f7', '#ff7a1a']
  const cuantos = Math.round(nivel.purpurina * 0.35)

  for (let i = 0; i < cuantos; i++) {
    const x = Math.random() * w
    const y = Math.random() * h
    const borde = Math.min(x, y, w - x, h - y) / Math.min(w, h)
    if (borde > 0.26 && Math.random() > 0.18) continue   // sobre todo en los bordes

    const lado = (4 + Math.random() * 9) * (w / 700)
    k.save()
    k.translate(x, y)
    k.rotate(Math.random() * Math.PI)
    k.fillStyle = colores[(Math.random() * colores.length) | 0]
    k.globalAlpha = 0.5 + Math.random() * 0.5
    k.fillRect(-lado / 2, -lado / 4, lado, lado / 2)
    k.restore()
  }
  fundir(ctx, conf, 'source-over', 0.9, 0.6)

  // Resplandor magenta/cyan en las esquinas
  const glow = lienzo(w, h)
  const q = glow.getContext('2d')
  const r1 = q.createRadialGradient(w * .88, h * .12, 0, w * .88, h * .12, w * .5)
  r1.addColorStop(0, 'rgba(255,46,166,.55)'); r1.addColorStop(1, 'rgba(255,46,166,0)')
  q.fillStyle = r1; q.fillRect(0, 0, w, h)
  const r2 = q.createRadialGradient(w * .1, h * .85, 0, w * .1, h * .85, w * .5)
  r2.addColorStop(0, 'rgba(34,211,238,.5)'); r2.addColorStop(1, 'rgba(34,211,238,0)')
  q.fillStyle = r2; q.fillRect(0, 0, w, h)
  fundir(ctx, glow, 'screen', nivel.arcoiris * 1.9, 0)
}

// ------------------------------------------------------------
//  Extras del nivel DRAG
// ------------------------------------------------------------

// Escala un poligono hacia fuera desde su centro (labios sobredibujados)
function inflar (puntos, indices, w, h, factor) {
  const pts = indices.map(i => ({ x: puntos[i].x * w, y: puntos[i].y * h }))
  const cx = pts.reduce((a, p) => a + p.x, 0) / pts.length
  const cy = pts.reduce((a, p) => a + p.y, 0) / pts.length
  return pts.map(p => ({ x: cx + (p.x - cx) * factor, y: cy + (p.y - cy) * factor }))
}

function trazarPts (ctx, pts) {
  pts.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y))
  ctx.closePath()
}

// Cut-crease: sombra amplia que sube hasta la ceja y se abre hacia la sien
function pintarCutCrease (ctx, puntos, w, h, nivel) {
  const capa = lienzo(w, h)
  const c = capa.getContext('2d')
  const anchoCara = Math.abs(puntos[454].x - puntos[234].x) * w

  const ojos = [
    { lid: OJO_IZQ, brow: CEJA_IZQ, fuera: 33,  dentro: 133 },
    { lid: OJO_DER, brow: CEJA_DER, fuera: 263, dentro: 362 }
  ]

  for (const o of ojos) {
    const pf = puntos[o.fuera], pd = puntos[o.dentro]
    const dx = (pf.x - pd.x) * w, dy = (pf.y - pd.y) * h
    const len = Math.hypot(dx, dy) || 1
    const ux = dx / len, uy = dy / len

    const pts = [...o.lid, ...o.brow].map(i => ({ x: puntos[i].x * w, y: puntos[i].y * h }))
    // Estirar hacia la sien y hacia arriba
    const sienX = pf.x * w + ux * len * 0.30
    const sienY = pf.y * h + uy * len * 0.15 - len * 0.22
    pts.splice(1, 0, { x: sienX, y: sienY })

    const g = c.createLinearGradient(pd.x * w, pd.y * h, sienX, sienY)
    g.addColorStop(0,    '#1a3fd4')
    g.addColorStop(0.35, '#22d3ee')
    g.addColorStop(0.7,  '#a855f7')
    g.addColorStop(1,    '#ff2ea6')
    c.fillStyle = g
    c.beginPath()
    trazarPts(c, pts)
    c.fill()
  }

  fundir(ctx, capa, 'source-over', nivel.sombra * .78, anchoCara * 0.013)
  fundir(ctx, capa, 'screen', nivel.sombra * .22, anchoCara * 0.028)
}

// Cejas drag: tapa la ceja natural y dibuja una alta y arqueada
function pintarCejas (ctx, puntos, w, h, nivel) {
  const anchoCara = Math.abs(puntos[454].x - puntos[234].x) * w

  // 1. Difuminar la ceja propia con tono piel de la frente
  const tapa = lienzo(w, h)
  const t = tapa.getContext('2d')
  for (const ceja of [CEJA_IZQ_FULL, CEJA_DER_FULL]) {
    const pts = ceja.map(i => ({ x: puntos[i].x * w, y: puntos[i].y * h }))
    const cx = pts.reduce((a, p) => a + p.x, 0) / pts.length
    const cy = pts.reduce((a, p) => a + p.y, 0) / pts.length
    t.fillStyle = 'rgba(210,170,140,1)'
    t.beginPath()
    trazarPts(t, pts.map(p => ({ x: cx + (p.x - cx) * 1.25, y: cy + (p.y - cy) * 1.35 })))
    t.fill()
  }
  fundir(ctx, tapa, 'soft-light', .55, anchoCara * 0.03)
  fundir(ctx, tapa, 'source-over', .18, anchoCara * 0.035)

  // 2. Ceja nueva: arco alto y afilado
  const capa = lienzo(w, h)
  const c = capa.getContext('2d')

  const cejas = [
    { pts: [46, 53, 52, 65, 55], fuera: 46, dentro: 55 },
    { pts: [276, 283, 282, 295, 285], fuera: 276, dentro: 285 }
  ]

  for (const cj of cejas) {
    const base = cj.pts.map(i => ({ x: puntos[i].x * w, y: puntos[i].y * h }))
    const pf = base[0], pd = base[base.length - 1]
    const largo = Math.hypot(pf.x - pd.x, pf.y - pd.y) || 1
    const alto = largo * 0.30            // cuanto sube respecto a la natural

    // Punto mas alto del arco, a dos tercios hacia fuera
    const arcoX = pd.x + (pf.x - pd.x) * 0.62
    const arcoY = pd.y + (pf.y - pd.y) * 0.62 - alto * 1.35

    const colaX = pf.x + (pf.x - pd.x) * 0.10
    const colaY = pf.y + (pf.y - pd.y) * 0.10 - alto * 0.35

    const grueso = largo * 0.115

    c.fillStyle = '#1a0f18'
    c.beginPath()
    // Borde superior: de la cabeza al arco y a la cola
    c.moveTo(pd.x, pd.y - alto * 0.45)
    c.quadraticCurveTo(arcoX - (arcoX - pd.x) * 0.3, arcoY - grueso * 0.2, arcoX, arcoY)
    c.quadraticCurveTo(arcoX + (colaX - arcoX) * 0.5, arcoY + (colaY - arcoY) * 0.35, colaX, colaY)
    // Borde inferior de vuelta
    c.quadraticCurveTo(arcoX + (colaX - arcoX) * 0.4, arcoY + grueso, arcoX, arcoY + grueso)
    c.quadraticCurveTo(arcoX - (arcoX - pd.x) * 0.35, arcoY + grueso * 1.25,
                       pd.x, pd.y - alto * 0.45 + grueso * 1.15)
    c.closePath()
    c.fill()
  }

  fundir(ctx, capa, 'source-over', .93, anchoCara * 0.0035)
}

// Delineado grueso con ala
function pintarDelineado (ctx, puntos, w, h, nivel) {
  const capa = lienzo(w, h)
  const c = capa.getContext('2d')
  const anchoCara = Math.abs(puntos[454].x - puntos[234].x) * w
  const grosor = anchoCara * 0.020

  const ojos = [
    { lid: OJO_IZQ, fuera: 33,  dentro: 133 },
    { lid: OJO_DER, fuera: 263, dentro: 362 }
  ]

  c.strokeStyle = '#07030f'
  c.lineCap = 'round'
  c.lineJoin = 'round'

  for (const o of ojos) {
    const pf = puntos[o.fuera], pd = puntos[o.dentro]
    const dx = (pf.x - pd.x) * w, dy = (pf.y - pd.y) * h
    const len = Math.hypot(dx, dy) || 1
    const ux = dx / len, uy = dy / len

    // Linea del parpado, mas gruesa hacia fuera
    const pts = o.lid.map(i => ({ x: puntos[i].x * w, y: puntos[i].y * h }))
    for (let i = 0; i < pts.length - 1; i++) {
      const t = i / (pts.length - 1)
      c.lineWidth = grosor * (0.45 + t * 1.1)
      c.beginPath()
      c.moveTo(pts[i].x, pts[i].y)
      c.lineTo(pts[i + 1].x, pts[i + 1].y)
      c.stroke()
    }

    // El ala
    const alaX = pf.x * w + ux * len * 0.34
    const alaY = pf.y * h + uy * len * 0.14 - len * 0.30
    c.beginPath()
    c.moveTo(pf.x * w, pf.y * h + grosor * 0.5)
    c.lineTo(alaX, alaY)
    c.lineTo(pf.x * w - ux * len * 0.10, pf.y * h - grosor * 1.5)
    c.closePath()
    c.fillStyle = '#07030f'
    c.fill()
  }

  fundir(ctx, capa, 'source-over', nivel.delineado * .92, anchoCara * 0.003)
}

// Pestanas postizas
function pintarPestanas (ctx, puntos, w, h, nivel) {
  const capa = lienzo(w, h)
  const c = capa.getContext('2d')
  const anchoCara = Math.abs(puntos[454].x - puntos[234].x) * w

  c.strokeStyle = '#07030f'
  c.lineCap = 'round'

  const ojos = [
    { lid: [246,161,160,159,158,157,173], fuera: 33,  dentro: 133 },
    { lid: [466,388,387,386,385,384,398], fuera: 263, dentro: 362 }
  ]

  for (const o of ojos) {
    const pf = puntos[o.fuera], pd = puntos[o.dentro]
    const dx = (pf.x - pd.x) * w, dy = (pf.y - pd.y) * h
    const len = Math.hypot(dx, dy) || 1
    const ux = dx / len, uy = dy / len

    o.lid.forEach((idx, i) => {
      const t = i / (o.lid.length - 1)
      const cerca = (o.lid[0] === 246) ? t : 1 - t     // largo creciente hacia fuera
      const largo = len * (0.10 + cerca * 0.16)
      const x = puntos[idx].x * w, y = puntos[idx].y * h

      c.lineWidth = anchoCara * 0.008
      c.beginPath()
      c.moveTo(x, y)
      c.quadraticCurveTo(
        x + ux * largo * 0.4, y - largo * 0.7,
        x + ux * largo * 0.9, y - largo * 0.95
      )
      c.stroke()
    })
  }

  fundir(ctx, capa, 'source-over', nivel.pestanas * .9, anchoCara * 0.002)
}

// Labios rojos sobredibujados
function pintarLabiosDrag (ctx, puntos, w, h, nivel) {
  const capa = lienzo(w, h)
  const c = capa.getContext('2d')

  const fuera = inflar(puntos, LABIOS_FUERA, w, h, nivel.sobredibujo)
  // Anclar a las comisuras reales (61 y 291) para que no baje
  const comX = (puntos[61].x + puntos[291].x) / 2 * w
  const comY = (puntos[61].y + puntos[291].y) / 2 * h
  const oriX = (LABIOS_FUERA.map(i => puntos[i].x * w).reduce((a,b)=>a+b,0)) / LABIOS_FUERA.length
  const oriY = (LABIOS_FUERA.map(i => puntos[i].y * h).reduce((a,b)=>a+b,0)) / LABIOS_FUERA.length
  const cgx = fuera.reduce((a, p) => a + p.x, 0) / fuera.length
  const cgy = fuera.reduce((a, p) => a + p.y, 0) / fuera.length
  fuera.forEach(p => { p.x += oriX - cgx; p.y += oriY - cgy })
  void comX; void comY
  const xs = fuera.map(p => p.x), ys = fuera.map(p => p.y)
  const ancho = Math.max(...xs) - Math.min(...xs)

  const g = c.createLinearGradient(Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys))
  g.addColorStop(0,   '#d40f2a')
  g.addColorStop(0.5, '#f01536')
  g.addColorStop(1,   '#b80d22')
  c.fillStyle = g
  c.beginPath()
  trazarPts(c, fuera)
  c.fill()

  fundir(ctx, capa, 'source-over', nivel.labios * .82, ancho * 0.01)
  fundir(ctx, capa, 'multiply', nivel.labios * .5, ancho * 0.015)

  // Brillo central
  const brillo = lienzo(w, h)
  const b = brillo.getContext('2d')
  const cx = puntos[14].x * w, cy = puntos[17].y * h
  const rg = b.createRadialGradient(cx, cy, 0, cx, cy, ancho * 0.2)
  rg.addColorStop(0, 'rgba(255,255,255,.9)')
  rg.addColorStop(1, 'rgba(255,255,255,0)')
  b.fillStyle = rg
  b.fillRect(cx - ancho * .25, cy - ancho * .25, ancho * .5, ancho * .5)
  fundir(ctx, brillo, 'screen', nivel.brillo * .55, ancho * 0.018)
}

// Contorno e iluminador
function pintarContorno (ctx, puntos, w, h, nivel) {
  const anchoCara = Math.abs(puntos[454].x - puntos[234].x) * w

  // Sombra bajo los pomulos
  const sombra = lienzo(w, h)
  const s = sombra.getContext('2d')
  for (const [a, b] of [[234, 132], [454, 361]]) {
    const x = (puntos[a].x * 0.55 + puntos[b].x * 0.45) * w
    const y = (puntos[a].y * 0.5 + puntos[b].y * 0.5) * h
    const r = anchoCara * 0.16
    const g = s.createRadialGradient(x, y, 0, x, y, r)
    g.addColorStop(0, 'rgba(70,30,45,.75)')
    g.addColorStop(1, 'rgba(70,30,45,0)')
    s.fillStyle = g
    s.fillRect(x - r, y - r, r * 2, r * 2)
  }
  fundir(ctx, sombra, 'multiply', nivel.contorno, anchoCara * 0.035)

  // Iluminador: puente de la nariz y pomulos altos
  const luz = lienzo(w, h)
  const l = luz.getContext('2d')
  for (const [idx, esc] of [[6, .09], [117, .10], [346, .10], [10, .11]]) {
    const p = puntos[idx]
    if (!p) continue
    const x = p.x * w, y = p.y * h, r = anchoCara * esc
    const g = l.createRadialGradient(x, y, 0, x, y, r)
    g.addColorStop(0, 'rgba(255,245,220,.85)')
    g.addColorStop(1, 'rgba(255,245,220,0)')
    l.fillStyle = g
    l.fillRect(x - r, y - r, r * 2, r * 2)
  }
  fundir(ctx, luz, 'screen', nivel.contorno * .95, anchoCara * 0.03)
}

// ------------------------------------------------------------
//  Funcion principal
// ------------------------------------------------------------
async function maquillar (dataUrl, intensidad = 'fabuloso') {
  const nivel = NIVELES[intensidad] ?? NIVELES.fabuloso
  const lm = await preparar()

  const img = await new Promise((ok, mal) => {
    const i = new Image()
    i.onload = () => ok(i)
    i.onerror = () => mal(new Error('No se pudo leer la imagen'))
    i.src = dataUrl
  })

  const escala = Math.min(1, 1200 / Math.max(img.width, img.height))
  const w = Math.round(img.width * escala)
  const h = Math.round(img.height * escala)

  const salida = lienzo(w, h)
  const ctx = salida.getContext('2d')
  ctx.drawImage(img, 0, 0, w, h)

  const res = lm.detect(salida)
  const caras = res?.faceLandmarks ?? []

  if (!caras.length) {
    throw new Error('No detectamos ninguna cara en la foto. Prueba con una imagen mas frontal y con buena luz.')
  }

  // Un poco mas de color y contraste de base
  const base = lienzo(w, h)
  base.getContext('2d').drawImage(salida, 0, 0)
  ctx.save()
  ctx.filter = 'saturate(1.28) contrast(1.08) brightness(1.05)'
  ctx.drawImage(base, 0, 0)
  ctx.restore()

  for (const puntos of caras) {
    if (nivel.cutCrease) {
      pintarContorno(ctx, puntos, w, h, nivel)
      pintarRubor(ctx, puntos, w, h, nivel)
      pintarCutCrease(ctx, puntos, w, h, nivel)
      pintarLabiosDrag(ctx, puntos, w, h, nivel)
      pintarCejas(ctx, puntos, w, h, nivel)
      pintarDelineado(ctx, puntos, w, h, nivel)   // encima de la sombra
      pintarPestanas(ctx, puntos, w, h, nivel)    // lo ultimo, siempre visible
      pintarPurpurina(ctx, puntos, w, h, nivel)
    } else {
      pintarSombra(ctx, puntos, w, h, nivel)
      pintarRubor(ctx, puntos, w, h, nivel)
      pintarLabios(ctx, puntos, w, h, nivel)
      pintarPurpurina(ctx, puntos, w, h, nivel)
    }
  }

  pintarAmbiente(ctx, w, h, nivel)

  return { imagen: salida.toDataURL('image/jpeg', 0.92), caras: caras.length }
}

window.glamMaquillar = maquillar
window.glamListo = true
window.dispatchEvent(new Event('glam-listo'))
