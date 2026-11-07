"""Analisis cromatico por fruto (v3): segmentacion GrabCut + umbral adaptativo Otsu
sobre el canal a* de CIE Lab restringido a la superficie del fruto.
Devuelve el % de superficie con color rosa/rojo (criterio USDA/ASEAN)."""
import cv2, numpy as np

# GrabCut reparte trabajo entre hilos y el orden de reduccion cambia el resultado
# en unas pocas cajas. Con un hilo por proceso la medicion es reproducible bit a bit
# (el paralelismo real lo da el Pool de procesos de extract.py).
cv2.setNumThreads(1)

CFG = dict(
    PAD=4, MAXSIDE=200, GC_ITERS=4,
    SPEC_V=244, SPEC_S=34,                        # brillo especular
    LEAF_HMIN=33, LEAF_HMAX=100, LEAF_SMIN=30,    # verde (hoja, caliz)
    FRG_HMIN=22, FRG_HMAX=37, FRG_VMIN=170,       # verde palido del fruto inmaduro (se conserva)
    DARK_V=80,                                    # sombra profunda -> no clasificable
    T_SEED=150,                                   # a* semilla: rojo inequivoco
    T_GROW=134,                                   # a* de crecimiento por conectividad
    T_SOLO=139,                                   # a* para manchas sin semilla
    MIN_BLOB=0.015,                               # area minima de mancha (frac. superficie)
    SPILL_MAXAREA=0.22,                           # solo se audita el vertido en manchas pequenas
    SPILL_RING=2, SPILL_RATIO=1.0,                # color externo conectado > mancha -> vecino
    A_LO=132, A_HI=152,                           # rango del umbral Otsu (diagnostico)
    A_FIXED=138,
    A_NONE=133,                                   # si p99 de a* no llega aqui -> 0 % color
    A_ALL=152,
    A_DEEP=168,                                   # rojo intenso
    A_PALE=148,                                   # por debajo: rosa/naranja palido
    DARKRED_A=162, DARKRED_L=118,                 # rojo oscuro (granate): fruto pasado
    BROWN_AMIN=120, BROWN_AMAX=155,               # mancha oscura no roja: dano/pudricion
    BROWN_BMIN=135, BROWN_LMAX=145, BROWN_DL=30,  # y claramente mas oscura que el resto
    BROWN_MINBLOB=0.03,
    ASMOOTH=7,
    SEED=20260926,
)


def _largest_cc(mask, cx, cy):
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask
    best, bs = 0, -1.0
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        d = np.hypot(cent[i][0] - cx, cent[i][1] - cy)
        s = a / (1.0 + d)
        if s > bs:
            bs, best = s, i
    return (lab == best).astype(np.uint8)


def _fill_holes(mask):
    h, w = mask.shape
    ff = mask.copy()
    mm = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, mm, (0, 0), 1)
    return ((mask | (ff == 0)) > 0).astype(np.uint8)


def ellipse_mask(h, w, kx=0.50, ky=0.50):
    e = np.zeros((h, w), np.uint8)
    cv2.ellipse(e, (w // 2, h // 2), (max(1, int(w * kx)), max(1, int(h * ky))), 0, 0, 360, 1, -1)
    return e


def segment_fruit(crop, cfg=CFG):
    h, w = crop.shape[:2]
    ell, core = ellipse_mask(h, w, 0.50, 0.50), ellipse_mask(h, w, 0.26, 0.26)
    if min(h, w) < 14:
        return ell, "ellipse-small"
    gm = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    gm[ell > 0] = cv2.GC_PR_FGD
    gm[core > 0] = cv2.GC_FGD
    bw = max(1, int(round(min(h, w) * 0.045)))
    gm[:bw, :] = cv2.GC_BGD
    gm[-bw:, :] = cv2.GC_BGD
    gm[:, :bw] = cv2.GC_BGD
    gm[:, -bw:] = cv2.GC_BGD
    try:
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        # GrabCut inicializa sus GMM con k-means++ y RNG global: sin semilla fija la
        # segmentacion varia entre ejecuciones en un 5-20 % de las cajas
        cv2.setRNGSeed(cfg["SEED"])
        cv2.grabCut(crop, gm, None, bgd, fgd, cfg["GC_ITERS"], cv2.GC_INIT_WITH_MASK)
        m = np.where((gm == cv2.GC_FGD) | (gm == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    except Exception:
        return ell, "ellipse-gcfail"
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = _fill_holes(_largest_cc(m, w / 2.0, h / 2.0))
    if m.sum() / float(h * w) < 0.15:
        return ell, "ellipse-fallback"
    ell_d = cv2.dilate(ell, np.ones((3, 3), np.uint8), iterations=max(2, int(min(h, w) * 0.06)))
    m2 = ((m & ell_d) > 0).astype(np.uint8)
    if m2.sum() < 0.45 * m.sum():
        m2 = m
    return _fill_holes(_largest_cc(m2, w / 2.0, h / 2.0)), "grabcut"


def otsu_a(vals, cfg=CFG):
    """Umbral Otsu sobre a* limitado a [A_LO, A_HI]. Devuelve (umbral, separabilidad)."""
    hist = np.bincount(np.clip(vals, 100, 220) - 100, minlength=121).astype(np.float64)
    tot = hist.sum()
    if tot < 30:
        return cfg["A_FIXED"], 0.0
    p = hist / tot
    idx = np.arange(121)
    w0 = np.cumsum(p)
    mu = np.cumsum(p * idx)
    muT = mu[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        sb = (muT * w0 - mu) ** 2 / (w0 * (1 - w0))
    sb = np.nan_to_num(sb)
    lo, hi = cfg["A_LO"] - 100, cfg["A_HI"] - 100
    k = int(np.argmax(sb[lo:hi + 1])) + lo
    var = float(((idx - muT) ** 2 * p).sum())
    sep = float(sb[k] / var) if var > 1e-6 else 0.0
    return k + 100, sep


def hysteresis(As, surf, cfg=CFG, fruit=None):
    """Mascara de superficie coloreada.

    Regiones a*>=T_GROW conectadas a una semilla a*>=T_SEED, mas manchas
    a*>=T_SOLO suficientemente grandes (rubor incipiente). Las manchas pequenas
    cuyo color continua FUERA de la mascara del fruto se descartan: proceden de
    otro fruto rojo vecino dentro del bbox.
    """
    n = max(1, int(surf.sum()))
    grow = ((As >= cfg["T_GROW"]) & (surf > 0)).astype(np.uint8)
    grow = cv2.morphologyEx(grow, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    grow = cv2.morphologyEx(grow, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    grow = (grow & (surf > 0)).astype(np.uint8)
    seed = ((As >= cfg["T_SEED"]) & (grow > 0)).astype(np.uint8)
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    solo = ((As >= cfg["T_SOLO"]) & (grow > 0)).astype(np.uint8)
    solo = cv2.morphologyEx(solo, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    outside = None
    if fruit is not None:
        outside = ((As >= cfg["T_GROW"]) & (fruit == 0)).astype(np.uint8)
    ncc, lbl, stats, _ = cv2.connectedComponentsWithStats(grow, 8)
    out = np.zeros_like(grow)
    rej = np.zeros_like(grow)
    minblob = cfg["MIN_BLOB"] * n
    ncomp = 0
    for i in range(1, ncc):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < max(12, 0.004 * n):
            continue
        comp = (lbl == i).astype(np.uint8)
        ok = bool(seed[comp > 0].any()) or int(solo[comp > 0].sum()) >= minblob
        if not ok:
            continue
        if outside is not None and area < cfg["SPILL_MAXAREA"] * n:
            # cuanto color hay FUERA del fruto conectado con esta mancha: si la
            # zona roja externa es mayor que la interna, el color es de un fruto vecino
            seedzone = cv2.dilate(comp, np.ones((3, 3), np.uint8), iterations=cfg["SPILL_RING"])
            conn = ((outside > 0) & (seedzone > 0)).astype(np.uint8)
            if conn.any():
                nc, lb = cv2.connectedComponents(((outside > 0) | (comp > 0)).astype(np.uint8), 8)
                ids = np.unique(lb[comp > 0])
                ext = 0
                for t in ids:
                    if t == 0:
                        continue
                    ext += int(((lb == t) & (outside > 0)).sum())
                if ext > cfg["SPILL_RATIO"] * area:
                    rej |= comp                 # el color continua fuera: fruto vecino
                    continue
        out |= comp
        ncomp += 1
    return out.astype(np.uint8), rej.astype(np.uint8), ncomp


def clean(mask, surf):
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return (m & (surf > 0)).astype(np.uint8)


def brown_mask(L, A, B, S, body, cfg=CFG):
    """Manchas oscuras NO rojas dentro del fruto: dano, pudricion, magulladura o
    fruto momificado. Es un detector de candidatos (prioriza no perderse casos);
    los falsos positivos por sombra se descartan luego en la revision visual."""
    nb = int(body.sum())
    if nb < 60:
        return np.zeros_like(body), 0.0, 0.0
    Lmed = float(np.median(L[body > 0]))
    m = ((body > 0) & (L <= Lmed - cfg["BROWN_DL"]) & (L <= cfg["BROWN_LMAX"]) &
         (A >= cfg["BROWN_AMIN"]) & (A <= cfg["BROWN_AMAX"]) & (B >= cfg["BROWN_BMIN"])).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    ncc, lbl, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    big = 0
    for i in range(1, ncc):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a >= cfg["BROWN_MINBLOB"] * nb:
            keep[lbl == i] = 1
            big = max(big, a)
    return keep, 100.0 * float(keep.sum()) / nb, 100.0 * big / nb


def analyze(crop, cfg=CFG):
    h0, w0 = crop.shape[:2]
    if max(h0, w0) > cfg["MAXSIDE"]:
        s = cfg["MAXSIDE"] / max(h0, w0)
        crop = cv2.resize(crop, (max(8, int(w0 * s)), max(8, int(h0 * s))), interpolation=cv2.INTER_AREA)
    sm = cv2.medianBlur(crop, 3)
    fruit, how = segment_fruit(sm, cfg)
    hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(sm, cv2.COLOR_BGR2LAB)
    H, S, V = [hsv[..., i].astype(np.int16) for i in range(3)]
    L, A, B = [lab[..., i].astype(np.int16) for i in range(3)]
    As = cv2.medianBlur(lab[..., 1], cfg["ASMOOTH"]).astype(np.int16)
    spec = ((V >= cfg["SPEC_V"]) & (S <= cfg["SPEC_S"])).astype(np.uint8)
    # ocluyentes verdes: hoja, caliz y pedunculo. El fruto (verde palido o crema)
    # vive en H<38; por encima de ese tono solo hay vegetacion.
    leaf_hi = (H >= 38) & (H <= cfg["LEAF_HMAX"]) & (S >= 20)
    leaf_lo = (H >= cfg["LEAF_HMIN"]) & (H < 38) & (S >= cfg["LEAF_SMIN"]) & (V < cfg["FRG_VMIN"])
    leaf = (leaf_hi | leaf_lo).astype(np.uint8)
    leaf = cv2.morphologyEx(leaf, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    leaf = cv2.morphologyEx(leaf, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    dark = (V < cfg["DARK_V"]).astype(np.uint8)
    surf = (fruit & (1 - spec) & (1 - leaf) & (1 - dark)).astype(np.uint8)
    n = int(surf.sum())
    nf = max(1, int(fruit.sum()))
    bl = float(cv2.Laplacian(cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    bd = np.zeros_like(fruit); bd[0, :] = bd[-1, :] = 1; bd[:, 0] = bd[:, -1] = 1
    res = dict(how=how, nsurf=n, nfruit=int(fruit.sum()), area_px=h0 * w0, w_px=w0, h_px=h0,
               blur=round(bl, 1), border_frac=float((fruit & bd).sum()) / max(1, int(bd.sum())),
               leaf_frac=float((leaf & fruit).sum()) / nf,
               spec_frac=float((spec & fruit).sum()) / nf,
               dark_frac=float((dark & fruit).sum()) / nf,
               surf_frac=float(n) / nf)
    if n < 40:
        # superficie util minuscula (fruto casi todo hoja/sombra): se mide sobre
        # la mascara completa del fruto solo para decidir si hay pigmento o no
        fb = fruit.astype(np.uint8)
        if int(fb.sum()) >= 60:
            mfb, _, _ = hysteresis(As, fb, cfg, None)
            pf = 100.0 * float(mfb.sum()) / max(1, int(fb.sum()))
        else:
            mfb, pf = np.zeros_like(surf), 0.0
        res.update(pct_brown=0.0, brown_blob=0.0, pct_darkred=0.0, pct_pale_col=0.0,
                   a_col_std=0.0, gloss=0.0)
        res.update(valid=False, pct=pf, pct_fruit=pf, pct_rej=0.0, ncomp=0, pct_fix=pf, pct_deep=0.0,
                   thr=0, sep=0.0, a_p2=0.0, a_p50=0.0, a_p90=0.0, a_p99=0.0, a_col=0.0, L_col=0.0,
                   L_mean=0.0, S_mean=0.0, b_mean=0.0, mode="fallback-fruit")
        return res, dict(crop=sm, fruit=fruit, surface=fb, colored=mfb, leaf=leaf, spec=spec)
    v = As[surf > 0]
    p2, p50, p90, p99 = [float(np.percentile(v, q)) for q in (2, 50, 90, 99)]
    thr, sep = otsu_a(v, cfg)
    if p99 < cfg["A_NONE"]:
        mode = "none"
        m_col = np.zeros_like(surf)
        m_rej = np.zeros_like(surf)
        ncomp = 0
    else:
        mode = "hyst"
        m_col, m_rej, ncomp = hysteresis(As, surf, cfg, fruit)
    m_fix = clean(((As >= cfg["A_FIXED"]) & (surf > 0)).astype(np.uint8), surf)
    m_deep = clean(((As >= cfg["A_DEEP"]) & (surf > 0)).astype(np.uint8), surf)
    m_col_f = hysteresis(As, fruit.astype(np.uint8), cfg)[0] if mode != "none" else np.zeros_like(surf)
    col_vals = A[m_col > 0]
    body = (fruit & (1 - leaf) & (1 - spec)).astype(np.uint8)   # fruto sin hoja ni brillo
    m_brown, pct_brown, brown_blob = brown_mask(L, A, B, S, body, cfg)
    m_darkred = clean((((As >= cfg["DARKRED_A"]) & (L <= cfg["DARKRED_L"])) & (surf > 0)).astype(np.uint8), surf)
    n_col = max(1, int(m_col.sum()))
    pale = int(((A < cfg["A_PALE"]) & (m_col > 0)).sum())
    res.update(
        pct_brown=pct_brown, brown_blob=brown_blob,
        pct_darkred=100.0 * float(m_darkred.sum()) / n,
        pct_pale_col=100.0 * pale / n_col,
        a_col_std=float(col_vals.std()) if col_vals.size else 0.0,
        gloss=100.0 * float(((spec > 0) & (cv2.dilate(m_col, np.ones((5, 5), np.uint8)) > 0)).sum()) / n_col)
    res.update(valid=True, thr=int(thr), sep=float(sep), mode=mode,
               pct=100.0 * float(m_col.sum()) / n,
               pct_fruit=100.0 * float(m_col_f.sum()) / nf,
               pct_rej=100.0 * float(m_rej.sum()) / n, ncomp=int(ncomp),
               pct_fix=100.0 * float(m_fix.sum()) / n,
               pct_deep=100.0 * float(m_deep.sum()) / n,
               a_p2=p2, a_p50=p50, a_p90=p90, a_p99=p99,
               a_col=float(col_vals.mean()) if col_vals.size else 0.0,
               L_col=float(L[m_col > 0].mean()) if m_col.sum() else 0.0,
               L_mean=float(L[surf > 0].mean()), S_mean=float(S[surf > 0].mean()),
               b_mean=float(B[surf > 0].mean()))
    return res, dict(crop=sm, fruit=fruit, surface=surf, colored=m_col, leaf=leaf, spec=spec,
                     rejected=m_rej, brown=m_brown, darkred=m_darkred)
