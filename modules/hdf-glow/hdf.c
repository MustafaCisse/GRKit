/* HDF Glow: the glow of a highlight diffusion filter on each recorded picture.

   A diffusion filter in front of the lens spreads a little of the light of
   bright areas over their surroundings. The module does the same on the
   YCbCr planes of each JPEG, in three steps:

   1. gather: the light that spills from each cell of a grid over the
      picture, and its colour on a grid half as fine; with MIST, the mean
      light of all pixels on a grid a quarter as fine;
   2. spread: blur the grid at three radii, a tight halo and a wide veil;
      with MIST, blur the mean light very wide;
   3. blend: add the glow to each pixel in linear light as a screen, so
      bright areas never clip harder, and tint it with the colour of the
      lights. HALO adds the red ring of CineStill film around the lights.
      MIST adds the veil of a Black Pro-Mist first: a little of all the
      light, so the shadows lift and the contrast drops. STREAK adds the
      horizontal flare of an anamorphic lens, row by row, from the lights
      of that row.

   The grids fit the scratch memory of the call: a 24 MP picture uses cells
   of 32 x 32 pixels. The glow is soft, so the cells do not show. The live
   view and DNG files stay as they are. */
#include "grkit.h"
#include "grkit_module.h"
#include "hdf_tables.h"

void hdf_apply(PictureSet *set, u32 strength);
u32 hdf_grid_shift(u32 width, u32 height);

#define MAX_GRID 256U
/* Glow of each strength, x256: Off, then 1/8, 1/4, and 1/2 of a Black Pro-Mist. */
static const u16 AMOUNT[4] = {0, 70, 141, 282};
/* The Black Mist veil of each strength: the share of the local mean light
   (x256) that the filter scatters over the shadows. MIST 1 is Light, MIST 2
   Heavy, twice as much. */
static const u16 VEIL[4] = {0, 28, 56, 112};
/* The CineStill halation: red-orange light (1, 0.2, 0.06), its chroma and its
   luma in linear light (x256). */
#define HALO_CB (-50)
#define HALO_CR 70
#define HALO_LUMA 92U
/* The anamorphic streak: a short bright core and a long faint tail (x256),
   decays of 1/8 and 1/64 per sample, and the chroma of each colour: Off,
   blue (0.3, 0.55, 1), amber (1, 0.6, 0.2). */
#define STREAK_CORE 110U
#define STREAK_TAIL 420U
/* Lights wider than about 2^STREAK_WIDE_K samples streak less: a white car
   or a lit wall is no point of light. */
#define STREAK_WIDE_K 4
#define STREAK_CORE_K 3
#define STREAK_TAIL_K 6
static const int STREAK_CB[3] = {0, 60, -58};
static const int STREAK_CR[3] = {0, -40, 50};
/* A streak row has about 700 to 1500 samples, whatever the plane. */
#define STREAK_SAMPLES 700U

typedef struct {
    u32 left, top, w, h;     /* the visible area */
    u32 shift, gw, gh;       /* cells of 1 << shift pixels; the coarse grid has cells twice as large */
    u32 cw, ch, qw, qh;      /* the coarse grid, and the veil grid with cells four times as large */
    u32 ss, sn;              /* streak samples of 1 << ss pixels, sn of them in a row */
    u16 *e, *g;              /* spilled light, then glow, per cell */
    u16 *wt, *nb, *nr;       /* coarse: spilled light, and its Cb and Cr (offset 32768) */
    u16 *q;                  /* veil: mean light */
    u16 *line, *row, *rb, *rr, *vrow;
    u16 *sa, *sb;            /* the streak of one row: core and tail */
    u32 *acc, *accw, *accq;
    int *accb, *accr;
} Grid;

typedef struct { u32 i0, i1, f; } Lerp;

static u32 align4(u32 n) { return (n + 3U) & ~3U; }
static u32 min_u(u32 a, u32 b) { return a < b ? a : b; }
static u32 clamp8(int v) { return v < 0 ? 0U : v > 255 ? 255U : (u32)v; }
static int sdiv(int n, u32 d) { return n < 0 ? -(int)divide((u32)-n, d) : (int)divide((u32)n, d); }

static u32 streak_shift(u32 width)
{
    u32 s = 0;
    while (s < 3U && (width >> (s + 1U)) >= STREAK_SAMPLES) ++s;
    return s;
}

static u32 grid_bytes(u32 width, u32 gw, u32 gh)
{
    u32 cw = (gw + 1U) >> 1, ch = (gh + 1U) >> 1, qw = (gw + 3U) >> 2, qh = (gh + 3U) >> 2;
    u32 ss = streak_shift(width), sn = (width + (1U << ss) - 1U) >> ss;
    return align4(gw * gh * 2U) * 2U + align4(cw * ch * 2U) * 3U + align4((gw > gh ? gw : gh) * 2U) +
           align4((gw + 1U) * 2U) + align4((cw + 1U) * 2U) * 2U + gw * 4U + cw * 16U +
           align4(qw * qh * 2U) + align4((qw + 1U) * 2U) + qw * 4U + align4(sn * 2U) * 2U;
}

/* The shift of the finest grid of a width x height area that fits the
   scratch memory, or 32 when none does. */
u32 hdf_grid_shift(u32 width, u32 height)
{
    u32 s;
    for (s = 0; s < 12U; ++s) {
        u32 cell = 1U << s, gw = (width + cell - 1U) >> s, gh = (height + cell - 1U) >> s;
        if (gw <= MAX_GRID && gh <= MAX_GRID && grid_bytes(width, gw, gh) <= GRKIT_PLANE_SCRATCH)
            return s;
    }
    return 32U;
}

static u32 layout(Grid *gr, const Plane *p, u8 *m)
{
    u32 n;
    if (p->right > p->width || p->bottom > p->height || p->right <= p->left || p->bottom <= p->top) return 0;
    gr->left = p->left; gr->top = p->top;
    gr->w = p->right - p->left; gr->h = p->bottom - p->top;
    gr->shift = hdf_grid_shift(gr->w, gr->h);
    if (gr->shift == 32U) return 0;
    gr->gw = (gr->w + (1U << gr->shift) - 1U) >> gr->shift;
    gr->gh = (gr->h + (1U << gr->shift) - 1U) >> gr->shift;
    gr->cw = (gr->gw + 1U) >> 1; gr->ch = (gr->gh + 1U) >> 1;
    gr->qw = (gr->gw + 3U) >> 2; gr->qh = (gr->gh + 3U) >> 2;
    gr->ss = streak_shift(gr->w);
    gr->sn = (gr->w + (1U << gr->ss) - 1U) >> gr->ss;
    n = align4(gr->gw * gr->gh * 2U);
    gr->e = (u16 *)m; m += n;
    gr->g = (u16 *)m; m += n;
    n = align4(gr->cw * gr->ch * 2U);
    gr->wt = (u16 *)m; m += n;
    gr->nb = (u16 *)m; m += n;
    gr->nr = (u16 *)m; m += n;
    gr->q = (u16 *)m; m += align4(gr->qw * gr->qh * 2U);
    gr->line = (u16 *)m; m += align4((gr->gw > gr->gh ? gr->gw : gr->gh) * 2U);
    gr->row = (u16 *)m; m += align4((gr->gw + 1U) * 2U);
    gr->rb = (u16 *)m; m += align4((gr->cw + 1U) * 2U);
    gr->rr = (u16 *)m; m += align4((gr->cw + 1U) * 2U);
    gr->vrow = (u16 *)m; m += align4((gr->qw + 1U) * 2U);
    gr->sa = (u16 *)m; m += align4(gr->sn * 2U);
    gr->sb = (u16 *)m; m += align4(gr->sn * 2U);
    gr->acc = (u32 *)m; m += gr->gw * 4U;
    gr->accq = (u32 *)m; m += gr->qw * 4U;
    gr->accw = (u32 *)m; m += gr->cw * 4U;
    gr->accb = (int *)m; m += gr->cw * 4U;
    gr->accr = (int *)m;
    return 1;
}

/* The samples from `from` to `to` every 1 << sshift pixels. */
static u32 samples(u32 from, u32 to, u32 sshift)
{
    return to > from ? (to - from + (1U << sshift) - 1U) >> sshift : 0U;
}

/* The brightest channel of a pixel. */
static u32 brightest(int yv, int cb, int cr)
{
    u32 r = clamp8(yv + ((359 * cr) >> 8));
    u32 g = clamp8(yv - ((88 * cb + 183 * cr) >> 8));
    u32 b = clamp8(yv + ((454 * cb) >> 8));
    u32 mx = r > g ? r : g;
    return b > mx ? b : mx;
}

/* Step 1: the mean spilled light of each cell, in linear light (4095 is
   white, clipped areas up to 3x), and on the coarse grid its Cb and Cr.
   With `mist`, the mean light of all pixels on the veil grid. */
static void gather(const Plane *p, const Grid *gr, u32 mono, u32 mist)
{
    /* Big cells sample every second pixel: at most 16 x 16 samples a cell. */
    u32 sshift = gr->shift > 4U ? gr->shift - 4U : 0U, step = 1U << sshift;
    u32 cell = 1U << gr->shift, right = gr->left + gr->w, bottom = gr->top + gr->h;
    u32 cy, gx, crows = 0, qrows = 0;
    for (cy = 0; cy < gr->gh; ++cy) {
        u32 y, y0 = gr->top + (cy << gr->shift), y1 = min_u(y0 + cell, bottom);
        u32 rows = samples(y0, y1, sshift);
        for (gx = 0; gx < gr->gw; ++gx) gr->acc[gx] = 0;
        if (!(cy & 1U)) {
            for (gx = 0; gx < gr->cw; ++gx) {
                gr->accw[gx] = 0;
                gr->accb[gx] = 0;
                gr->accr[gx] = 0;
            }
            crows = 0;
        }
        if (!(cy & 3U)) {
            for (gx = 0; gx < gr->qw; ++gx) gr->accq[gx] = 0;
            qrows = 0;
        }
        crows += rows;
        qrows += rows;
        for (y = y0; y < y1; y += step) {
            volatile u8 *luma = GRKIT_BYTES(p->y + y * p->pitch);
            volatile u8 *crow = GRKIT_BYTES(chroma_row(p, y));
            u32 x;
            for (x = gr->left; x < right; x += step) {
                int yv = luma[x], cb = 0, cr = 0;
                u32 spill, e, at;
                if (mist) gr->accq[(x - gr->left) >> (gr->shift + 2U)] += LIN[yv];
                if (!mono) {
                    cb = (int)crow[x & ~1U] - 128;
                    cr = (int)crow[(x & ~1U) + 1U] - 128;
                }
                /* The brightest channel decides, so coloured lamps and neon spill too. */
                spill = SPILL[brightest(yv, cb, cr)];
                if (!spill) continue;
                {
                    u32 r = clamp8(yv + ((359 * cr) >> 8));
                    u32 g = clamp8(yv - ((88 * cb + 183 * cr) >> 8));
                    u32 b = clamp8(yv + ((454 * cb) >> 8));
                    e = (((54U * LIN[r] + 183U * LIN[g] + 19U * LIN[b]) >> 8) * spill) >> 8;
                }
                at = (x - gr->left) >> gr->shift;
                gr->acc[at] += e;
                at >>= 1;
                gr->accw[at] += e;
                gr->accb[at] += (int)e * cb;
                gr->accr[at] += (int)e * cr;
            }
        }
        for (gx = 0; gx < gr->gw; ++gx) {
            u32 x0 = gr->left + (gx << gr->shift);
            gr->e[cy * gr->gw + gx] = (u16)divide(gr->acc[gx], samples(x0, min_u(x0 + cell, right), sshift) * rows);
        }
        if ((cy & 1U) || cy + 1U == gr->gh) {
            u32 c, at = (cy >> 1) * gr->cw;
            for (c = 0; c < gr->cw; ++c) {
                u32 x0 = gr->left + (c << (gr->shift + 1U));
                u32 n = samples(x0, min_u(x0 + 2U * cell, right), sshift) * crows;
                gr->wt[at + c] = (u16)divide(gr->accw[c], n);
                gr->nb[at + c] = (u16)(32768 + (sdiv(gr->accb[c], n) >> 7));
                gr->nr[at + c] = (u16)(32768 + (sdiv(gr->accr[c], n) >> 7));
            }
        }
        if (mist && ((cy & 3U) == 3U || cy + 1U == gr->gh)) {
            u32 c, at = (cy >> 2) * gr->qw;
            for (c = 0; c < gr->qw; ++c) {
                u32 x0 = gr->left + (c << (gr->shift + 2U));
                u32 n = samples(x0, min_u(x0 + 4U * cell, right), sshift) * qrows;
                gr->q[at + c] = (u16)divide(gr->accq[c], n);
            }
        }
    }
}

/* One box blur of n values `stride` apart, with the edge values repeated. */
static void blur_line(u16 *d, u32 n, u32 stride, u32 r, u16 *tmp, u32 inv)
{
    u32 i, sum;
    for (i = 0; i < n; ++i) tmp[i] = d[i * stride];
    sum = tmp[0] * (r + 1U);
    for (i = 1; i <= r; ++i) sum += tmp[min_u(i, n - 1U)];
    for (i = 0; i < n; ++i) {
        d[i * stride] = (u16)(((unsigned long long)sum * inv + 32768U) >> 16);
        sum += tmp[min_u(i + r + 1U, n - 1U)];
        sum -= tmp[i >= r ? i - r : 0U];
    }
}

/* A Gaussian blur of radius about r cells: three box blurs each way. */
static void blur(u16 *a, u32 w, u32 h, u32 r, u16 *tmp)
{
    /* Rounded, so many passes do not drift: the colour grids sit at 32768. */
    u32 pass, i, inv = divide(65536U + r, 2U * r + 1U);
    for (pass = 0; pass < 3U; ++pass) {
        for (i = 0; i < h; ++i) blur_line(a + i * w, w, 1U, r, tmp, inv);
        for (i = 0; i < w; ++i) blur_line(a + i, h, w, r, tmp, inv);
    }
}

/* A radius given for a grid of 188 cells on its long side, the grid of a
   24 MP picture, so each plane of a shot gets the same glow. */
static u32 radius(u32 base, u32 longest)
{
    u32 r = divide(base * longest + 94U, 188U);
    return r ? r : 1U;
}

/* Step 2: the glow, half a tight halo and half a wide veil, and the colour
   of the light that makes it. With `mist`, the very wide veil of all light. */
static void spread(const Grid *gr, u32 mist)
{
    static const u8 BASE[3] = {1, 2, 8}, WEIGHT[3] = {128, 77, 51};
    u32 i, k, r, n = gr->gw * gr->gh, longest = gr->gw > gr->gh ? gr->gw : gr->gh;
    for (i = 0; i < n; ++i) gr->g[i] = 0;
    /* The blurs follow each other, so each radius adds to the last one. */
    for (k = 0; k < 3U; ++k) {
        blur(gr->e, gr->gw, gr->gh, radius(BASE[k], longest), gr->line);
        for (i = 0; i < n; ++i) gr->g[i] = (u16)(gr->g[i] + ((gr->e[i] * WEIGHT[k]) >> 8));
    }
    if (mist) blur(gr->q, gr->qw, gr->qh, radius(24U, longest), gr->line);
    n = gr->cw * gr->ch;
    r = radius(4U, gr->cw > gr->ch ? gr->cw : gr->ch);
    blur(gr->wt, gr->cw, gr->ch, r, gr->line);
    blur(gr->nb, gr->cw, gr->ch, r, gr->line);
    blur(gr->nr, gr->cw, gr->ch, r, gr->line);
    /* The mean chroma of the light, from its weighted sums. */
    for (i = 0; i < n; ++i) {
        int tb = 0, tr = 0;
        /* Too little light gives no reliable colour: the glow stays neutral. */
        if (gr->wt[i] >= 64U) {
            tb = sdiv(((int)gr->nb[i] - 32768) * 128, gr->wt[i]);
            tr = sdiv(((int)gr->nr[i] - 32768) * 128, gr->wt[i]);
            tb = tb < -127 ? -127 : tb > 127 ? 127 : tb;
            tr = tr < -127 ? -127 : tr > 127 ? 127 : tr;
        }
        gr->nb[i] = (u16)(32768 + tb);
        gr->nr[i] = (u16)(32768 + tr);
    }
}

/* Where pixel `at` falls between the centres of cells of 1 << shift pixels. */
static Lerp lerp(u32 at, u32 shift, u32 n)
{
    Lerp l;
    int pos = (int)((((at << 8) + 128U) >> shift)) - 128;
    if (pos <= 0) {
        l.i0 = l.i1 = 0;
        l.f = 0;
        return l;
    }
    l.i0 = (u32)pos >> 8;
    l.f = (u32)pos & 255U;
    if (l.i0 + 1U >= n) {
        l.i0 = l.i1 = n - 1U;
        l.f = 0;
    } else {
        l.i1 = l.i0 + 1U;
    }
    return l;
}

static u32 mix(const u16 *v, Lerp l) { return (v[l.i0] * (256U - l.f) + v[l.i1] * l.f) >> 8; }

/* A symmetric exponential blur of n values in place: one pass each way,
   each keeping 1 - 2^-k of the last value. */
static void decay(u16 *s, u32 n, u32 k)
{
    int f = 0;
    u32 i;
    for (i = 0; i < n; ++i) {
        f += (((int)s[i] << 4) - f) >> k;
        s[i] = (u16)(f >> 4);
    }
    f = 0;
    for (i = n; i-- > 0;) {
        f += (((int)s[i] << 4) - f) >> k;
        s[i] = (u16)(f >> 4);
    }
}

/* The streak of row y, from its lights before the blend changes them: the
   brightest source in each sample, then a short and a long blur sideways.
   Returns 0 when the row has no light. */
static u32 streak_row(const Plane *p, const Grid *gr, u32 y, u32 mono)
{
    volatile u8 *luma = GRKIT_BYTES(p->y + y * p->pitch);
    volatile u8 *crow = GRKIT_BYTES(chroma_row(p, y));
    u32 right = gr->left + gr->w, step = 1U << gr->ss, inner = step > 2U ? 2U : 1U, i, any = 0;
    for (i = 0; i < gr->sn; ++i) {
        u32 x0 = gr->left + (i << gr->ss), x1 = min_u(x0 + step, right), x, best = 0;
        for (x = x0; x < x1; x += inner) {
            int yv = luma[x], cb = 0, cr = 0;
            u32 v;
            if (yv < 150) continue;
            if (!mono) {
                cb = (int)crow[x & ~1U] - 128;
                cr = (int)crow[(x & ~1U) + 1U] - 128;
            }
            v = STREAK_SRC[brightest(yv, cb, cr)];
            if (v > best) best = v;
        }
        gr->sa[i] = gr->sb[i] = (u16)best;
        any |= best;
    }
    if (!any) return 0;
    /* Keep only what stands out of its surroundings in the row. */
    decay(gr->sb, gr->sn, STREAK_WIDE_K);
    any = 0;
    for (i = 0; i < gr->sn; ++i) {
        u32 v = gr->sa[i], m = 2U * gr->sb[i];
        v = v > m ? v - m : 0U;
        gr->sa[i] = gr->sb[i] = (u16)v;
        any |= v;
    }
    if (!any) return 0;
    decay(gr->sa, gr->sn, STREAK_CORE_K);
    decay(gr->sb, gr->sn, STREAK_TAIL_K);
    return 1;
}

/* Step 3: add the veil, the glow, the halo and the streak to each pixel, and
   mix the colour of each into the chroma by its share of the new light. */
static void blend(const Plane *p, const Grid *gr, u32 strength, u32 halo, u32 mist, u32 streak, u32 mono)
{
    u32 right = gr->left + gr->w, bottom = gr->top + gr->h, y, i;
    u32 amount = AMOUNT[strength], veil = mist ? VEIL[strength] * mist : 0U;
    int scb = mono ? 0 : STREAK_CB[streak], scr = mono ? 0 : STREAK_CR[streak];
    for (y = gr->top; y < bottom; ++y) {
        volatile u8 *luma = GRKIT_BYTES(p->y + y * p->pitch);
        volatile u8 *crow = GRKIT_BYTES(chroma_row(p, y));
        u32 chroma = !mono && (!p->half || !(y & 1U)), any = 0, st = 0, x;
        Lerp ly = lerp(y - gr->top, gr->shift, gr->gh);
        const u16 *a = gr->g + ly.i0 * gr->gw, *b = gr->g + ly.i1 * gr->gw;
        for (i = 0; i < gr->gw; ++i) {
            u32 v = (a[i] * (256U - ly.f) + b[i] * ly.f) >> 8;
            gr->row[i] = (u16)v;
            any |= v;
        }
        if (veil) {
            Lerp lq = lerp(y - gr->top, gr->shift + 2U, gr->qh);
            const u16 *q0 = gr->q + lq.i0 * gr->qw, *q1 = gr->q + lq.i1 * gr->qw;
            for (i = 0; i < gr->qw; ++i) gr->vrow[i] = (u16)((q0[i] * (256U - lq.f) + q1[i] * lq.f) >> 8);
        }
        if (streak) st = streak_row(p, gr, y, mono);
        if (!any && !veil && !st) continue;
        if (chroma && any) {
            Lerp lc = lerp(y - gr->top, gr->shift + 1U, gr->ch);
            const u16 *b0 = gr->nb + lc.i0 * gr->cw, *b1 = gr->nb + lc.i1 * gr->cw;
            const u16 *r0 = gr->nr + lc.i0 * gr->cw, *r1 = gr->nr + lc.i1 * gr->cw;
            for (i = 0; i < gr->cw; ++i) {
                gr->rb[i] = (u16)((b0[i] * (256U - lc.f) + b1[i] * lc.f) >> 8);
                gr->rr[i] = (u16)((r0[i] * (256U - lc.f) + r1[i] * lc.f) >> 8);
            }
        }
        for (x = gr->left; x < right; ++x) {
            u32 glow = any ? mix(gr->row, lerp(x - gr->left, gr->shift, gr->gw)) : 0U;
            u32 vl = veil ? (mix(gr->vrow, lerp(x - gr->left, gr->shift + 2U, gr->qw)) * veil) >> 8 : 0U;
            u32 sl = 0, yv, l0, lv, l1, l2, l3;
            if (st) {
                Lerp ls = lerp(x - gr->left, gr->ss, gr->sn);
                sl = (mix(gr->sa, ls) * STREAK_CORE + mix(gr->sb, ls) * STREAK_TAIL) >> 8;
            }
            if (!glow && !vl && !sl) continue;
            yv = luma[x];
            l0 = LIN[yv];
            lv = l0 + ((min_u(vl, 4095U) * (4095U - l0)) >> 12);
            l1 = lv + ((min_u((glow * amount) >> 8, 4095U) * (4095U - lv)) >> 12);
            l2 = l1;
            if (halo && glow) {
                u32 h = min_u((((glow * HALO_MASK[yv]) >> 8) * HALO_LUMA) >> 8, 4095U);
                l2 = l1 + ((h * (4095U - l1)) >> 12);
            }
            l3 = l2 + ((min_u(sl, 4095U) * (4095U - l2)) >> 12);
            luma[x] = (u8)SRGB[l3];
            if (chroma && !(x & 1U)) {
                u32 rc = RECIP[l3 >> 4];
                int sv = (int)min_u(((lv - l0) * rc) >> 12, 256U);
                int sg = (int)min_u(((l1 - lv) * rc) >> 12, 256U);
                int sh = (int)min_u(((l2 - l1) * rc) >> 12, 256U);
                int sk = (int)min_u(((l3 - l2) * rc) >> 12, 256U);
                int cb = (int)crow[x] - 128, cr = (int)crow[x + 1U] - 128;
                /* The veil is the mixed light of the whole scene: neutral. */
                cb -= (cb * sv) >> 8;
                cr -= (cr * sv) >> 8;
                if (any) {
                    Lerp lc = lerp(x - gr->left, gr->shift + 1U, gr->cw);
                    int tb = (int)mix(gr->rb, lc) - 32768, tr = (int)mix(gr->rr, lc) - 32768;
                    cb += ((tb - cb) * sg) >> 8;
                    cr += ((tr - cr) * sg) >> 8;
                }
                cb += ((HALO_CB - cb) * sh) >> 8;
                cr += ((HALO_CR - cr) * sh) >> 8;
                cb += ((scb - cb) * sk) >> 8;
                cr += ((scr - cr) * sk) >> 8;
                crow[x] = (u8)clamp8(cb + 128);
                crow[x + 1U] = (u8)clamp8(cr + 128);
            }
        }
    }
}

void hdf_apply(PictureSet *set, u32 strength)
{
    u32 i, halo = setting_get(SETTING_HALO), mist = setting_get(SETTING_MIST), streak = setting_get(SETTING_STREAK);
    Grid gr;
    if (strength > 3U) strength = 3U;
    if (mist > 2U) mist = 2U;
    if (streak > 2U) streak = 2U;
    for (i = 0; i < set->count; ++i) {
        const Plane *p = &set->planes[i];
        if (!layout(&gr, p, set->scratch)) continue;
        gather(p, &gr, set->mono, mist);
        spread(&gr, mist);
        blend(p, &gr, strength, halo, mist, streak, set->mono);
    }
}
