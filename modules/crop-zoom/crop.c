/* Crop Zoom: a 35, 40, or 50 mm field of view from the 28 mm lens, at the
   full 24 MP size of the JPEG.

   The native crop of the GR III records a smaller JPEG and a cropped DNG.
   This module keeps both whole: the DNG stays the full sensor, and the JPEG
   keeps its size. The filter takes the centre of each recorded plane (28/40
   = 70% of the width and height for 40 mm) and resamples it back to the
   whole plane with a Catmull-Rom filter, a sharp 4 x 4 cubic. A 40 mm JPEG
   then holds the detail of about 11.8 MP of the sensor at 6000 x 4000, so it
   prints and crops like a native file. The live view shows the frame of the
   crop.

   The resample is separable and runs in place in the plane:

   1. horizontal: each source row that the vertical step reads is copied to
      the scratch memory, then written back zoomed;
   2. vertical: the output rows read source rows nearer the centre, so the
      rows above the centre are written top-down and the rows below it
      bottom-up, each before any row it reads is overwritten. The few rows
      at the centre, which read rows on both sides, go to the scratch memory
      first and are copied back last.

   A crop of ratio a/b maps output pixel x of an n-pixel axis to source
   pixel (2a x + (b - a)(n - 1)) / 2b, centre on centre, so the phase of the
   cubic repeats every 2b pixels and its weights come from a table. */
#include "grkit.h"
#include "grkit_module.h"
#include "crop_tables.h"

void crop_apply(PictureSet *set, u32 focal);
void crop_mask(const Surface *s, u32 focal);
u32 crop_fits(u32 rowbytes, u32 rows, u32 focal);
void crop_event(u32 event, u32 a, u32 b);

/* An axis shorter than this is left as it is: the cubic needs margins. */
#define MIN_AXIS 32U

/* One channel layout of a plane area: `w` samples of `step` bytes (1 for
   luma, 2 for interleaved CbCr) in each of `h` rows, `pitch` bytes apart. */
typedef struct { u32 base, pitch, w, h, step; } Area;

/* The rows of the vertical step: rows < top are written top-down, rows >
   bottom bottom-up, and the middle rows from top to bottom go through the
   scratch memory. */
typedef struct { u32 top, bottom; } Middle;

/* The source position of output 0, in whole pixels and in phases of 1/2b. */
static void start(u32 n, u32 focal, u32 *idx, u32 *ph)
{
    u32 b2 = 2U * CROP_B[focal], s = (u32)(CROP_B[focal] - CROP_A[focal]) * (n - 1U);
    *idx = divide(s, b2);
    *ph = s - *idx * b2;
}

static u32 source_of(u32 j, u32 n, u32 focal)
{
    u32 b2 = 2U * CROP_B[focal];
    return divide(2U * CROP_A[focal] * j + (u32)(CROP_B[focal] - CROP_A[focal]) * (n - 1U), b2);
}

static u32 clamp8(int v) { return v < 0 ? 0U : v > 255 ? 255U : (u32)v; }

/* The middle rows of an axis of n rows, and whether the passes do not
   overwrite a row that another pass still reads. */
static u32 middle(u32 n, u32 focal, Middle *m)
{
    u32 top = 0, bottom = n - 1U;
    while (top < n && source_of(top, n, focal) >= top + 1U) ++top;
    while (bottom > 0 && source_of(bottom, n, focal) + 2U <= bottom) --bottom;
    if (top == 0 || bottom + 1U >= n || top > bottom) return 0;
    /* The top pass reads up to row source(top - 1) + 2, which the bottom
       pass must not have written: it writes rows > bottom. */
    if (source_of(top - 1U, n, focal) + 2U > bottom) return 0;
    if (source_of(bottom + 1U, n, focal) < top + 1U) return 0;
    m->top = top;
    m->bottom = bottom;
    return 1;
}

/* Whether an area of `rows` rows of `rowbytes` bytes fits the scratch
   memory, for the host test and for each plane. */
u32 crop_fits(u32 rowbytes, u32 rows, u32 focal)
{
    Middle m;
    if (focal == 0 || focal > 3U || rows < MIN_AXIS || !middle(rows, focal, &m)) return 0;
    return rowbytes <= GRKIT_PLANE_SCRATCH && (m.bottom - m.top + 1U) * rowbytes <= GRKIT_PLANE_SCRATCH;
}

/* Step 1: zoom row `r` horizontally, through the scratch copy `buf`. */
static void zoom_row(const Area *a, u32 r, u32 focal, u8 *buf)
{
    volatile u8 *row = GRKIT_BYTES(a->base + r * a->pitch);
    u32 bytes = a->w * a->step, b2 = 2U * CROP_B[focal], inc = 2U * CROP_A[focal];
    u32 phase0 = CROP_PHASE0[focal], idx, ph, i, c;
    for (i = 0; i < bytes; ++i) buf[i] = row[i];
    start(a->w, focal, &idx, &ph);
    for (i = 0; i < a->w; ++i) {
        const short *w = CROP_W[phase0 + ph];
        const u8 *src = buf + (idx - 1U) * a->step;
        for (c = 0; c < a->step; ++c) {
            int v = w[0] * src[c] + w[1] * src[c + a->step] + w[2] * src[c + 2U * a->step] +
                    w[3] * src[c + 3U * a->step];
            row[i * a->step + c] = (u8)clamp8((v + 2048) >> 12);
        }
        ph += inc;
        if (ph >= b2) { ph -= b2; ++idx; }
    }
}

/* Step 2 for output row j: from the four source rows of its phase into
   `out`, which is a plane row or a scratch row. */
static void zoom_column(const Area *a, u32 j, u32 focal, volatile u8 *out)
{
    u32 b2 = 2U * CROP_B[focal];
    u32 s = 2U * CROP_A[focal] * j + (u32)(CROP_B[focal] - CROP_A[focal]) * (a->h - 1U);
    u32 idx = divide(s, b2), bytes = a->w * a->step, i;
    const short *w = CROP_W[CROP_PHASE0[focal] + (s - idx * b2)];
    volatile u8 *r0 = GRKIT_BYTES(a->base + (idx - 1U) * a->pitch);
    volatile u8 *r1 = GRKIT_BYTES(a->base + idx * a->pitch);
    volatile u8 *r2 = GRKIT_BYTES(a->base + (idx + 1U) * a->pitch);
    volatile u8 *r3 = GRKIT_BYTES(a->base + (idx + 2U) * a->pitch);
    for (i = 0; i < bytes; ++i) {
        int v = w[0] * r0[i] + w[1] * r1[i] + w[2] * r2[i] + w[3] * r3[i];
        out[i] = (u8)clamp8((v + 2048) >> 12);
    }
}

static void zoom_area(const Area *a, u32 focal, u8 *scratch)
{
    u32 bytes = a->w * a->step, j, first, last;
    Middle m;
    if (a->w < MIN_AXIS || a->h < MIN_AXIS || !crop_fits(bytes, a->h, focal) || !middle(a->h, focal, &m)) return;
    /* Step 1 on the rows that step 2 reads. */
    first = source_of(0, a->h, focal) - 1U;
    last = source_of(a->h - 1U, a->h, focal) + 2U;
    for (j = first; j <= last; ++j) zoom_row(a, j, focal, scratch);
    /* Step 2: the middle rows to the scratch memory, the sides in place,
       then the middle rows back. */
    for (j = m.top; j <= m.bottom; ++j) zoom_column(a, j, focal, scratch + (j - m.top) * bytes);
    for (j = 0; j < m.top; ++j) zoom_column(a, j, focal, GRKIT_BYTES(a->base + j * a->pitch));
    for (j = a->h - 1U; j > m.bottom; --j) zoom_column(a, j, focal, GRKIT_BYTES(a->base + j * a->pitch));
    for (j = m.top; j <= m.bottom; ++j) {
        volatile u8 *row = GRKIT_BYTES(a->base + j * a->pitch);
        const u8 *src = scratch + (j - m.top) * bytes;
        u32 i;
        for (i = 0; i < bytes; ++i) row[i] = src[i];
    }
}

static void zoom_plane(const Plane *p, u32 mono, u32 focal, u8 *scratch)
{
    Area a;
    if (p->right > p->width || p->bottom > p->height || p->right <= p->left || p->bottom <= p->top) return;
    a.base = p->y + p->top * p->pitch + p->left;
    a.pitch = p->pitch;
    a.w = p->right - p->left;
    a.h = p->bottom - p->top;
    a.step = 1;
    zoom_area(&a, focal, scratch);
    if (mono) return;
    /* The CbCr pairs of the visible area: one pair for each two pixels, and
       with `half` one chroma row for each two luma rows. */
    {
        u32 r0 = p->half ? p->top >> 1 : p->top, r1 = p->half ? (p->bottom - 1U) >> 1 : p->bottom - 1U;
        a.base = p->c + r0 * p->pitch + (p->left & ~1U);
        a.w = ((p->right - 1U) >> 1) - (p->left >> 1) + 1U;
        a.h = r1 - r0 + 1U;
        a.step = 2;
        zoom_area(&a, focal, scratch);
    }
}

void crop_apply(PictureSet *set, u32 focal)
{
    u32 i;
    if (focal == 0 || focal > 3U) return;
    for (i = 0; i < set->count && i < 4U; ++i) zoom_plane(&set->planes[i], set->mono, focal, set->scratch);
}

/* The live view: the frame of the crop inside the photo area, in the style
   of MASK: black at 50% (0) or 100% (1) outside it, or frame lines only (2).
   A frame line is 2 white pixels inside each edge. */
static const u32 mask_colors[2] = {0x80000000U, 0xff000000U};
#define FRAME_COLOR 0xffffffffU
#define FRAME_WIDTH 2U

static void span(const Surface *s, u32 y, u32 from, u32 to, u32 color)
{
    volatile u32 *row = &SURFACE_PIXEL(s, 0, y);
    for (; from < to; ++from) row[from] = color;
}

void crop_mask(const Surface *s, u32 focal)
{
    const u16 *box = s->box;
    u32 bw, bh, iw, ih, l, t, r, b, y, style;
    if (focal == 0 || focal > 3U || box[2] <= box[0] || box[3] <= box[1]) return;
    bw = box[2] - box[0];
    bh = box[3] - box[1];
    iw = divide(bw * CROP_A[focal], CROP_B[focal]);
    ih = divide(bh * CROP_A[focal], CROP_B[focal]);
    l = box[0] + ((bw - iw) >> 1);
    t = box[1] + ((bh - ih) >> 1);
    r = l + iw;
    b = t + ih;
    style = setting_get(SETTING_MASK);
    if (style == 2) {
        for (y = t; y < b; ++y) {
            if (y < t + FRAME_WIDTH || y + FRAME_WIDTH >= b) { span(s, y, l, r, FRAME_COLOR); continue; }
            span(s, y, l, l + FRAME_WIDTH, FRAME_COLOR);
            span(s, y, r - FRAME_WIDTH, r, FRAME_COLOR);
        }
        return;
    }
    {
        u32 color = mask_colors[style == 1];
        for (y = box[1]; y < box[3]; ++y) {
            if (y < t || y >= b) { span(s, y, box[0], box[2], color); continue; }
            span(s, y, box[0], l, color);
            span(s, y, r, box[2], color);
        }
    }
}

/* Half Frame: the 4:3 aspect of the core while HALF is on. Held upright,
   the camera records a portrait 3:4 picture, the shape of a half-frame
   negative, and FOCAL zooms it as any other area. Off returns to the native
   3:2. Another ratio from Framing turns HALF off, so the row never shows On
   for a picture that is not 3:4. A setting_set in this handler sends no
   event, so the two settings cannot loop. */
/* The host harness has no handles for used settings (its context lists
   none), so a host test reads the aspect from entry 15 of its settings,
   below those of the module. */
#ifdef GRKIT_TEST_H
#define ASPECT_HANDLE 15U
#else
#define ASPECT_HANDLE USED_CORE_ASPECT
#endif
#define ASPECT_NATIVE 0U
#define ASPECT_4_3 3U

void crop_event(u32 event, u32 a, u32 b)
{
    if (event == GRKIT_EVENT_SYSTEM_READY) {
        if (setting_get(SETTING_HALF)) setting_set(ASPECT_HANDLE, ASPECT_4_3);
        return;
    }
    if (event != GRKIT_EVENT_SETTING_CHANGED) return;
    if (a == SETTING_HALF) {
        if (b) setting_set(ASPECT_HANDLE, ASPECT_4_3);
        else if (setting_get(ASPECT_HANDLE) == ASPECT_4_3) setting_set(ASPECT_HANDLE, ASPECT_NATIVE);
    } else if (a == ASPECT_HANDLE && b != ASPECT_4_3 && setting_get(SETTING_HALF)) {
        setting_set(SETTING_HALF, 0);
    }
}
