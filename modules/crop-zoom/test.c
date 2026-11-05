/* The host test of Crop Zoom (grkit module test): the zoomed planes against
   a reference resample in floating point, the area outside the visible
   area, the scratch memory of the GR III sizes, and the live-view frame. */
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include "grkit_module.h"

void crop_apply(PictureSet *set, u32 focal);
void crop_mask(const Surface *s, u32 focal);
u32 crop_fits(u32 rowbytes, u32 rows, u32 focal);
void crop_event(u32 event, u32 a, u32 b);

static const double RATIO[4] = {1.0, 0.8, 0.7, 0.56};

static u8 get(u32 address) { return GRKIT_BYTES(address)[0]; }
static void put(u32 address, u8 v) { GRKIT_BYTES(address)[0] = v; }

static void cubic(double t, double w[4])
{
    w[0] = (-t * t * t + 2 * t * t - t) / 2;
    w[1] = (3 * t * t * t - 5 * t * t + 2) / 2;
    w[2] = (-3 * t * t * t + 4 * t * t + t) / 2;
    w[3] = (t * t * t - t * t) / 2;
}
static u8 round8(double v) { return v < 0 ? 0 : v > 255 ? 255 : (u8)floor(v + 0.5); }

/* The source of output j on an axis of n samples, centre on centre. */
static double source(u32 j, u32 n, u32 focal) { return (j + 0.5 - n / 2.0) * RATIO[focal] + n / 2.0 - 0.5; }

/* The reference: `src` holds w x h samples of `step` channels; the result
   goes to `out`, horizontal then vertical, rounded to bytes in between as
   the module does. */
static void reference(const u8 *src, u8 *out, u32 w, u32 h, u32 step, u32 focal)
{
    u8 *mid = malloc((size_t)w * h * step);
    for (u32 y = 0; y < h; ++y)
        for (u32 x = 0; x < w; ++x) {
            double s = source(x, w, focal), wt[4];
            int i = (int)floor(s);
            cubic(s - i, wt);
            for (u32 c = 0; c < step; ++c) {
                double v = 0;
                for (int k = 0; k < 4; ++k) v += wt[k] * src[((size_t)y * w + (u32)(i - 1 + k)) * step + c];
                mid[((size_t)y * w + x) * step + c] = round8(v);
            }
        }
    for (u32 y = 0; y < h; ++y) {
        double s = source(y, h, focal), wt[4];
        int i = (int)floor(s);
        cubic(s - i, wt);
        for (u32 x = 0; x < w * step; ++x) {
            double v = 0;
            for (int k = 0; k < 4; ++k) v += wt[k] * mid[(size_t)(u32)(i - 1 + k) * w * step + x];
            out[(size_t)y * w * step + x] = round8(v);
        }
    }
    free(mid);
}

/* A plane of W x H with 4:2:0 chroma after the luma, a smooth pattern in
   luma and chroma, and the visible area (l, t, r, b). */
#define W 1200U
#define H 800U
static Plane scene(u32 l, u32 t, u32 r, u32 b)
{
    Plane p = {GRKIT_TEST_PLANES, GRKIT_TEST_PLANES + W * H, W, H, W, 1, l, t, r, b};
    for (u32 y = 0; y < H; ++y)
        for (u32 x = 0; x < W; ++x) put(p.y + y * W + x, round8(128 + 90 * sin(x * 0.05) * cos(y * 0.07) + 20 * sin(x * 0.31 + y * 0.17)));
    for (u32 y = 0; y < H / 2; ++y)
        for (u32 x = 0; x < W / 2; ++x) {
            put(p.c + y * W + 2 * x, round8(128 + 60 * sin(x * 0.11 + y * 0.03)));
            put(p.c + y * W + 2 * x + 1, round8(128 - 50 * cos(x * 0.02 - y * 0.09)));
        }
    return p;
}

/* The largest difference between an area of the plane and the reference
   from the saved samples of the same area. */
static int check_area(u32 base, u32 pitch, u32 w, u32 h, u32 step, const u8 *before, u32 focal)
{
    u8 *ref = malloc((size_t)w * h * step);
    int worst = 0;
    reference(before, ref, w, h, step, focal);
    for (u32 y = 0; y < h; ++y)
        for (u32 x = 0; x < w * step; ++x) {
            int d = abs((int)get(base + y * pitch + x) - (int)ref[(size_t)y * w * step + x]);
            if (d > worst) worst = d;
        }
    free(ref);
    return worst;
}

static u8 *save(u32 base, u32 pitch, u32 w, u32 h, u32 step)
{
    u8 *s = malloc((size_t)w * h * step);
    for (u32 y = 0; y < h; ++y)
        for (u32 x = 0; x < w * step; ++x) s[(size_t)y * w * step + x] = get(base + y * pitch + x);
    return s;
}

static void check_zoom(u32 focal, u32 l, u32 t, u32 r, u32 b)
{
    Plane p = scene(l, t, r, b);
    PictureSet set = {{p}, 1, 0, 0, grkit_test_scratch, 0};
    u32 cl = l & ~1U, cw = ((r - 1) >> 1) - (l >> 1) + 1, ct = t >> 1, ch = ((b - 1) >> 1) - ct + 1;
    u8 *ly = save(p.y + t * W + l, W, r - l, b - t, 1);
    u8 *lc = save(p.c + ct * W + cl, W, cw, ch, 2);
    u8 *all = save(p.y, W, W, H, 1);
    crop_apply(&set, focal);
    assert(check_area(p.y + t * W + l, W, r - l, b - t, 1, ly, focal) <= 2);
    assert(check_area(p.c + ct * W + cl, W, cw, ch, 2, lc, focal) <= 2);
    /* Luma outside the visible area stays as it was. */
    for (u32 y = 0; y < H; ++y)
        for (u32 x = 0; x < W; ++x)
            if (x < l || x >= r || y < t || y >= b) assert(get(p.y + y * W + x) == all[y * W + x]);
    free(ly);
    free(lc);
    free(all);
}

int main(void)
{
    /* The descriptor: FOCAL (4 values), HALF (2), and MASK (3), the use of
       CORE.ASPECT, a page with its settings rows, a layer, a filter, and the
       event subscriber. */
    assert(grkit_module.setting_count == 3 && grkit_module.settings[0].values == 4 &&
           grkit_module.settings[1].values == 2 && grkit_module.settings[2].values == 3);
    assert(grkit_module.use_count == 1 && grkit_module.registration_count == 4 && grkit_module.tile_count == 4);
    assert(SETTING_FOCAL == GRKIT_SETTING(0) && SETTING_HALF == GRKIT_SETTING(1) && SETTING_MASK == GRKIT_SETTING(2));

    /* Half Frame: On sets 4:3, Off returns to 3:2, another ratio turns it
       off, and a start with HALF on sets 4:3 again. */
    {
        u32 *v = grkit_test_settings;
        v[15U] = 0;
        v[SETTING_HALF] = 1;
        crop_event(GRKIT_EVENT_SETTING_CHANGED, SETTING_HALF, 1);
        assert(v[15U] == 3);
        v[SETTING_HALF] = 0;
        crop_event(GRKIT_EVENT_SETTING_CHANGED, SETTING_HALF, 0);
        assert(v[15U] == 0);
        /* Off leaves a ratio that is not 4:3 alone. */
        v[15U] = 5;
        crop_event(GRKIT_EVENT_SETTING_CHANGED, SETTING_HALF, 0);
        assert(v[15U] == 5);
        v[SETTING_HALF] = 1;
        crop_event(GRKIT_EVENT_SETTING_CHANGED, SETTING_HALF, 1);
        v[15U] = 2;
        crop_event(GRKIT_EVENT_SETTING_CHANGED, 15U, 2);
        assert(v[SETTING_HALF] == 0);
        /* 4:3 chosen in Framing keeps HALF as it is. */
        v[SETTING_HALF] = 1;
        v[15U] = 3;
        crop_event(GRKIT_EVENT_SETTING_CHANGED, 15U, 3);
        assert(v[SETTING_HALF] == 1);
        v[15U] = 0;
        crop_event(GRKIT_EVENT_SYSTEM_READY, 0, 0);
        assert(v[15U] == 3);
        v[SETTING_HALF] = 0;
        v[15U] = 0;
    }

    /* The GR III planes fit the scratch memory at each focal length: the
       24 MP picture (6000 x 4000, its CbCr rows of 6000 bytes, 4:2:2 or
       4:2:0), the screennail, and the thumbnail. */
    for (u32 f = 1; f <= 3; ++f) {
        assert(crop_fits(6000, 4000, f) && crop_fits(6000, 2000, f) && crop_fits(4000, 6000, f));
        assert(crop_fits(1620, 1080, f) && crop_fits(160, 120, f));
    }
    assert(!crop_fits(6000, 4000, 0) && !crop_fits(160, 20, 2));

    /* Each focal length matches the reference over the full plane, and over
       a visible area with odd edges, as a Framing crop gives. */
    for (u32 f = 1; f <= 3; ++f) check_zoom(f, 0, 0, W, H);
    check_zoom(2, 101, 37, 1061, 763);
    /* Half Frame at 40 mm: the 4:3 area of the plane, centred. */
    check_zoom(2, 67, 0, 1133, 800);

    /* A flat picture stays flat: the weights sum to one. */
    {
        Plane p = {GRKIT_TEST_PLANES, GRKIT_TEST_PLANES + W * H, W, H, W, 0, 0, 0, W, H};
        PictureSet set = {{p}, 1, 0, 1, grkit_test_scratch, 0};
        for (u32 i = 0; i < W * H; ++i) put(p.y + i, 77);
        crop_apply(&set, 2);
        for (u32 i = 0; i < W * H; ++i) assert(get(p.y + i) == 77);
    }

    /* The centre stays put and the edges show 70% of the frame: a ramp of
       x / 4 at 40 mm is x * 0.7 / 4 + 45 / 4 after the zoom. */
    {
        Plane p = {GRKIT_TEST_PLANES, GRKIT_TEST_PLANES + W * H, W, H, W, 0, 0, 0, W, H};
        PictureSet set = {{p}, 1, 0, 1, grkit_test_scratch, 0};
        for (u32 y = 0; y < H; ++y)
            for (u32 x = 0; x < W; ++x) put(p.y + y * W + x, (u8)(x * 255 / (W - 1)));
        crop_apply(&set, 2);
        assert(abs((int)get(p.y + 400 * W + 600) - 127) <= 1);
        assert(abs((int)get(p.y + 400 * W + 0) - 38) <= 1);   /* 255 * 0.15 */
        assert(abs((int)get(p.y + 400 * W + W - 1) - 217) <= 1); /* 255 * 0.85 */
    }

    /* The live view: at 40 mm, the frame is 70% of the photo area, centred. */
    {
        static const u16 full[4] = {0, 0, 720, 480};
        Surface s = grkit_test_surface(full);
        /* Off draws nothing. */
        grkit_test_clear();
        crop_mask(&s, 0);
        assert(grkit_test_pixel(0, 0) == GRKIT_TEST_CLEAR);
        /* Dark: 50% black outside the frame (108..612 x 72..408), clear inside. */
        grkit_test_settings[SETTING_MASK] = 0;
        grkit_test_clear();
        crop_mask(&s, 2);
        for (u32 y = 0; y < 480; ++y)
            for (u32 x = 0; x < 720; ++x) {
                int in = x >= 108 && x < 612 && y >= 72 && y < 408;
                assert(grkit_test_pixel(x, y) == (in ? GRKIT_TEST_CLEAR : 0x80000000U));
            }
        /* Frame lines: white on the edges of the frame, clear elsewhere. */
        grkit_test_settings[SETTING_MASK] = 2;
        grkit_test_clear();
        crop_mask(&s, 2);
        assert(grkit_test_pixel(108, 200) == 0xffffffffU && grkit_test_pixel(611, 200) == 0xffffffffU);
        assert(grkit_test_pixel(300, 72) == 0xffffffffU && grkit_test_pixel(300, 407) == 0xffffffffU);
        assert(grkit_test_pixel(360, 240) == GRKIT_TEST_CLEAR && grkit_test_pixel(50, 50) == GRKIT_TEST_CLEAR);
        grkit_test_settings[SETTING_MASK] = 0;
    }
    return 0;
}
