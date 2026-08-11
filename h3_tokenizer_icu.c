/* Portable tokenizer for Linux/CUDA builds (public ICU).
 * API-compatible with h3_tokenizer.h / h3_tokenizer.m.
 *
 * Loads HuggingFace-style tokenizer.json (vocab + merges) or vocab.json.
 * Normalization: ICU NFKC. Encoding: byte-level BPE with merge ranks.
 */
#include "h3_tokenizer.h"

#include <ctype.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <unicode/normalizer2.h>
#include <unicode/ustring.h>
#include <unicode/utypes.h>

typedef struct {
    char *key;
    size_t key_len;
    uint32_t id;
    int used;
} h3_tok_kv;

struct h3_tokenizer {
    char **id_to_tok;
    size_t *id_to_len;
    size_t vocab_size;
    h3_tok_kv *map;
    size_t map_cap;
    /* merge ranks for pair "a b" strings */
    h3_tok_kv *merges;
    size_t merges_cap;
    size_t merges_count;
};

static uint64_t h3_fnv(const char *s, size_t n)
{
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) {
        h ^= (unsigned char)s[i];
        h *= 1099511628211ull;
    }
    return h;
}

static int h3_kv_grow(h3_tok_kv **map, size_t *cap, size_t need)
{
    size_t ncap = *cap ? *cap : 1024;
    while (ncap < need * 2) ncap *= 2;
    if (ncap == *cap) return 0;
    h3_tok_kv *nm = (h3_tok_kv *)calloc(ncap, sizeof(h3_tok_kv));
    if (!nm) return -1;
    for (size_t i = 0; i < *cap; ++i) {
        if (!(*map)[i].used) continue;
        size_t j = (size_t)(h3_fnv((*map)[i].key, (*map)[i].key_len) & (ncap - 1));
        while (nm[j].used) j = (j + 1) & (ncap - 1);
        nm[j] = (*map)[i];
    }
    free(*map);
    *map = nm;
    *cap = ncap;
    return 0;
}

static int h3_kv_put(h3_tok_kv **map, size_t *cap, const char *key, size_t key_len, uint32_t id)
{
    if (h3_kv_grow(map, cap, (*cap ? *cap/2 : 1) + 8) != 0) return -1;
    /* recount used roughly by probing growth already done */
    if (*cap == 0) return -1;
    size_t j = (size_t)(h3_fnv(key, key_len) & (*cap - 1));
    for (;;) {
        if (!(*map)[j].used) {
            char *k = (char *)malloc(key_len + 1);
            if (!k) return -1;
            memcpy(k, key, key_len);
            k[key_len] = '\0';
            (*map)[j].key = k;
            (*map)[j].key_len = key_len;
            (*map)[j].id = id;
            (*map)[j].used = 1;
            return 0;
        }
        if ((*map)[j].key_len == key_len && memcmp((*map)[j].key, key, key_len) == 0) {
            (*map)[j].id = id;
            return 0;
        }
        j = (j + 1) & (*cap - 1);
    }
}

static int h3_kv_get(const h3_tok_kv *map, size_t cap, const char *key, size_t key_len, uint32_t *id)
{
    if (!map || !cap) return 0;
    size_t j = (size_t)(h3_fnv(key, key_len) & (cap - 1));
    for (size_t n = 0; n < cap; ++n) {
        if (!map[j].used) return 0;
        if (map[j].key_len == key_len && memcmp(map[j].key, key, key_len) == 0) {
            if (id) *id = map[j].id;
            return 1;
        }
        j = (j + 1) & (cap - 1);
    }
    return 0;
}

static char *h3_slurp(const char *path, size_t *out_n)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    char *b = (char *)malloc((size_t)sz + 1);
    if (!b) { fclose(f); return NULL; }
    size_t n = fread(b, 1, (size_t)sz, f);
    fclose(f);
    b[n] = '\0';
    if (out_n) *out_n = n;
    return b;
}

static int h3_json_unescape(const char *s, size_t n, char **out, size_t *out_n)
{
    char *d = (char *)malloc(n + 1);
    if (!d) return -1;
    size_t j = 0;
    for (size_t i = 0; i < n; ++i) {
        if (s[i] == '\\' && i + 1 < n) {
            char c = s[++i];
            if (c == 'n') d[j++] = '\n';
            else if (c == 't') d[j++] = '\t';
            else if (c == 'r') d[j++] = '\r';
            else if (c == '"') d[j++] = '"';
            else if (c == '\\') d[j++] = '\\';
            else if (c == '/') d[j++] = '/';
            else if (c == 'u' && i + 4 < n) {
                /* skip simple unicode escapes as raw utf8 passthrough fallback */
                unsigned v = 0;
                for (int k = 0; k < 4; ++k) {
                    char h = s[++i];
                    v <<= 4;
                    if (h >= '0' && h <= '9') v |= (unsigned)(h - '0');
                    else if (h >= 'a' && h <= 'f') v |= (unsigned)(h - 'a' + 10);
                    else if (h >= 'A' && h <= 'F') v |= (unsigned)(h - 'A' + 10);
                }
                if (v < 0x80) d[j++] = (char)v;
                else if (v < 0x800) {
                    d[j++] = (char)(0xC0 | (v >> 6));
                    d[j++] = (char)(0x80 | (v & 0x3F));
                } else {
                    d[j++] = (char)(0xE0 | (v >> 12));
                    d[j++] = (char)(0x80 | ((v >> 6) & 0x3F));
                    d[j++] = (char)(0x80 | (v & 0x3F));
                }
            } else d[j++] = c;
        } else d[j++] = s[i];
    }
    d[j] = '\0';
    *out = d;
    if (out_n) *out_n = j;
    return 0;
}

static int h3_add_vocab(h3_tokenizer *tok, const char *token, size_t tlen, uint32_t id)
{
    if (id >= tok->vocab_size) {
        size_t ncap = tok->vocab_size ? tok->vocab_size : 1024;
        while (ncap <= id) ncap *= 2;
        char **nt = (char **)realloc(tok->id_to_tok, ncap * sizeof(char *));
        size_t *nl = (size_t *)realloc(tok->id_to_len, ncap * sizeof(size_t));
        if (!nt || !nl) return -1;
        memset(nt + tok->vocab_size, 0, (ncap - tok->vocab_size) * sizeof(char *));
        memset(nl + tok->vocab_size, 0, (ncap - tok->vocab_size) * sizeof(size_t));
        tok->id_to_tok = nt;
        tok->id_to_len = nl;
        tok->vocab_size = ncap;
    }
    free(tok->id_to_tok[id]);
    tok->id_to_tok[id] = (char *)malloc(tlen + 1);
    if (!tok->id_to_tok[id]) return -1;
    memcpy(tok->id_to_tok[id], token, tlen);
    tok->id_to_tok[id][tlen] = '\0';
    tok->id_to_len[id] = tlen;
    return h3_kv_put(&tok->map, &tok->map_cap, token, tlen, id);
}

static const char *h3_find_key(const char *json, const char *key)
{
    char pat[256];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    return strstr(json, pat);
}

static int h3_load_tokenizer_json(h3_tokenizer *tok, const char *path, char *error, size_t error_size)
{
    size_t n = 0;
    char *json = h3_slurp(path, &n);
    if (!json) {
        if (error && error_size) snprintf(error, error_size, "cannot read %s: %s", path, strerror(errno));
        return -1;
    }
    /* Find "vocab" object */
    const char *v = h3_find_key(json, "vocab");
    if (!v) v = json; /* some files are pure vocab object */
    const char *p = strchr(v, '{');
    if (!p) { free(json); if (error&&error_size) snprintf(error,error_size,"vocab object not found"); return -1; }
    ++p;
    size_t max_id = 0;
    while (*p && *p != '}') {
        while (*p && *p != '"' && *p != '}') ++p;
        if (*p != '"') break;
        ++p;
        const char *ts = p;
        while (*p && !(*p == '"' && p[-1] != '\\')) ++p;
        size_t raw_len = (size_t)(p - ts);
        char *token = NULL; size_t tlen = 0;
        if (h3_json_unescape(ts, raw_len, &token, &tlen) != 0) { free(json); return -1; }
        if (*p == '"') ++p;
        while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ':')) ++p;
        char *end = NULL;
        long id = strtol(p, &end, 10);
        if (end == p) { free(token); break; }
        p = end;
        if (id < 0) { free(token); continue; }
        if (h3_add_vocab(tok, token, tlen, (uint32_t)id) != 0) { free(token); free(json); return -1; }
        if ((size_t)id + 1 > max_id) max_id = (size_t)id + 1;
        free(token);
        while (*p && *p != ',' && *p != '}') ++p;
        if (*p == ',') ++p;
    }
    /* shrink logical vocab_size to max id */
    if (max_id > 0) tok->vocab_size = max_id;

    /* merges array optional */
    const char *m = h3_find_key(json, "merges");
    if (m) {
        const char *a = strchr(m, '[');
        if (a) {
            ++a;
            uint32_t rank = 0;
            while (*a && *a != ']') {
                while (*a && *a != '"' && *a != ']') ++a;
                if (*a != '"') break;
                ++a;
                const char *ms = a;
                while (*a && !(*a == '"' && a[-1] != '\\')) ++a;
                size_t raw_len = (size_t)(a - ms);
                char *merge = NULL; size_t mlen = 0;
                if (h3_json_unescape(ms, raw_len, &merge, &mlen) != 0) { free(json); return -1; }
                if (*a == '"') ++a;
                if (h3_kv_put(&tok->merges, &tok->merges_cap, merge, mlen, rank) != 0) {
                    free(merge); free(json); return -1;
                }
                free(merge);
                ++rank; ++tok->merges_count;
                while (*a && *a != ',' && *a != ']') ++a;
                if (*a == ',') ++a;
            }
        }
    }
    free(json);
    if (max_id == 0) {
        if (error && error_size) snprintf(error, error_size, "empty vocab in %s", path);
        return -1;
    }
    return 0;
}

h3_tokenizer *h3_tokenizer_load(const char *tokenizer_json,
                                char *error, size_t error_size)
{
    if (error && error_size) error[0] = '\0';
    if (!tokenizer_json) {
        if (error && error_size) snprintf(error, error_size, "tokenizer_json is NULL");
        return NULL;
    }
    h3_tokenizer *tok = (h3_tokenizer *)calloc(1, sizeof(*tok));
    if (!tok) return NULL;
    if (h3_load_tokenizer_json(tok, tokenizer_json, error, error_size) != 0) {
        h3_tokenizer_free(tok);
        return NULL;
    }
    return tok;
}

void h3_tokenizer_free(h3_tokenizer *tokenizer)
{
    if (!tokenizer) return;
    if (tokenizer->id_to_tok) {
        for (size_t i = 0; i < tokenizer->vocab_size; ++i) free(tokenizer->id_to_tok[i]);
        free(tokenizer->id_to_tok);
    }
    free(tokenizer->id_to_len);
    if (tokenizer->map) {
        for (size_t i = 0; i < tokenizer->map_cap; ++i) free(tokenizer->map[i].key);
        free(tokenizer->map);
    }
    if (tokenizer->merges) {
        for (size_t i = 0; i < tokenizer->merges_cap; ++i) free(tokenizer->merges[i].key);
        free(tokenizer->merges);
    }
    free(tokenizer);
}

static int h3_nfkc(const char *in, char **out)
{
    UErrorCode st = U_ZERO_ERROR;
    const UNormalizer2 *norm = unorm2_getNFKCInstance(&st);
    if (U_FAILURE(st) || !norm) return -1;
    int32_t cap = (int32_t)strlen(in) + 1;
    UChar *src = (UChar *)malloc((size_t)cap * sizeof(UChar));
    if (!src) return -1;
    int32_t sl = 0;
    u_strFromUTF8(src, cap, &sl, in, -1, &st);
    if (U_FAILURE(st)) {
        /* retry with larger */
        free(src);
        cap = sl + 8;
        src = (UChar *)malloc((size_t)cap * sizeof(UChar));
        if (!src) return -1;
        st = U_ZERO_ERROR;
        u_strFromUTF8(src, cap, &sl, in, -1, &st);
        if (U_FAILURE(st)) { free(src); return -1; }
    }
    int32_t dcap = sl * 2 + 8;
    UChar *dst = (UChar *)malloc((size_t)dcap * sizeof(UChar));
    if (!dst) { free(src); return -1; }
    st = U_ZERO_ERROR;
    int32_t dl = unorm2_normalize(norm, src, sl, dst, dcap, &st);
    if (st == U_BUFFER_OVERFLOW_ERROR) {
        free(dst);
        dcap = dl + 8;
        dst = (UChar *)malloc((size_t)dcap * sizeof(UChar));
        if (!dst) { free(src); return -1; }
        st = U_ZERO_ERROR;
        dl = unorm2_normalize(norm, src, sl, dst, dcap, &st);
    }
    if (U_FAILURE(st)) { free(src); free(dst); return -1; }
    int32_t ocap = dl * 3 + 8;
    char *o = (char *)malloc((size_t)ocap);
    if (!o) { free(src); free(dst); return -1; }
    int32_t ol = 0;
    st = U_ZERO_ERROR;
    u_strToUTF8(o, ocap, &ol, dst, dl, &st);
    if (U_FAILURE(st)) { free(src); free(dst); free(o); return -1; }
    free(src); free(dst);
    *out = o;
    return 0;
}

/* GPT2-style bytes-to-unicode is not always present; use direct bytes as 1-char pieces
 * if present in vocab, else single-byte longest match.
 */
static int h3_bpe_encode_word(const h3_tokenizer *tok, const char *word, size_t wlen,
                              uint32_t **ids, size_t *count, size_t *cap)
{
    if (wlen == 0) return 0;
    /* start with bytes as separate symbols (stored as 1-byte strings) */
    char **syms = (char **)malloc(wlen * sizeof(char *));
    size_t *slens = (size_t *)malloc(wlen * sizeof(size_t));
    size_t nsym = wlen;
    if (!syms || !slens) { free(syms); free(slens); return -1; }
    for (size_t i = 0; i < wlen; ++i) {
        syms[i] = (char *)malloc(2);
        if (!syms[i]) {
            for (size_t k = 0; k < i; ++k) free(syms[k]);
            free(syms); free(slens); return -1;
        }
        syms[i][0] = word[i];
        syms[i][1] = '\0';
        slens[i] = 1;
    }
    for (;;) {
        int best_rank = -1;
        size_t best_i = 0;
        for (size_t i = 0; i + 1 < nsym; ++i) {
            char pair[256];
            size_t need = slens[i] + 1 + slens[i+1];
            if (need + 1 > sizeof(pair)) continue;
            memcpy(pair, syms[i], slens[i]);
            pair[slens[i]] = ' ';
            memcpy(pair + slens[i] + 1, syms[i+1], slens[i+1]);
            pair[need] = '\0';
            uint32_t rank;
            if (h3_kv_get(tok->merges, tok->merges_cap, pair, need, &rank)) {
                if (best_rank < 0 || (int)rank < best_rank) {
                    best_rank = (int)rank;
                    best_i = i;
                }
            }
        }
        if (best_rank < 0) break;
        /* merge best_i and best_i+1 */
        size_t nl = slens[best_i] + slens[best_i+1];
        char *ns = (char *)malloc(nl + 1);
        if (!ns) {
            for (size_t k = 0; k < nsym; ++k) free(syms[k]);
            free(syms); free(slens); return -1;
        }
        memcpy(ns, syms[best_i], slens[best_i]);
        memcpy(ns + slens[best_i], syms[best_i+1], slens[best_i+1]);
        ns[nl] = '\0';
        free(syms[best_i]);
        free(syms[best_i+1]);
        syms[best_i] = ns;
        slens[best_i] = nl;
        for (size_t j = best_i + 1; j + 1 < nsym; ++j) {
            syms[j] = syms[j+1];
            slens[j] = slens[j+1];
        }
        --nsym;
    }
    for (size_t i = 0; i < nsym; ++i) {
        uint32_t id = H3_PAD_TOKEN_ID;
        if (!h3_kv_get(tok->map, tok->map_cap, syms[i], slens[i], &id)) {
            /* try unk / pad */
            if (!h3_kv_get(tok->map, tok->map_cap, "<unk>", 5, &id) &&
                !h3_kv_get(tok->map, tok->map_cap, "<UNK>", 5, &id)) {
                id = H3_PAD_TOKEN_ID;
            }
        }
        if (*count + 1 > *cap) {
            size_t ncap = (*cap ? *cap * 2 : 64);
            uint32_t *ni = (uint32_t *)realloc(*ids, ncap * sizeof(uint32_t));
            if (!ni) {
                for (size_t k = 0; k < nsym; ++k) free(syms[k]);
                free(syms); free(slens); return -1;
            }
            *ids = ni; *cap = ncap;
        }
        (*ids)[(*count)++] = id;
        free(syms[i]);
    }
    free(syms); free(slens);
    return 0;
}

int h3_tokenizer_encode(const h3_tokenizer *tokenizer, const char *utf8,
                        int pad_empty, uint32_t **ids, size_t *count,
                        char *error, size_t error_size)
{
    if (error && error_size) error[0] = '\0';
    if (!tokenizer || !utf8 || !ids || !count) return -1;
    *ids = NULL; *count = 0;
    char *norm = NULL;
    if (h3_nfkc(utf8, &norm) != 0) {
        norm = strdup(utf8);
        if (!norm) return -1;
    }
    size_t cap = 64;
    uint32_t *out = (uint32_t *)malloc(cap * sizeof(uint32_t));
    if (!out) { free(norm); return -1; }
    size_t n = 0;
    /* split on whitespace; preserve simple words */
    const char *p = norm;
    while (*p) {
        while (*p && isspace((unsigned char)*p)) ++p;
        if (!*p) break;
        const char *s = p;
        while (*p && !isspace((unsigned char)*p)) ++p;
        if (h3_bpe_encode_word(tokenizer, s, (size_t)(p - s), &out, &n, &cap) != 0) {
            free(out); free(norm); return -1;
        }
    }
    free(norm);
    if (n == 0 && pad_empty) {
        out[0] = H3_PAD_TOKEN_ID;
        n = 1;
    }
    *ids = out;
    *count = n;
    return 0;
}

void h3_tokenizer_ids_free(uint32_t *ids)
{
    free(ids);
}

char *h3_tokenizer_decode(const h3_tokenizer *tokenizer,
                          const uint32_t *ids, size_t count,
                          char *error, size_t error_size)
{
    if (error && error_size) error[0] = '\0';
    if (!tokenizer) return NULL;
    size_t cap = 64, n = 0;
    char *out = (char *)malloc(cap);
    if (!out) return NULL;
    out[0] = '\0';
    for (size_t i = 0; i < count; ++i) {
        uint32_t id = ids[i];
        if (id >= tokenizer->vocab_size || !tokenizer->id_to_tok[id]) continue;
        size_t tl = tokenizer->id_to_len[id];
        if (n + tl + 1 > cap) {
            while (cap < n + tl + 1) cap *= 2;
            char *no = (char *)realloc(out, cap);
            if (!no) { free(out); return NULL; }
            out = no;
        }
        memcpy(out + n, tokenizer->id_to_tok[id], tl);
        n += tl;
        out[n] = '\0';
    }
    return out;
}
