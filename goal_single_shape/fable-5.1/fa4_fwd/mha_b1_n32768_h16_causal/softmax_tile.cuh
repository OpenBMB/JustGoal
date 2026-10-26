#pragma once
// generated softmax tile kernels

DEVI void pair_mufu(uint32_t s0, uint32_t s1, uint64_t c2, uint64_t negm2, uint64_t& lacc, uint32_t& pk) {
    asm volatile("{\n\t.reg .b64 sab, xab, pab;\n\t.reg .f32 xa, xb, pa, pb;\n\t"
                 "mov.b64 sab, {%2, %3};\n\t"
                 "fma.rn.f32x2 xab, sab, %4, %5;\n\t"
                 "mov.b64 {xa, xb}, xab;\n\t"
                 "ex2.approx.ftz.f32 pa, xa;\n\t"
                 "ex2.approx.ftz.f32 pb, xb;\n\t"
                 "mov.b64 pab, {pa, pb};\n\t"
                 "add.rn.f32x2 %1, %1, pab;\n\t"
                 "cvt.rn.bf16x2.f32 %0, pb, pa;\n\t}"
                 : "=r"(pk), "+l"(lacc) : "r"(s0), "r"(s1), "l"(c2), "l"(negm2));
}
DEVI void pair_emul(uint32_t s0, uint32_t s1, uint64_t c2, uint64_t negm2, uint64_t magic2, uint64_t k3, uint64_t k2, uint64_t k1, uint64_t k0, uint64_t& lacc, uint32_t& pk) {
    asm volatile("{\n\t.reg .b64 sab, xab, tab, nab, fab, pab;\n\t.reg .f32 xa, xb, pa, pb;\n\t.reg .b32 ta, tb, qa, qb, ra, rb;\n\t"
                 "mov.b64 sab, {%2, %3};\n\t"
                 "fma.rn.f32x2 xab, sab, %4, %5;\n\t"
                 "mov.b64 {xa, xb}, xab;\n\t"
                 "max.f32 xa, xa, 0fC2FC0000;\n\t"
                 "max.f32 xb, xb, 0fC2FC0000;\n\t"
                 "mov.b64 xab, {xa, xb};\n\t"
                 "add.rn.f32x2 tab, xab, %6;\n\t"
                 "sub.rn.f32x2 nab, tab, %6;\n\t"
                 "sub.rn.f32x2 fab, xab, nab;\n\t"
                 "fma.rn.f32x2 pab, %7, fab, %8;\n\t"
                 "fma.rn.f32x2 pab, pab, fab, %9;\n\t"
                 "fma.rn.f32x2 pab, pab, fab, %10;\n\t"
                 "mov.b64 {ta, tb}, tab;\n\t"
                 "mov.b64 {qa, qb}, pab;\n\t"
                 "mad.lo.u32 ra, ta, 8388608, qa;\n\t"
                 "mad.lo.u32 rb, tb, 8388608, qb;\n\t"
                 "mov.b32 pa, ra;\n\t"
                 "mov.b32 pb, rb;\n\t"
                 "mov.b64 pab, {pa, pb};\n\t"
                 "add.rn.f32x2 %1, %1, pab;\n\t"
                 "cvt.rn.bf16x2.f32 %0, pb, pa;\n\t}"
                 : "=r"(pk), "+l"(lacc) : "r"(s0), "r"(s1), "l"(c2), "l"(negm2), "l"(magic2), "l"(k3), "l"(k2), "l"(k1), "l"(k0));
}
DEVI uint64_t dup2(float x) { uint64_t r; asm("mov.b64 %0, {%1, %1};" : "=l"(r) : "f"(x)); return r; }
DEVI float lo2(uint64_t v) { float a, b; asm("mov.b64 {%0, %1}, %2;" : "=f"(a), "=f"(b) : "l"(v)); return a; }
DEVI float hi2(uint64_t v) { float a, b; asm("mov.b64 {%0, %1}, %2;" : "=f"(a), "=f"(b) : "l"(v)); return b; }

#if EMUL_EVERY == 0
// exp/sum/pack over 128 elements; emulate every 0-th pair on the FMA pipe
DEVI void softmax_exp_tile(const uint32_t* s, float c, float neg_m, uint32_t* pk, float& l) {
    const uint64_t c2 = dup2(c), negm2 = dup2(neg_m), magic2 = dup2(12582912.f);
    const uint64_t k3 = dup2(0.05517025f), k2 = dup2(0.2426079f), k1 = dup2(0.69326093f), k0 = dup2(0.99992828f);
    uint64_t lacc[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) lacc[i] = 0ull;
    pair_mufu(s[0], s[1], c2, negm2, lacc[0], pk[0]);
    pair_mufu(s[2], s[3], c2, negm2, lacc[1], pk[1]);
    pair_mufu(s[4], s[5], c2, negm2, lacc[2], pk[2]);
    pair_mufu(s[6], s[7], c2, negm2, lacc[3], pk[3]);
    pair_mufu(s[8], s[9], c2, negm2, lacc[0], pk[4]);
    pair_mufu(s[10], s[11], c2, negm2, lacc[1], pk[5]);
    pair_mufu(s[12], s[13], c2, negm2, lacc[2], pk[6]);
    pair_mufu(s[14], s[15], c2, negm2, lacc[3], pk[7]);
    pair_mufu(s[16], s[17], c2, negm2, lacc[0], pk[8]);
    pair_mufu(s[18], s[19], c2, negm2, lacc[1], pk[9]);
    pair_mufu(s[20], s[21], c2, negm2, lacc[2], pk[10]);
    pair_mufu(s[22], s[23], c2, negm2, lacc[3], pk[11]);
    pair_mufu(s[24], s[25], c2, negm2, lacc[0], pk[12]);
    pair_mufu(s[26], s[27], c2, negm2, lacc[1], pk[13]);
    pair_mufu(s[28], s[29], c2, negm2, lacc[2], pk[14]);
    pair_mufu(s[30], s[31], c2, negm2, lacc[3], pk[15]);
    pair_mufu(s[32], s[33], c2, negm2, lacc[0], pk[16]);
    pair_mufu(s[34], s[35], c2, negm2, lacc[1], pk[17]);
    pair_mufu(s[36], s[37], c2, negm2, lacc[2], pk[18]);
    pair_mufu(s[38], s[39], c2, negm2, lacc[3], pk[19]);
    pair_mufu(s[40], s[41], c2, negm2, lacc[0], pk[20]);
    pair_mufu(s[42], s[43], c2, negm2, lacc[1], pk[21]);
    pair_mufu(s[44], s[45], c2, negm2, lacc[2], pk[22]);
    pair_mufu(s[46], s[47], c2, negm2, lacc[3], pk[23]);
    pair_mufu(s[48], s[49], c2, negm2, lacc[0], pk[24]);
    pair_mufu(s[50], s[51], c2, negm2, lacc[1], pk[25]);
    pair_mufu(s[52], s[53], c2, negm2, lacc[2], pk[26]);
    pair_mufu(s[54], s[55], c2, negm2, lacc[3], pk[27]);
    pair_mufu(s[56], s[57], c2, negm2, lacc[0], pk[28]);
    pair_mufu(s[58], s[59], c2, negm2, lacc[1], pk[29]);
    pair_mufu(s[60], s[61], c2, negm2, lacc[2], pk[30]);
    pair_mufu(s[62], s[63], c2, negm2, lacc[3], pk[31]);
    pair_mufu(s[64], s[65], c2, negm2, lacc[0], pk[32]);
    pair_mufu(s[66], s[67], c2, negm2, lacc[1], pk[33]);
    pair_mufu(s[68], s[69], c2, negm2, lacc[2], pk[34]);
    pair_mufu(s[70], s[71], c2, negm2, lacc[3], pk[35]);
    pair_mufu(s[72], s[73], c2, negm2, lacc[0], pk[36]);
    pair_mufu(s[74], s[75], c2, negm2, lacc[1], pk[37]);
    pair_mufu(s[76], s[77], c2, negm2, lacc[2], pk[38]);
    pair_mufu(s[78], s[79], c2, negm2, lacc[3], pk[39]);
    pair_mufu(s[80], s[81], c2, negm2, lacc[0], pk[40]);
    pair_mufu(s[82], s[83], c2, negm2, lacc[1], pk[41]);
    pair_mufu(s[84], s[85], c2, negm2, lacc[2], pk[42]);
    pair_mufu(s[86], s[87], c2, negm2, lacc[3], pk[43]);
    pair_mufu(s[88], s[89], c2, negm2, lacc[0], pk[44]);
    pair_mufu(s[90], s[91], c2, negm2, lacc[1], pk[45]);
    pair_mufu(s[92], s[93], c2, negm2, lacc[2], pk[46]);
    pair_mufu(s[94], s[95], c2, negm2, lacc[3], pk[47]);
    pair_mufu(s[96], s[97], c2, negm2, lacc[0], pk[48]);
    pair_mufu(s[98], s[99], c2, negm2, lacc[1], pk[49]);
    pair_mufu(s[100], s[101], c2, negm2, lacc[2], pk[50]);
    pair_mufu(s[102], s[103], c2, negm2, lacc[3], pk[51]);
    pair_mufu(s[104], s[105], c2, negm2, lacc[0], pk[52]);
    pair_mufu(s[106], s[107], c2, negm2, lacc[1], pk[53]);
    pair_mufu(s[108], s[109], c2, negm2, lacc[2], pk[54]);
    pair_mufu(s[110], s[111], c2, negm2, lacc[3], pk[55]);
    pair_mufu(s[112], s[113], c2, negm2, lacc[0], pk[56]);
    pair_mufu(s[114], s[115], c2, negm2, lacc[1], pk[57]);
    pair_mufu(s[116], s[117], c2, negm2, lacc[2], pk[58]);
    pair_mufu(s[118], s[119], c2, negm2, lacc[3], pk[59]);
    pair_mufu(s[120], s[121], c2, negm2, lacc[0], pk[60]);
    pair_mufu(s[122], s[123], c2, negm2, lacc[1], pk[61]);
    pair_mufu(s[124], s[125], c2, negm2, lacc[2], pk[62]);
    pair_mufu(s[126], s[127], c2, negm2, lacc[3], pk[63]);
    float tot = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; i++) tot += lo2(lacc[i]) + hi2(lacc[i]);
    l += tot;
}
#elif EMUL_EVERY == 2
// exp/sum/pack over 128 elements; emulate every 2-th pair on the FMA pipe
DEVI void softmax_exp_tile(const uint32_t* s, float c, float neg_m, uint32_t* pk, float& l) {
    const uint64_t c2 = dup2(c), negm2 = dup2(neg_m), magic2 = dup2(12582912.f);
    const uint64_t k3 = dup2(0.05517025f), k2 = dup2(0.2426079f), k1 = dup2(0.69326093f), k0 = dup2(0.99992828f);
    uint64_t lacc[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) lacc[i] = 0ull;
    pair_mufu(s[0], s[1], c2, negm2, lacc[0], pk[0]);
    pair_emul(s[2], s[3], c2, negm2, magic2, k3, k2, k1, k0, lacc[1], pk[1]);
    pair_mufu(s[4], s[5], c2, negm2, lacc[2], pk[2]);
    pair_emul(s[6], s[7], c2, negm2, magic2, k3, k2, k1, k0, lacc[3], pk[3]);
    pair_mufu(s[8], s[9], c2, negm2, lacc[0], pk[4]);
    pair_emul(s[10], s[11], c2, negm2, magic2, k3, k2, k1, k0, lacc[1], pk[5]);
    pair_mufu(s[12], s[13], c2, negm2, lacc[2], pk[6]);
    pair_emul(s[14], s[15], c2, negm2, magic2, k3, k2, k1, k0, lacc[3], pk[7]);
    pair_mufu(s[16], s[17], c2, negm2, lacc[0], pk[8]);
    pair_emul(s[18], s[19], c2, negm2, magic2, k3, k2, k1, k0, lacc[1], pk[9]);
    pair_mufu(s[20], s[21], c2, negm2, lacc[2], pk[10]);
    pair_emul(s[22], s[23], c2, negm2, magic2, k3, k2, k1, k0, lacc[3], pk[11]);
    pair_mufu(s[24], s[25], c2, negm2, lacc[0], pk[12]);
    pair_emul(s[26], s[27], c2, negm2, magic2, k3, k2, k1, k0, lacc[1], pk[13]);
    pair_mufu(s[28], s[29], c2, negm2, lacc[2], pk[14]);
    pair_emul(s[30], s[31], c2, negm2, magic2, k3, k2, k1, k0, lacc[3], pk[15]);
    pair_mufu(s[32], s[33], c2, negm2, lacc[0], pk[16]);
    pair_emul(s[34], s[35], c2, negm2, magic2, k3, k2, k1, k0, lacc[1], pk[17]);
    pair_mufu(s[36], s[37], c2, negm2, lacc[2], pk[18]);
    pair_emul(s[38], s[39], c2, negm2, magic2, k3, k2, k1, k0, lacc[3], pk[19]);
    pair_mufu(s[40], s[41], c2, negm2, lacc[0], pk[20]);
    pair_emul(s[42], s[43], c2, negm2, magic2, k3, k2, k1, k0, lacc[1], pk[21]);
    pair_mufu(s[44], s[45], c2, negm2, lacc[2], pk[22]);
    pair_emul(s[46], s[47], c2, negm2, magic2, k3, k2, k1, k0, lacc[3], pk[23]);
    pair_mufu(s[48], s[49], c2, negm2, lacc[0], pk[24]);
    pair_emul(s[50], s[51], c2, negm2, magic2, k3, k2, k1, k0, lacc[1], pk[25]);
    pair_mufu(s[52], s[53], c2, negm2, lacc[2], pk[26]);
    pair_emul(s[54], s[55], c2, negm2, magic2, k3, k2, k1, k0, lacc[3], pk[27]);
    pair_mufu(s[56], s[57], c2, negm2, lacc[0], pk[28]);
    pair_emul(s[58], s[59], c2, negm2, magic2, k3, k2, k1, k0, lacc[1], pk[29]);
    pair_mufu(s[60], s[61], c2, negm2, lacc[2], pk[30]);
    pair_emul(s[62], s[63], c2, negm2, magic2, k3, k2, k1, k0, lacc[3], pk[31]);
    pair_mufu(s[64], s[65], c2, negm2, lacc[0], pk[32]);
    pair_emul(s[66], s[67], c2, negm2, magic2, k3, k2, k1, k0, lacc[1], pk[33]);
    pair_mufu(s[68], s[69], c2, negm2, lacc[2], pk[34]);
    pair_emul(s[70], s[71], c2, negm2, magic2, k3, k2, k1, k0, lacc[3], pk[35]);
    pair_mufu(s[72], s[73], c2, negm2, lacc[0], pk[36]);
    pair_emul(s[74], s[75], c2, negm2, magic2, k3, k2, k1, k0, lacc[1], pk[37]);
    pair_mufu(s[76], s[77], c2, negm2, lacc[2], pk[38]);
    pair_emul(s[78], s[79], c2, negm2, magic2, k3, k2, k1, k0, lacc[3], pk[39]);
    pair_mufu(s[80], s[81], c2, negm2, lacc[0], pk[40]);
    pair_emul(s[82], s[83], c2, negm2, magic2, k3, k2, k1, k0, lacc[1], pk[41]);
    pair_mufu(s[84], s[85], c2, negm2, lacc[2], pk[42]);
    pair_emul(s[86], s[87], c2, negm2, magic2, k3, k2, k1, k0, lacc[3], pk[43]);
    pair_mufu(s[88], s[89], c2, negm2, lacc[0], pk[44]);
    pair_emul(s[90], s[91], c2, negm2, magic2, k3, k2, k1, k0, lacc[1], pk[45]);
    pair_mufu(s[92], s[93], c2, negm2, lacc[2], pk[46]);
    pair_emul(s[94], s[95], c2, negm2, magic2, k3, k2, k1, k0, lacc[3], pk[47]);
    pair_mufu(s[96], s[97], c2, negm2, lacc[0], pk[48]);
    pair_emul(s[98], s[99], c2, negm2, magic2, k3, k2, k1, k0, lacc[1], pk[49]);
    pair_mufu(s[100], s[101], c2, negm2, lacc[2], pk[50]);
    pair_emul(s[102], s[103], c2, negm2, magic2, k3, k2, k1, k0, lacc[3], pk[51]);
    pair_mufu(s[104], s[105], c2, negm2, lacc[0], pk[52]);
    pair_emul(s[106], s[107], c2, negm2, magic2, k3, k2, k1, k0, lacc[1], pk[53]);
    pair_mufu(s[108], s[109], c2, negm2, lacc[2], pk[54]);
    pair_emul(s[110], s[111], c2, negm2, magic2, k3, k2, k1, k0, lacc[3], pk[55]);
    pair_mufu(s[112], s[113], c2, negm2, lacc[0], pk[56]);
    pair_emul(s[114], s[115], c2, negm2, magic2, k3, k2, k1, k0, lacc[1], pk[57]);
    pair_mufu(s[116], s[117], c2, negm2, lacc[2], pk[58]);
    pair_emul(s[118], s[119], c2, negm2, magic2, k3, k2, k1, k0, lacc[3], pk[59]);
    pair_mufu(s[120], s[121], c2, negm2, lacc[0], pk[60]);
    pair_emul(s[122], s[123], c2, negm2, magic2, k3, k2, k1, k0, lacc[1], pk[61]);
    pair_mufu(s[124], s[125], c2, negm2, lacc[2], pk[62]);
    pair_emul(s[126], s[127], c2, negm2, magic2, k3, k2, k1, k0, lacc[3], pk[63]);
    float tot = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; i++) tot += lo2(lacc[i]) + hi2(lacc[i]);
    l += tot;
}
#elif EMUL_EVERY == 4
// exp/sum/pack over 128 elements; emulate every 4-th pair on the FMA pipe
DEVI void softmax_exp_tile(const uint32_t* s, float c, float neg_m, uint32_t* pk, float& l) {
    const uint64_t c2 = dup2(c), negm2 = dup2(neg_m), magic2 = dup2(12582912.f);
    const uint64_t k3 = dup2(0.05517025f), k2 = dup2(0.2426079f), k1 = dup2(0.69326093f), k0 = dup2(0.99992828f);
    uint64_t lacc[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) lacc[i] = 0ull;
    pair_mufu(s[0], s[1], c2, negm2, lacc[0], pk[0]);
    pair_mufu(s[2], s[3], c2, negm2, lacc[1], pk[1]);
    pair_mufu(s[4], s[5], c2, negm2, lacc[2], pk[2]);
    pair_emul(s[6], s[7], c2, negm2, magic2, k3, k2, k1, k0, lacc[3], pk[3]);
    pair_mufu(s[8], s[9], c2, negm2, lacc[0], pk[4]);
    pair_mufu(s[10], s[11], c2, negm2, lacc[1], pk[5]);
    pair_mufu(s[12], s[13], c2, negm2, lacc[2], pk[6]);
    pair_emul(s[14], s[15], c2, negm2, magic2, k3, k2, k1, k0, lacc[3], pk[7]);
    pair_mufu(s[16], s[17], c2, negm2, lacc[0], pk[8]);
    pair_mufu(s[18], s[19], c2, negm2, lacc[1], pk[9]);
    pair_mufu(s[20], s[21], c2, negm2, lacc[2], pk[10]);
    pair_emul(s[22], s[23], c2, negm2, magic2, k3, k2, k1, k0, lacc[3], pk[11]);
    pair_mufu(s[24], s[25], c2, negm2, lacc[0], pk[12]);
    pair_mufu(s[26], s[27], c2, negm2, lacc[1], pk[13]);
    pair_mufu(s[28], s[29], c2, negm2, lacc[2], pk[14]);
    pair_emul(s[30], s[31], c2, negm2, magic2, k3, k2, k1, k0, lacc[3], pk[15]);
    pair_mufu(s[32], s[33], c2, negm2, lacc[0], pk[16]);
    pair_mufu(s[34], s[35], c2, negm2, lacc[1], pk[17]);
    pair_mufu(s[36], s[37], c2, negm2, lacc[2], pk[18]);
    pair_emul(s[38], s[39], c2, negm2, magic2, k3, k2, k1, k0, lacc[3], pk[19]);
    pair_mufu(s[40], s[41], c2, negm2, lacc[0], pk[20]);
    pair_mufu(s[42], s[43], c2, negm2, lacc[1], pk[21]);
    pair_mufu(s[44], s[45], c2, negm2, lacc[2], pk[22]);
    pair_emul(s[46], s[47], c2, negm2, magic2, k3, k2, k1, k0, lacc[3], pk[23]);
    pair_mufu(s[48], s[49], c2, negm2, lacc[0], pk[24]);
    pair_mufu(s[50], s[51], c2, negm2, lacc[1], pk[25]);
    pair_mufu(s[52], s[53], c2, negm2, lacc[2], pk[26]);
    pair_emul(s[54], s[55], c2, negm2, magic2, k3, k2, k1, k0, lacc[3], pk[27]);
    pair_mufu(s[56], s[57], c2, negm2, lacc[0], pk[28]);
    pair_mufu(s[58], s[59], c2, negm2, lacc[1], pk[29]);
    pair_mufu(s[60], s[61], c2, negm2, lacc[2], pk[30]);
    pair_emul(s[62], s[63], c2, negm2, magic2, k3, k2, k1, k0, lacc[3], pk[31]);
    pair_mufu(s[64], s[65], c2, negm2, lacc[0], pk[32]);
    pair_mufu(s[66], s[67], c2, negm2, lacc[1], pk[33]);
    pair_mufu(s[68], s[69], c2, negm2, lacc[2], pk[34]);
    pair_emul(s[70], s[71], c2, negm2, magic2, k3, k2, k1, k0, lacc[3], pk[35]);
    pair_mufu(s[72], s[73], c2, negm2, lacc[0], pk[36]);
    pair_mufu(s[74], s[75], c2, negm2, lacc[1], pk[37]);
    pair_mufu(s[76], s[77], c2, negm2, lacc[2], pk[38]);
    pair_emul(s[78], s[79], c2, negm2, magic2, k3, k2, k1, k0, lacc[3], pk[39]);
    pair_mufu(s[80], s[81], c2, negm2, lacc[0], pk[40]);
    pair_mufu(s[82], s[83], c2, negm2, lacc[1], pk[41]);
    pair_mufu(s[84], s[85], c2, negm2, lacc[2], pk[42]);
    pair_emul(s[86], s[87], c2, negm2, magic2, k3, k2, k1, k0, lacc[3], pk[43]);
    pair_mufu(s[88], s[89], c2, negm2, lacc[0], pk[44]);
    pair_mufu(s[90], s[91], c2, negm2, lacc[1], pk[45]);
    pair_mufu(s[92], s[93], c2, negm2, lacc[2], pk[46]);
    pair_emul(s[94], s[95], c2, negm2, magic2, k3, k2, k1, k0, lacc[3], pk[47]);
    pair_mufu(s[96], s[97], c2, negm2, lacc[0], pk[48]);
    pair_mufu(s[98], s[99], c2, negm2, lacc[1], pk[49]);
    pair_mufu(s[100], s[101], c2, negm2, lacc[2], pk[50]);
    pair_emul(s[102], s[103], c2, negm2, magic2, k3, k2, k1, k0, lacc[3], pk[51]);
    pair_mufu(s[104], s[105], c2, negm2, lacc[0], pk[52]);
    pair_mufu(s[106], s[107], c2, negm2, lacc[1], pk[53]);
    pair_mufu(s[108], s[109], c2, negm2, lacc[2], pk[54]);
    pair_emul(s[110], s[111], c2, negm2, magic2, k3, k2, k1, k0, lacc[3], pk[55]);
    pair_mufu(s[112], s[113], c2, negm2, lacc[0], pk[56]);
    pair_mufu(s[114], s[115], c2, negm2, lacc[1], pk[57]);
    pair_mufu(s[116], s[117], c2, negm2, lacc[2], pk[58]);
    pair_emul(s[118], s[119], c2, negm2, magic2, k3, k2, k1, k0, lacc[3], pk[59]);
    pair_mufu(s[120], s[121], c2, negm2, lacc[0], pk[60]);
    pair_mufu(s[122], s[123], c2, negm2, lacc[1], pk[61]);
    pair_mufu(s[124], s[125], c2, negm2, lacc[2], pk[62]);
    pair_emul(s[126], s[127], c2, negm2, magic2, k3, k2, k1, k0, lacc[3], pk[63]);
    float tot = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; i++) tot += lo2(lacc[i]) + hi2(lacc[i]);
    l += tot;
}
#else
// exp/sum/pack over 128 elements; emulate every 3-th pair on the FMA pipe
DEVI void softmax_exp_tile(const uint32_t* s, float c, float neg_m, uint32_t* pk, float& l) {
    const uint64_t c2 = dup2(c), negm2 = dup2(neg_m), magic2 = dup2(12582912.f);
    const uint64_t k3 = dup2(0.05517025f), k2 = dup2(0.2426079f), k1 = dup2(0.69326093f), k0 = dup2(0.99992828f);
    uint64_t lacc[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) lacc[i] = 0ull;
    pair_mufu(s[0], s[1], c2, negm2, lacc[0], pk[0]);
    pair_mufu(s[2], s[3], c2, negm2, lacc[1], pk[1]);
    pair_emul(s[4], s[5], c2, negm2, magic2, k3, k2, k1, k0, lacc[2], pk[2]);
    pair_mufu(s[6], s[7], c2, negm2, lacc[3], pk[3]);
    pair_mufu(s[8], s[9], c2, negm2, lacc[0], pk[4]);
    pair_emul(s[10], s[11], c2, negm2, magic2, k3, k2, k1, k0, lacc[1], pk[5]);
    pair_mufu(s[12], s[13], c2, negm2, lacc[2], pk[6]);
    pair_mufu(s[14], s[15], c2, negm2, lacc[3], pk[7]);
    pair_emul(s[16], s[17], c2, negm2, magic2, k3, k2, k1, k0, lacc[0], pk[8]);
    pair_mufu(s[18], s[19], c2, negm2, lacc[1], pk[9]);
    pair_mufu(s[20], s[21], c2, negm2, lacc[2], pk[10]);
    pair_emul(s[22], s[23], c2, negm2, magic2, k3, k2, k1, k0, lacc[3], pk[11]);
    pair_mufu(s[24], s[25], c2, negm2, lacc[0], pk[12]);
    pair_mufu(s[26], s[27], c2, negm2, lacc[1], pk[13]);
    pair_emul(s[28], s[29], c2, negm2, magic2, k3, k2, k1, k0, lacc[2], pk[14]);
    pair_mufu(s[30], s[31], c2, negm2, lacc[3], pk[15]);
    pair_mufu(s[32], s[33], c2, negm2, lacc[0], pk[16]);
    pair_emul(s[34], s[35], c2, negm2, magic2, k3, k2, k1, k0, lacc[1], pk[17]);
    pair_mufu(s[36], s[37], c2, negm2, lacc[2], pk[18]);
    pair_mufu(s[38], s[39], c2, negm2, lacc[3], pk[19]);
    pair_emul(s[40], s[41], c2, negm2, magic2, k3, k2, k1, k0, lacc[0], pk[20]);
    pair_mufu(s[42], s[43], c2, negm2, lacc[1], pk[21]);
    pair_mufu(s[44], s[45], c2, negm2, lacc[2], pk[22]);
    pair_emul(s[46], s[47], c2, negm2, magic2, k3, k2, k1, k0, lacc[3], pk[23]);
    pair_mufu(s[48], s[49], c2, negm2, lacc[0], pk[24]);
    pair_mufu(s[50], s[51], c2, negm2, lacc[1], pk[25]);
    pair_emul(s[52], s[53], c2, negm2, magic2, k3, k2, k1, k0, lacc[2], pk[26]);
    pair_mufu(s[54], s[55], c2, negm2, lacc[3], pk[27]);
    pair_mufu(s[56], s[57], c2, negm2, lacc[0], pk[28]);
    pair_emul(s[58], s[59], c2, negm2, magic2, k3, k2, k1, k0, lacc[1], pk[29]);
    pair_mufu(s[60], s[61], c2, negm2, lacc[2], pk[30]);
    pair_mufu(s[62], s[63], c2, negm2, lacc[3], pk[31]);
    pair_emul(s[64], s[65], c2, negm2, magic2, k3, k2, k1, k0, lacc[0], pk[32]);
    pair_mufu(s[66], s[67], c2, negm2, lacc[1], pk[33]);
    pair_mufu(s[68], s[69], c2, negm2, lacc[2], pk[34]);
    pair_emul(s[70], s[71], c2, negm2, magic2, k3, k2, k1, k0, lacc[3], pk[35]);
    pair_mufu(s[72], s[73], c2, negm2, lacc[0], pk[36]);
    pair_mufu(s[74], s[75], c2, negm2, lacc[1], pk[37]);
    pair_emul(s[76], s[77], c2, negm2, magic2, k3, k2, k1, k0, lacc[2], pk[38]);
    pair_mufu(s[78], s[79], c2, negm2, lacc[3], pk[39]);
    pair_mufu(s[80], s[81], c2, negm2, lacc[0], pk[40]);
    pair_emul(s[82], s[83], c2, negm2, magic2, k3, k2, k1, k0, lacc[1], pk[41]);
    pair_mufu(s[84], s[85], c2, negm2, lacc[2], pk[42]);
    pair_mufu(s[86], s[87], c2, negm2, lacc[3], pk[43]);
    pair_emul(s[88], s[89], c2, negm2, magic2, k3, k2, k1, k0, lacc[0], pk[44]);
    pair_mufu(s[90], s[91], c2, negm2, lacc[1], pk[45]);
    pair_mufu(s[92], s[93], c2, negm2, lacc[2], pk[46]);
    pair_emul(s[94], s[95], c2, negm2, magic2, k3, k2, k1, k0, lacc[3], pk[47]);
    pair_mufu(s[96], s[97], c2, negm2, lacc[0], pk[48]);
    pair_mufu(s[98], s[99], c2, negm2, lacc[1], pk[49]);
    pair_emul(s[100], s[101], c2, negm2, magic2, k3, k2, k1, k0, lacc[2], pk[50]);
    pair_mufu(s[102], s[103], c2, negm2, lacc[3], pk[51]);
    pair_mufu(s[104], s[105], c2, negm2, lacc[0], pk[52]);
    pair_emul(s[106], s[107], c2, negm2, magic2, k3, k2, k1, k0, lacc[1], pk[53]);
    pair_mufu(s[108], s[109], c2, negm2, lacc[2], pk[54]);
    pair_mufu(s[110], s[111], c2, negm2, lacc[3], pk[55]);
    pair_emul(s[112], s[113], c2, negm2, magic2, k3, k2, k1, k0, lacc[0], pk[56]);
    pair_mufu(s[114], s[115], c2, negm2, lacc[1], pk[57]);
    pair_mufu(s[116], s[117], c2, negm2, lacc[2], pk[58]);
    pair_emul(s[118], s[119], c2, negm2, magic2, k3, k2, k1, k0, lacc[3], pk[59]);
    pair_mufu(s[120], s[121], c2, negm2, lacc[0], pk[60]);
    pair_mufu(s[122], s[123], c2, negm2, lacc[1], pk[61]);
    pair_emul(s[124], s[125], c2, negm2, magic2, k3, k2, k1, k0, lacc[2], pk[62]);
    pair_mufu(s[126], s[127], c2, negm2, lacc[3], pk[63]);
    float tot = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; i++) tot += lo2(lacc[i]) + hi2(lacc[i]);
    l += tot;
}
#endif

DEVI float rowmax128(const float* s) {
    float t[43];
    #pragma unroll
    for (int i = 0; i < 42; i++) t[i] = fmax3(s[3 * i], s[3 * i + 1], s[3 * i + 2]);
    t[42] = fmaxf(s[126], s[127]);
    float u[15];
    #pragma unroll
    for (int i = 0; i < 14; i++) u[i] = fmax3(t[3 * i], t[3 * i + 1], t[3 * i + 2]);
    u[14] = t[42];
    float v[5];
    #pragma unroll
    for (int i = 0; i < 5; i++) v[i] = fmax3(u[3 * i], u[3 * i + 1], u[3 * i + 2]);
    return fmaxf(fmax3(v[0], v[1], v[2]), fmaxf(v[3], v[4]));
}
