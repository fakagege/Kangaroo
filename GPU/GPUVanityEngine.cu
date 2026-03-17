#include "GPUVanityEngine.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <random>
#include <string.h>

#include "../Timer.h"
#include "GPUEngine.h"
#include "GPUMath.h"

using namespace std;

namespace {

string FormatCudaError(const char *where,cudaError_t err) {
  string msg(where);
  msg += ": ";
  msg += cudaGetErrorString(err);
  return msg;
}

__device__ __constant__ uint64_t VANITY_GX[4] = {
  0x59F2815B16F81798ULL,0x029BFCDB2DCE28D9ULL,0x55A06295CE870B07ULL,0x79BE667EF9DCBBACULL
};

__device__ __constant__ uint64_t VANITY_GY[4] = {
  0x9C47D08FFB10D4B8ULL,0xFD17B448A6855419ULL,0x5DA4FBFC0E1108A8ULL,0x483ADA7726A3C465ULL
};

__device__ __constant__ uint64_t VANITY_ORDER[4] = {
  0xBFD25E8CD0364141ULL,0xBAAEDCE6AF48A03BULL,0xFFFFFFFFFFFFFFFEULL,0xFFFFFFFFFFFFFFFFULL
};

__device__ __constant__ char VANITY_PREFIX[VANITY_ADDRESS_LENGTH + 1];
__device__ __constant__ char VANITY_SUFFIX[VANITY_ADDRESS_LENGTH + 1];
__device__ __constant__ uint32_t VANITY_PREFIX_LEN;
__device__ __constant__ uint32_t VANITY_SUFFIX_LEN;
__device__ __constant__ uint32_t VANITY_REPEAT_TAIL_LEN;

__device__ __forceinline__ uint32_t RotR32(uint32_t value,uint32_t shift) {
  return (value >> shift) | (value << (32 - shift));
}

__device__ __forceinline__ uint64_t RotL64Vanity(uint64_t value,uint32_t shift) {
  return shift == 0 ? value : ((value << shift) | (value >> (64 - shift)));
}

__device__ __forceinline__ uint32_t LoadBE32Dev(const unsigned char *src) {
  return ((uint32_t)src[0] << 24) | ((uint32_t)src[1] << 16) |
         ((uint32_t)src[2] << 8) | (uint32_t)src[3];
}

__device__ __forceinline__ void StoreBE32Dev(unsigned char *dst,uint32_t value) {
  dst[0] = (unsigned char)(value >> 24);
  dst[1] = (unsigned char)(value >> 16);
  dst[2] = (unsigned char)(value >> 8);
  dst[3] = (unsigned char)value;
}

__device__ __forceinline__ uint64_t LoadLE64Dev(const unsigned char *src) {
  return ((uint64_t)src[0]) |
         ((uint64_t)src[1] << 8) |
         ((uint64_t)src[2] << 16) |
         ((uint64_t)src[3] << 24) |
         ((uint64_t)src[4] << 32) |
         ((uint64_t)src[5] << 40) |
         ((uint64_t)src[6] << 48) |
         ((uint64_t)src[7] << 56);
}

__device__ __forceinline__ void StoreLE64Dev(unsigned char *dst,uint64_t value) {
  dst[0] = (unsigned char)value;
  dst[1] = (unsigned char)(value >> 8);
  dst[2] = (unsigned char)(value >> 16);
  dst[3] = (unsigned char)(value >> 24);
  dst[4] = (unsigned char)(value >> 32);
  dst[5] = (unsigned char)(value >> 40);
  dst[6] = (unsigned char)(value >> 48);
  dst[7] = (unsigned char)(value >> 56);
}

__device__ __forceinline__ bool IsGreaterOrEqual256(const uint64_t *a,const uint64_t *b) {
  for(int i = 3; i >= 0; i--) {
    if(a[i] > b[i]) return true;
    if(a[i] < b[i]) return false;
  }
  return true;
}

__device__ __forceinline__ void AddOne256(uint64_t *a) {
  UADDO1(a[0],1ULL);
  UADDC1(a[1],0ULL);
  UADDC1(a[2],0ULL);
  UADD1(a[3],0ULL);
}

__device__ __forceinline__ void SubOrder256(uint64_t *a) {
  USUBO1(a[0],VANITY_ORDER[0]);
  USUBC1(a[1],VANITY_ORDER[1]);
  USUBC1(a[2],VANITY_ORDER[2]);
  USUB1(a[3],VANITY_ORDER[3]);
}

__device__ __forceinline__ void StoreBE256(unsigned char *dst,const uint64_t *src) {
  for(int limb = 0; limb < 4; limb++) {
    uint64_t value = src[limb];
    for(int byteIdx = 0; byteIdx < 8; byteIdx++) {
      dst[31 - (limb * 8 + byteIdx)] = (unsigned char)(value & 0xFF);
      value >>= 8;
    }
  }
}

__device__ void KeccakF(uint64_t st[25]) {
  static const int rho[24] = {
    1,3,6,10,15,21,28,36,45,55,2,14,
    27,41,56,8,25,43,62,18,39,61,20,44
  };
  static const int pi[24] = {
    10,7,11,17,18,3,5,16,8,21,24,4,
    15,23,19,13,12,2,20,14,22,9,6,1
  };
  static const uint64_t rc[24] = {
    0x0000000000000001ULL,0x0000000000008082ULL,
    0x800000000000808AULL,0x8000000080008000ULL,
    0x000000000000808BULL,0x0000000080000001ULL,
    0x8000000080008081ULL,0x8000000000008009ULL,
    0x000000000000008AULL,0x0000000000000088ULL,
    0x0000000080008009ULL,0x000000008000000AULL,
    0x000000008000808BULL,0x800000000000008BULL,
    0x8000000000008089ULL,0x8000000000008003ULL,
    0x8000000000008002ULL,0x8000000000000080ULL,
    0x000000000000800AULL,0x800000008000000AULL,
    0x8000000080008081ULL,0x8000000000008080ULL,
    0x0000000080000001ULL,0x8000000080008008ULL
  };

  for(int round = 0; round < 24; round++) {
    uint64_t c[5];
    for(int i = 0; i < 5; i++)
      c[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];

    for(int i = 0; i < 5; i++) {
      uint64_t d = c[(i + 4) % 5] ^ RotL64Vanity(c[(i + 1) % 5],1);
      for(int j = 0; j < 25; j += 5)
        st[j + i] ^= d;
    }

    uint64_t current = st[1];
    for(int i = 0; i < 24; i++) {
      int j = pi[i];
      uint64_t next = st[j];
      st[j] = RotL64Vanity(current,rho[i]);
      current = next;
    }

    for(int j = 0; j < 25; j += 5) {
      uint64_t row[5];
      for(int i = 0; i < 5; i++)
        row[i] = st[j + i];
      for(int i = 0; i < 5; i++)
        st[j + i] = row[i] ^ ((~row[(i + 1) % 5]) & row[(i + 2) % 5]);
    }

    st[0] ^= rc[round];
  }
}

__device__ void Keccak256Dev(const unsigned char *data,size_t size,unsigned char hash[32]) {
  const size_t rate = 136;
  uint64_t st[25];
  unsigned char block[136];

  for(int i = 0; i < 25; i++)
    st[i] = 0;

  for(int i = 0; i < 136; i++)
    block[i] = 0;

  for(size_t i = 0; i < size; i++)
    block[i] = data[i];

  block[size] = 0x01;
  block[rate - 1] |= 0x80;

  for(size_t i = 0; i < rate / 8; i++)
    st[i] ^= LoadLE64Dev(block + i * 8);

  KeccakF(st);

  for(int i = 0; i < 4; i++)
    StoreLE64Dev(hash + i * 8,st[i]);
}

__device__ void Sha256Dev(const unsigned char *data,size_t size,unsigned char hash[32]) {
  static const uint32_t k[64] = {
    0x428A2F98,0x71374491,0xB5C0FBCF,0xE9B5DBA5,0x3956C25B,0x59F111F1,0x923F82A4,0xAB1C5ED5,
    0xD807AA98,0x12835B01,0x243185BE,0x550C7DC3,0x72BE5D74,0x80DEB1FE,0x9BDC06A7,0xC19BF174,
    0xE49B69C1,0xEFBE4786,0x0FC19DC6,0x240CA1CC,0x2DE92C6F,0x4A7484AA,0x5CB0A9DC,0x76F988DA,
    0x983E5152,0xA831C66D,0xB00327C8,0xBF597FC7,0xC6E00BF3,0xD5A79147,0x06CA6351,0x14292967,
    0x27B70A85,0x2E1B2138,0x4D2C6DFC,0x53380D13,0x650A7354,0x766A0ABB,0x81C2C92E,0x92722C85,
    0xA2BFE8A1,0xA81A664B,0xC24B8B70,0xC76C51A3,0xD192E819,0xD6990624,0xF40E3585,0x106AA070,
    0x19A4C116,0x1E376C08,0x2748774C,0x34B0BCB5,0x391C0CB3,0x4ED8AA4A,0x5B9CCA4F,0x682E6FF3,
    0x748F82EE,0x78A5636F,0x84C87814,0x8CC70208,0x90BEFFFA,0xA4506CEB,0xBEF9A3F7,0xC67178F2
  };

  uint32_t h[8] = {
    0x6A09E667,0xBB67AE85,0x3C6EF372,0xA54FF53A,
    0x510E527F,0x9B05688C,0x1F83D9AB,0x5BE0CD19
  };

  unsigned char block[64];
  uint32_t w[64];

  for(int i = 0; i < 64; i++)
    block[i] = 0;

  for(size_t i = 0; i < size; i++)
    block[i] = data[i];

  block[size] = 0x80;
  uint64_t bitLength = (uint64_t)size * 8;
  for(int i = 0; i < 8; i++)
    block[63 - i] = (unsigned char)(bitLength >> (i * 8));

  for(int i = 0; i < 16; i++)
    w[i] = LoadBE32Dev(block + i * 4);

  for(int i = 16; i < 64; i++) {
    uint32_t s0 = RotR32(w[i - 15],7) ^ RotR32(w[i - 15],18) ^ (w[i - 15] >> 3);
    uint32_t s1 = RotR32(w[i - 2],17) ^ RotR32(w[i - 2],19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }

  uint32_t a = h[0];
  uint32_t b = h[1];
  uint32_t c = h[2];
  uint32_t d = h[3];
  uint32_t e = h[4];
  uint32_t f = h[5];
  uint32_t g = h[6];
  uint32_t hh = h[7];

  for(int i = 0; i < 64; i++) {
    uint32_t s1 = RotR32(e,6) ^ RotR32(e,11) ^ RotR32(e,25);
    uint32_t ch = (e & f) ^ ((~e) & g);
    uint32_t temp1 = hh + s1 + ch + k[i] + w[i];
    uint32_t s0 = RotR32(a,2) ^ RotR32(a,13) ^ RotR32(a,22);
    uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    uint32_t temp2 = s0 + maj;

    hh = g;
    g = f;
    f = e;
    e = d + temp1;
    d = c;
    c = b;
    b = a;
    a = temp1 + temp2;
  }

  h[0] += a;
  h[1] += b;
  h[2] += c;
  h[3] += d;
  h[4] += e;
  h[5] += f;
  h[6] += g;
  h[7] += hh;

  for(int i = 0; i < 8; i++)
    StoreBE32Dev(hash + i * 4,h[i]);
}

__device__ void Base58Encode25(const unsigned char payload[25],char out[VANITY_ADDRESS_LENGTH + 1]) {
  static const char *alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
  unsigned char digits[40];
  for(int i = 0; i < 40; i++)
    digits[i] = 0;

  int length = 0;
  for(int i = 0; i < 25; i++) {
    int carry = payload[i];
    int j = 39;
    int k = 0;
    while(carry != 0 || k < length) {
      carry += 256 * digits[j];
      digits[j] = (unsigned char)(carry % 58);
      carry /= 58;
      j--;
      k++;
    }
    length = k;
  }

  int start = 40 - length;
  int outPos = 0;
  while(start < 40 && digits[start] == 0)
    start++;

  for(int i = start; i < 40 && outPos < VANITY_ADDRESS_LENGTH; i++)
    out[outPos++] = alphabet[digits[i]];

  out[outPos] = 0;
}

__device__ void PublicKeyToTronAddress(const uint64_t *px,const uint64_t *py,unsigned char rawAddress[21],char address[VANITY_ADDRESS_LENGTH + 1]) {
  unsigned char pubKey[64];
  unsigned char keccak[32];
  unsigned char hash1[32];
  unsigned char hash2[32];
  unsigned char payload[25];

  StoreBE256(pubKey,px);
  StoreBE256(pubKey + 32,py);
  Keccak256Dev(pubKey,64,keccak);

  rawAddress[0] = 0x41;
  for(int i = 0; i < 20; i++)
    rawAddress[i + 1] = keccak[i + 12];

  for(int i = 0; i < 21; i++)
    payload[i] = rawAddress[i];
  Sha256Dev(payload,21,hash1);
  Sha256Dev(hash1,32,hash2);
  for(int i = 0; i < 4; i++)
    payload[21 + i] = hash2[i];

  Base58Encode25(payload,address);
}

__device__ bool MatchesAddress(const char address[VANITY_ADDRESS_LENGTH + 1]) {
  if(VANITY_PREFIX_LEN > 0) {
    for(uint32_t i = 0; i < VANITY_PREFIX_LEN; i++) {
      if(address[i] != VANITY_PREFIX[i])
        return false;
    }
  }

  if(VANITY_SUFFIX_LEN > 0) {
    for(uint32_t i = 0; i < VANITY_SUFFIX_LEN; i++) {
      if(address[VANITY_ADDRESS_LENGTH - VANITY_SUFFIX_LEN + i] != VANITY_SUFFIX[i])
        return false;
    }
  }

  if(VANITY_REPEAT_TAIL_LEN > 0) {
    char repeated = address[VANITY_ADDRESS_LENGTH - VANITY_REPEAT_TAIL_LEN];
    for(uint32_t i = VANITY_ADDRESS_LENGTH - VANITY_REPEAT_TAIL_LEN; i < VANITY_ADDRESS_LENGTH; i++) {
      if(address[i] != repeated)
        return false;
    }
  }

  return true;
}

__device__ void InvertBatch(uint64_t values[VANITY_BATCH_SIZE][4]) {
  uint64_t prefix[VANITY_BATCH_SIZE][4];
  uint64_t inverse[5];
  uint64_t newValue[4];

  Load256(prefix[0],values[0]);
  for(int i = 1; i < VANITY_BATCH_SIZE; i++)
    _ModMult(prefix[i],prefix[i - 1],values[i]);

  Load256(inverse,prefix[VANITY_BATCH_SIZE - 1]);
  inverse[4] = 0;
  _ModInv(inverse);

  for(int i = VANITY_BATCH_SIZE - 1; i > 0; i--) {
    _ModMult(newValue,prefix[i - 1],inverse);
    _ModMult(inverse,values[i]);
    Load256(values[i],newValue);
  }

  Load256(values[0],inverse);
}

__device__ void PointAddGeneratorWithInv(uint64_t px[4],uint64_t py[4],uint64_t inv[4]) {
  uint64_t dy[4];
  uint64_t s[4];
  uint64_t p[4];
  uint64_t rx[4];
  uint64_t ry[4];

  ModSub256(dy,py,(uint64_t *)VANITY_GY);
  _ModMult(s,dy,inv);
  _ModSqr(p,s);

  ModSub256(rx,p,(uint64_t *)VANITY_GX);
  ModSub256(rx,px);

  ModSub256(ry,px,rx);
  _ModMult(ry,s);
  ModSub256(ry,py);

  Load256(px,rx);
  Load256(py,ry);
}

__device__ void LoadState(uint64_t *states,uint64_t idx,uint64_t px[VANITY_BATCH_SIZE][4],
                          uint64_t py[VANITY_BATCH_SIZE][4],uint64_t priv[VANITY_BATCH_SIZE][4]) {
  uint64_t *src = states + idx * VANITY_THREAD_STATE_WORDS;
  for(int batch = 0; batch < VANITY_BATCH_SIZE; batch++) {
    uint64_t *batchSrc = src + batch * VANITY_STATE_WORDS;
    for(int i = 0; i < 4; i++) {
      px[batch][i] = batchSrc[i];
      py[batch][i] = batchSrc[4 + i];
      priv[batch][i] = batchSrc[8 + i];
    }
  }
}

__device__ void StoreState(uint64_t *states,uint64_t idx,const uint64_t px[VANITY_BATCH_SIZE][4],
                           const uint64_t py[VANITY_BATCH_SIZE][4],const uint64_t priv[VANITY_BATCH_SIZE][4]) {
  uint64_t *dst = states + idx * VANITY_THREAD_STATE_WORDS;
  for(int batch = 0; batch < VANITY_BATCH_SIZE; batch++) {
    uint64_t *batchDst = dst + batch * VANITY_STATE_WORDS;
    for(int i = 0; i < 4; i++) {
      batchDst[i] = px[batch][i];
      batchDst[4 + i] = py[batch][i];
      batchDst[8 + i] = priv[batch][i];
    }
  }
}

__global__ void search_vanity(uint64_t *states,uint32_t maxFound,uint32_t *hitCount,GPUVanityHit *hits) {
  uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t px[VANITY_BATCH_SIZE][4];
  uint64_t py[VANITY_BATCH_SIZE][4];
  uint64_t priv[VANITY_BATCH_SIZE][4];
  uint64_t dxInv[VANITY_BATCH_SIZE][4];
  unsigned char rawAddress[21];
  char address[VANITY_ADDRESS_LENGTH + 1];

  LoadState(states,idx,px,py,priv);

  for(int run = 0; run < VANITY_NB_RUN; run++) {
    for(int batch = 0; batch < VANITY_BATCH_SIZE; batch++) {
      PublicKeyToTronAddress(px[batch],py[batch],rawAddress,address);

      if(MatchesAddress(address)) {
        uint32_t pos = atomicAdd(hitCount,1);
        if(pos < maxFound) {
          for(int i = 0; i < 4; i++)
            hits[pos].priv[i] = priv[batch][i];
          for(int i = 0; i < 21; i++)
            hits[pos].rawAddress[i] = rawAddress[i];
          for(int i = 0; i <= VANITY_ADDRESS_LENGTH; i++)
            hits[pos].address[i] = address[i];
        }
      }
    }

    for(int batch = 0; batch < VANITY_BATCH_SIZE; batch++)
      ModSub256(dxInv[batch],px[batch],(uint64_t *)VANITY_GX);
    InvertBatch(dxInv);

    for(int batch = 0; batch < VANITY_BATCH_SIZE; batch++) {
      PointAddGeneratorWithInv(px[batch],py[batch],dxInv[batch]);
      AddOne256(priv[batch]);
      if(IsGreaterOrEqual256(priv[batch],VANITY_ORDER))
        SubOrder256(priv[batch]);
    }
  }

  StoreState(states,idx,px,py,priv);
}

}

GPUVanityEngine::GPUVanityEngine(int nbThreadGroup,int nbThreadPerGroup,int gpuId,uint32_t maxFound,
                                 const std::string &prefix,const std::string &suffix,int repeatTailLength) {

  this->gpuId = gpuId;
  this->nbThreadPerGroup = nbThreadPerGroup;
  this->nbThread = nbThreadGroup * nbThreadPerGroup;
  this->maxFound = maxFound;
  this->initialised = false;
  this->lastError = "uninitialized";
  this->deviceStates = NULL;
  this->deviceHits = NULL;
  this->deviceHitCount = NULL;

  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if(err != cudaSuccess) {
    lastError = FormatCudaError("cudaGetDeviceCount",err);
    return;
  }
  if(deviceCount == 0) {
    lastError = "No CUDA device found";
    return;
  }
  if(gpuId >= deviceCount) {
    lastError = "Requested gpuId is out of range";
    return;
  }

  err = cudaSetDevice(gpuId);
  if(err != cudaSuccess) {
    lastError = FormatCudaError("cudaSetDevice",err);
    return;
  }

  cudaDeviceProp prop;
  err = cudaGetDeviceProperties(&prop,gpuId);
  if(err != cudaSuccess) {
    lastError = FormatCudaError("cudaGetDeviceProperties",err);
    return;
  }

  char tmp[256];
  sprintf(tmp,"GPU #%d %s Grid(%dx%d) Batch(%d)",gpuId,prop.name,nbThreadGroup,nbThreadPerGroup,VANITY_BATCH_SIZE);
  deviceName = string(tmp);

  err = cudaMalloc((void **)&deviceStates,(size_t)nbThread * VANITY_THREAD_STATE_WORDS * sizeof(uint64_t));
  if(err != cudaSuccess) {
    lastError = FormatCudaError("cudaMalloc(deviceStates)",err);
    return;
  }

  err = cudaMalloc((void **)&deviceHits,(size_t)maxFound * sizeof(GPUVanityHit));
  if(err != cudaSuccess) {
    lastError = FormatCudaError("cudaMalloc(deviceHits)",err);
    return;
  }

  err = cudaMalloc((void **)&deviceHitCount,sizeof(uint32_t));
  if(err != cudaSuccess) {
    lastError = FormatCudaError("cudaMalloc(deviceHitCount)",err);
    return;
  }

  char prefixBuff[VANITY_ADDRESS_LENGTH + 1];
  char suffixBuff[VANITY_ADDRESS_LENGTH + 1];
  memset(prefixBuff,0,sizeof(prefixBuff));
  memset(suffixBuff,0,sizeof(suffixBuff));
  strncpy(prefixBuff,prefix.c_str(),VANITY_ADDRESS_LENGTH);
  strncpy(suffixBuff,suffix.c_str(),VANITY_ADDRESS_LENGTH);
  uint32_t prefixLen = (uint32_t)prefix.length();
  uint32_t suffixLen = (uint32_t)suffix.length();
  uint32_t repeatTail = (uint32_t)repeatTailLength;

  err = cudaMemcpyToSymbol(VANITY_PREFIX,prefixBuff,sizeof(prefixBuff));
  if(err != cudaSuccess) {
    lastError = FormatCudaError("cudaMemcpyToSymbol(VANITY_PREFIX)",err);
    return;
  }
  err = cudaMemcpyToSymbol(VANITY_SUFFIX,suffixBuff,sizeof(suffixBuff));
  if(err != cudaSuccess) {
    lastError = FormatCudaError("cudaMemcpyToSymbol(VANITY_SUFFIX)",err);
    return;
  }
  err = cudaMemcpyToSymbol(VANITY_PREFIX_LEN,&prefixLen,sizeof(prefixLen));
  if(err != cudaSuccess) {
    lastError = FormatCudaError("cudaMemcpyToSymbol(VANITY_PREFIX_LEN)",err);
    return;
  }
  err = cudaMemcpyToSymbol(VANITY_SUFFIX_LEN,&suffixLen,sizeof(suffixLen));
  if(err != cudaSuccess) {
    lastError = FormatCudaError("cudaMemcpyToSymbol(VANITY_SUFFIX_LEN)",err);
    return;
  }
  err = cudaMemcpyToSymbol(VANITY_REPEAT_TAIL_LEN,&repeatTail,sizeof(repeatTail));
  if(err != cudaSuccess) {
    lastError = FormatCudaError("cudaMemcpyToSymbol(VANITY_REPEAT_TAIL_LEN)",err);
    return;
  }

  initialised = true;
  lastError.clear();
}

GPUVanityEngine::~GPUVanityEngine() {
  if(deviceStates) cudaFree(deviceStates);
  if(deviceHits) cudaFree(deviceHits);
  if(deviceHitCount) cudaFree(deviceHitCount);
}

bool GPUVanityEngine::InitStates(Secp256K1 *secp,uint64_t seed) {

  if(!initialised)
    return false;

  cudaError_t err = cudaSetDevice(gpuId);
  if(err != cudaSuccess) {
    lastError = FormatCudaError("cudaSetDevice",err);
    return false;
  }

  mt19937_64 rng(seed);
  int totalStates = nbThread * VANITY_BATCH_SIZE;
  vector<Int> privKeys(totalStates);
  for(int i = 0; i < totalStates; i++) {
    do {
      privKeys[i].SetInt32(0);
      privKeys[i].SetQWord(0,rng());
      privKeys[i].SetQWord(1,rng());
      privKeys[i].SetQWord(2,rng());
      privKeys[i].SetQWord(3,rng());
      privKeys[i].SetQWord(4,0);
    } while(privKeys[i].IsZero() || privKeys[i].IsGreaterOrEqual(&secp->order) ||
            (privKeys[i].bits64[0] == 1ULL && privKeys[i].bits64[1] == 0ULL &&
             privKeys[i].bits64[2] == 0ULL && privKeys[i].bits64[3] == 0ULL));
  }

  vector<Point> pubKeys = secp->ComputePublicKeys(privKeys);
  vector<uint64_t> hostStates((size_t)nbThread * VANITY_THREAD_STATE_WORDS);
  for(int i = 0; i < nbThread; i++) {
    uint64_t *dst = &hostStates[(size_t)i * VANITY_THREAD_STATE_WORDS];
    for(int batch = 0; batch < VANITY_BATCH_SIZE; batch++) {
      int stateIdx = i * VANITY_BATCH_SIZE + batch;
      uint64_t *batchDst = dst + batch * VANITY_STATE_WORDS;
      for(int j = 0; j < 4; j++) {
        batchDst[j] = pubKeys[stateIdx].x.bits64[j];
        batchDst[4 + j] = pubKeys[stateIdx].y.bits64[j];
        batchDst[8 + j] = privKeys[stateIdx].bits64[j];
      }
    }
  }

  err = cudaMemcpy(deviceStates,&hostStates[0],hostStates.size() * sizeof(uint64_t),cudaMemcpyHostToDevice);
  if(err != cudaSuccess)
    lastError = FormatCudaError("cudaMemcpy(deviceStates)",err);
  return err == cudaSuccess;
}

bool GPUVanityEngine::Search(std::vector<GPUVanityHit> &hits,uint64_t *processed) {

  hits.clear();
  if(processed) *processed = 0;
  if(!initialised)
    return false;

  cudaError_t err = cudaSetDevice(gpuId);
  if(err != cudaSuccess) {
    lastError = FormatCudaError("cudaSetDevice",err);
    return false;
  }

  err = cudaMemset(deviceHitCount,0,sizeof(uint32_t));
  if(err != cudaSuccess) {
    lastError = FormatCudaError("cudaMemset(deviceHitCount)",err);
    return false;
  }

  search_vanity<<<nbThread / nbThreadPerGroup,nbThreadPerGroup>>>(deviceStates,maxFound,deviceHitCount,deviceHits);
  err = cudaDeviceSynchronize();
  if(err != cudaSuccess) {
    lastError = FormatCudaError("search_vanity kernel",err);
    return false;
  }

  uint32_t count = 0;
  err = cudaMemcpy(&count,deviceHitCount,sizeof(uint32_t),cudaMemcpyDeviceToHost);
  if(err != cudaSuccess) {
    lastError = FormatCudaError("cudaMemcpy(hitCount)",err);
    return false;
  }

  if(count > maxFound)
    count = maxFound;
  if(count > 0) {
    hits.resize(count);
    err = cudaMemcpy(&hits[0],deviceHits,(size_t)count * sizeof(GPUVanityHit),cudaMemcpyDeviceToHost);
    if(err != cudaSuccess) {
      lastError = FormatCudaError("cudaMemcpy(hits)",err);
      return false;
    }
  }

  if(processed)
    *processed = (uint64_t)nbThread * (uint64_t)VANITY_BATCH_SIZE * (uint64_t)VANITY_NB_RUN;

  return true;
}

int GPUVanityEngine::GetNbThread() const {
  return nbThread;
}

bool GPUVanityEngine::IsInitialised() const {
  return initialised;
}

const std::string &GPUVanityEngine::GetLastError() const {
  return lastError;
}
