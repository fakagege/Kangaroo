#include "TronAddress.h"

#include <stdint.h>
#include <string.h>
#include <vector>

namespace {

static inline uint32_t RotR32(uint32_t value, uint32_t shift) {
  return (value >> shift) | (value << (32 - shift));
}

static inline uint64_t RotL64(uint64_t value, uint32_t shift) {
  return shift == 0 ? value : ((value << shift) | (value >> (64 - shift)));
}

static uint32_t LoadBE32(const unsigned char *src) {
  return ((uint32_t)src[0] << 24) | ((uint32_t)src[1] << 16) |
         ((uint32_t)src[2] << 8) | (uint32_t)src[3];
}

static void StoreBE32(unsigned char *dst,uint32_t value) {
  dst[0] = (unsigned char)(value >> 24);
  dst[1] = (unsigned char)(value >> 16);
  dst[2] = (unsigned char)(value >> 8);
  dst[3] = (unsigned char)value;
}

static uint64_t LoadLE64(const unsigned char *src) {
  return ((uint64_t)src[0]) |
         ((uint64_t)src[1] << 8) |
         ((uint64_t)src[2] << 16) |
         ((uint64_t)src[3] << 24) |
         ((uint64_t)src[4] << 32) |
         ((uint64_t)src[5] << 40) |
         ((uint64_t)src[6] << 48) |
         ((uint64_t)src[7] << 56);
}

static void StoreLE64(unsigned char *dst,uint64_t value) {
  dst[0] = (unsigned char)value;
  dst[1] = (unsigned char)(value >> 8);
  dst[2] = (unsigned char)(value >> 16);
  dst[3] = (unsigned char)(value >> 24);
  dst[4] = (unsigned char)(value >> 32);
  dst[5] = (unsigned char)(value >> 40);
  dst[6] = (unsigned char)(value >> 48);
  dst[7] = (unsigned char)(value >> 56);
}

static std::string BytesToHex(const unsigned char *data,size_t size) {
  static const char *hex = "0123456789ABCDEF";
  std::string out;
  out.reserve(size * 2);
  for(size_t i = 0; i < size; i++) {
    out.push_back(hex[data[i] >> 4]);
    out.push_back(hex[data[i] & 0xF]);
  }
  return out;
}

static void Sha256(const unsigned char *data,size_t size,unsigned char hash[32]) {

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

  size_t paddedSize = size + 1 + 8;
  size_t rem = paddedSize % 64;
  if(rem) paddedSize += 64 - rem;

  std::vector<unsigned char> buffer(paddedSize,0);
  if(size > 0) memcpy(&buffer[0],data,size);
  buffer[size] = 0x80;

  uint64_t bitLength = (uint64_t)size * 8;
  for(int i = 0; i < 8; i++)
    buffer[paddedSize - 1 - i] = (unsigned char)(bitLength >> (i * 8));

  for(size_t offset = 0; offset < paddedSize; offset += 64) {
    uint32_t w[64];
    for(int i = 0; i < 16; i++)
      w[i] = LoadBE32(&buffer[offset + i * 4]);

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
  }

  for(int i = 0; i < 8; i++)
    StoreBE32(hash + i * 4,h[i]);

}

static void KeccakF(uint64_t st[25]) {

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
      uint64_t d = c[(i + 4) % 5] ^ RotL64(c[(i + 1) % 5],1);
      for(int j = 0; j < 25; j += 5)
        st[j + i] ^= d;
    }

    uint64_t current = st[1];
    for(int i = 0; i < 24; i++) {
      int j = pi[i];
      uint64_t next = st[j];
      st[j] = RotL64(current,rho[i]);
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

static void Keccak256(const unsigned char *data,size_t size,unsigned char hash[32]) {
  const size_t rate = 136;
  uint64_t st[25];
  memset(st,0,sizeof(st));

  while(size >= rate) {
    for(size_t i = 0; i < rate / 8; i++)
      st[i] ^= LoadLE64(data + i * 8);
    KeccakF(st);
    data += rate;
    size -= rate;
  }

  unsigned char block[rate];
  memset(block,0,sizeof(block));
  if(size > 0) memcpy(block,data,size);
  block[size] = 0x01;
  block[rate - 1] |= 0x80;

  for(size_t i = 0; i < rate / 8; i++)
    st[i] ^= LoadLE64(block + i * 8);
  KeccakF(st);

  for(int i = 0; i < 4; i++)
    StoreLE64(hash + i * 8,st[i]);
}

static std::string Base58Encode(const unsigned char *data,size_t size) {
  static const char *alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

  size_t zeroCount = 0;
  while(zeroCount < size && data[zeroCount] == 0)
    zeroCount++;

  std::vector<unsigned char> digits((size - zeroCount) * 138 / 100 + 1,0);
  size_t length = 0;

  for(size_t i = zeroCount; i < size; i++) {
    int carry = data[i];
    size_t j = digits.size();
    for(size_t k = 0; carry != 0 || k < length; k++) {
      j--;
      carry += 256 * digits[j];
      digits[j] = (unsigned char)(carry % 58);
      carry /= 58;
    }
    length = digits.size() - j;
  }

  size_t it = digits.size() - length;
  while(it < digits.size() && digits[it] == 0)
    it++;

  std::string out(zeroCount,'1');
  while(it < digits.size())
    out.push_back(alphabet[digits[it++]]);

  return out;
}

static void PublicKeyToRawAddress(const Point &pubKey,unsigned char address[21]) {
  Point point(pubKey);
  unsigned char pubBytes[64];
  unsigned char hash[32];

  point.x.Get32Bytes(pubBytes);
  point.y.Get32Bytes(pubBytes + 32);
  Keccak256(pubBytes,sizeof(pubBytes),hash);

  address[0] = 0x41;
  memcpy(address + 1,hash + 12,20);
}

}

namespace TronAddress {

std::string PublicKeyToHexAddress(const Point &pubKey) {
  unsigned char address[21];
  PublicKeyToRawAddress(pubKey,address);
  return BytesToHex(address,sizeof(address));
}

std::string PublicKeyToBase58Address(const Point &pubKey) {
  unsigned char payload[25];
  unsigned char hash1[32];
  unsigned char hash2[32];

  PublicKeyToRawAddress(pubKey,payload);
  Sha256(payload,21,hash1);
  Sha256(hash1,32,hash2);
  memcpy(payload + 21,hash2,4);

  return Base58Encode(payload,sizeof(payload));
}

}
