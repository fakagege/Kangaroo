#include "TronVanity.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <inttypes.h>
#include <mutex>
#include <random>
#include <stdio.h>
#include <thread>
#include <vector>

#include "Timer.h"
#include "TronAddress.h"

using namespace std;

namespace {

static const char *BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
static const size_t TRON_ADDRESS_LENGTH = 34;

typedef struct {
  Secp256K1 *secp;
  TRON_VANITY_CONFIG config;
  atomic<uint64_t> totalTried;
  atomic<uint64_t> foundCount;
  atomic<bool> stop;
  mutex outputMutex;
} TRON_VANITY_STATE;

bool isInBase58Alphabet(char c) {
  return strchr(BASE58_ALPHABET,c) != NULL;
}

bool hasRepeatedTail(const string &address,int repeatTailLength) {

  if(repeatTailLength <= 0)
    return true;
  if((size_t)repeatTailLength > address.length())
    return false;

  char repeated = address[address.length() - repeatTailLength];
  for(size_t i = address.length() - repeatTailLength; i < address.length(); i++) {
    if(address[i] != repeated)
      return false;
  }

  return true;

}

bool matches(const string &address,const TRON_VANITY_CONFIG &config) {

  if(config.prefix.length() > 0) {
    if(address.rfind(config.prefix,0) != 0)
      return false;
  }

  if(config.suffix.length() > 0) {
    if(address.length() < config.suffix.length())
      return false;
    if(address.compare(address.length() - config.suffix.length(),config.suffix.length(),config.suffix) != 0)
      return false;
  }

  return hasRepeatedTail(address,config.repeatTailLength);

}

string padHex64(const string &hex) {
  if(hex.length() >= 64)
    return hex;
  return string(64 - hex.length(),'0') + hex;
}

bool writeHit(const TRON_VANITY_CONFIG &config,const string &text) {

  if(config.outputFile.length() == 0)
    return true;

  FILE *f = fopen(config.outputFile.c_str(),"a");
  if(f == NULL) {
    printf("Cannot open %s for writing\n",config.outputFile.c_str());
    return false;
  }

  fprintf(f,"%s\n",text.c_str());
  fclose(f);
  return true;

}

void printHit(TRON_VANITY_STATE *state,uint64_t foundIndex,const string &address,const string &hexAddress,const string &privateKeyHex) {

  string pair = address + "---" + privateKeyHex;

  lock_guard<mutex> lock(state->outputMutex);
  printf("\nFOUND #%llu\n",(unsigned long long)foundIndex);
  printf("  TRON    : %s\n",address.c_str());
  printf("  TRONHEX : 0x%s\n",hexAddress.c_str());
  printf("  PRIV    : 0x%s\n",privateKeyHex.c_str());
  printf("  PAIR    : %s\n",pair.c_str());
  writeHit(state->config,pair);
}

void fillRandomPrivateKey(Int *privKey,mt19937_64 &rng,Int *order) {

  do {
    privKey->SetInt32(0);
    privKey->SetQWord(0,rng());
    privKey->SetQWord(1,rng());
    privKey->SetQWord(2,rng());
    privKey->SetQWord(3,rng());
    privKey->SetQWord(4,0);
  } while(privKey->IsZero() || privKey->IsGreaterOrEqual(order));

}

void worker(TRON_VANITY_STATE *state,int threadId) {

  uint64_t seed = ((uint64_t)Timer::getPID() << 32) ^
                  (uint64_t)chrono::high_resolution_clock::now().time_since_epoch().count() ^
                  (uint64_t)(threadId + 1) * 0x9E3779B97F4A7C15ULL;
  mt19937_64 rng(seed);
  Int privKey;

  while(!state->stop.load(memory_order_relaxed)) {
    fillRandomPrivateKey(&privKey,rng,&state->secp->order);

    Point pubKey = state->secp->ComputePublicKey(&privKey);
    string address = TronAddress::PublicKeyToBase58Address(pubKey);
    state->totalTried.fetch_add(1,memory_order_relaxed);

    if(matches(address,state->config)) {
      uint64_t foundIndex = state->foundCount.fetch_add(1,memory_order_relaxed) + 1;
      if(foundIndex <= state->config.maxFound) {
        printHit(state,foundIndex,address,TronAddress::PublicKeyToHexAddress(pubKey),padHex64(privKey.GetBase16()));
      }
      if(foundIndex >= state->config.maxFound) {
        state->stop.store(true,memory_order_relaxed);
        break;
      }
    }
  }

}

bool isConfigValid(const TRON_VANITY_CONFIG &config) {

  if(config.nbThread <= 0) {
    printf("TRON vanity: invalid thread count\n");
    return false;
  }

  if(config.maxFound == 0) {
    printf("TRON vanity: max found must be >= 1\n");
    return false;
  }

  if(config.repeatTailLength < 0 || config.repeatTailLength > (int)TRON_ADDRESS_LENGTH) {
    printf("TRON vanity: invalid repeated tail length\n");
    return false;
  }

  if(config.prefix.length() > TRON_ADDRESS_LENGTH || config.suffix.length() > TRON_ADDRESS_LENGTH) {
    printf("TRON vanity: prefix/suffix is too long for a TRON address\n");
    return false;
  }

  if(config.prefix.length() == 0 && config.suffix.length() == 0 && config.repeatTailLength == 0) {
    printf("TRON vanity: no matching rule specified\n");
    return false;
  }

  if(config.prefix.length() + (size_t)max((int)config.suffix.length(),config.repeatTailLength) > TRON_ADDRESS_LENGTH) {
    printf("TRON vanity: prefix + suffix/tail length is impossible for a 34-char TRON address\n");
    return false;
  }

  if(!TronVanity::IsValidBase58Text(config.prefix) || !TronVanity::IsValidBase58Text(config.suffix)) {
    printf("TRON vanity: prefix/suffix must use Base58 characters only\n");
    return false;
  }

  if(config.repeatTailLength > 0 && config.suffix.length() > 0) {
    size_t overlap = min((size_t)config.repeatTailLength,config.suffix.length());
    char repeated = config.suffix[config.suffix.length() - 1];
    for(size_t i = config.suffix.length() - overlap; i < config.suffix.length(); i++) {
      if(config.suffix[i] != repeated) {
        printf("TRON vanity: suffix conflicts with the repeated tail rule\n");
        return false;
      }
    }
  }

  return true;

}

}

namespace TronVanity {

bool IsValidBase58Text(const string &text) {
  for(char c : text) {
    if(!isInBase58Alphabet(c))
      return false;
  }
  return true;
}

bool Run(Secp256K1 *secp,const TRON_VANITY_CONFIG &config) {

  if(!isConfigValid(config))
    return false;

  TRON_VANITY_STATE state;
  state.secp = secp;
  state.config = config;
  state.totalTried.store(0);
  state.foundCount.store(0);
  state.stop.store(false);

  printf("TRON vanity mode\n");
  printf("Threads     : %d\n",config.nbThread);
  if(config.prefix.length() > 0)
    printf("Prefix      : %s\n",config.prefix.c_str());
  if(config.suffix.length() > 0)
    printf("Suffix      : %s\n",config.suffix.c_str());
  if(config.repeatTailLength > 0)
    printf("Repeat tail : %d same chars\n",config.repeatTailLength);
  printf("Need found  : %" PRIu64 "\n",config.maxFound);
  if(config.outputFile.length() > 0)
    printf("Output file : %s\n",config.outputFile.c_str());

  vector<thread> workers;
  workers.reserve(config.nbThread);
  for(int i = 0; i < config.nbThread; i++)
    workers.push_back(thread(worker,&state,i));

  double lastTick = Timer::get_tick();
  uint64_t lastCount = 0;
  while(!state.stop.load(memory_order_relaxed)) {
    Timer::SleepMillis(1000);
    double now = Timer::get_tick();
    uint64_t count = state.totalTried.load(memory_order_relaxed);
    double dt = now - lastTick;
    if(dt <= 0.0)
      dt = 1.0;
    double speed = (double)(count - lastCount) / dt;
    printf("\r[%.2f Addr/s][Total %" PRIu64 "][Found %" PRIu64 "]",
      speed,count,state.foundCount.load(memory_order_relaxed));
    fflush(stdout);
    lastTick = now;
    lastCount = count;
  }

  for(size_t i = 0; i < workers.size(); i++)
    workers[i].join();

  printf("\nDone. Tried %" PRIu64 " addresses, found %" PRIu64 "\n",
    state.totalTried.load(memory_order_relaxed),state.foundCount.load(memory_order_relaxed));

  return state.foundCount.load(memory_order_relaxed) > 0;

}

}
