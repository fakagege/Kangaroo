#include "TronVanity.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstring>
#include <errno.h>
#include <inttypes.h>
#include <mutex>
#include <random>
#include <set>
#include <stdio.h>
#include <thread>
#include <vector>
#include <sys/stat.h>

#ifdef WIN64
#include <direct.h>
#endif

#include "Timer.h"
#include "TronAddress.h"

#ifdef WITHGPU
#include <cuda_runtime.h>
#include "GPU/GPUEngine.h"
#include "GPU/GPUVanityEngine.h"
#endif

using namespace std;

namespace {

static const char *BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
static const size_t TRON_ADDRESS_LENGTH = 34;
static const char *DEFAULT_CLASSIFY_DIR = "tron_hits";

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

string bytesToHex(const unsigned char *data,size_t size) {
  static const char *hex = "0123456789ABCDEF";
  string out;
  out.reserve(size * 2);
  for(size_t i = 0; i < size; i++) {
    out.push_back(hex[data[i] >> 4]);
    out.push_back(hex[data[i] & 0x0F]);
  }
  return out;
}

string privWordsToHex(const uint64_t priv[4]) {
  Int value;
  value.SetInt32(0);
  for(int i = 0; i < 4; i++)
    value.bits64[i] = priv[i];
  value.bits64[4] = 0;
  return padHex64(value.GetBase16());
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

bool isAllDigits(const string &text) {
  for(char c : text) {
    if(!isdigit((unsigned char)c))
      return false;
  }
  return text.length() > 0;
}

bool isAllUpper(const string &text) {
  for(char c : text) {
    if(!isupper((unsigned char)c))
      return false;
  }
  return text.length() > 0;
}

bool isAllLower(const string &text) {
  for(char c : text) {
    if(!islower((unsigned char)c))
      return false;
  }
  return text.length() > 0;
}

bool ensureDir(const string &dir) {
  if(dir.length() == 0)
    return true;

  struct stat st;
  if(stat(dir.c_str(),&st) == 0)
    return (st.st_mode & S_IFDIR) != 0;

#ifdef WIN64
  int rc = _mkdir(dir.c_str());
#else
  int rc = mkdir(dir.c_str(),0755);
#endif
  return rc == 0 || errno == EEXIST;
}

bool appendLine(const string &filePath,const string &text) {
  FILE *f = fopen(filePath.c_str(),"a");
  if(f == NULL)
    return false;
  fprintf(f,"%s\n",text.c_str());
  fclose(f);
  return true;
}

string joinPath(const string &dir,const string &file) {
  if(dir.length() == 0)
    return file;
  char last = dir[dir.length() - 1];
  if(last == '/' || last == '\\')
    return dir + file;
  return dir + "/" + file;
}

vector<string> getClassificationLabels(const string &address,int repeatTailLength) {
  vector<string> labels;
  if(repeatTailLength <= 0 || !hasRepeatedTail(address,repeatTailLength))
    return labels;

  string tail = address.substr(address.length() - repeatTailLength);
  labels.push_back(tail);
  labels.push_back(to_string(repeatTailLength) + "nn");

  if(isAllDigits(tail))
    labels.push_back(to_string(repeatTailLength) + "n");
  if(isAllUpper(tail))
    labels.push_back(to_string(repeatTailLength) + "A");
  if(isAllLower(tail)) {
    labels.push_back(to_string(repeatTailLength) + "d");
    labels.push_back(to_string(repeatTailLength) + "a");
  }

  return labels;
}

void writeClassifiedHit(const TRON_VANITY_CONFIG &config,const string &pair,const vector<string> &labels) {
  if(config.repeatTailLength <= 0)
    return;

  string classifyDir = config.classifyDir.length() > 0 ? config.classifyDir : DEFAULT_CLASSIFY_DIR;
  if(!ensureDir(classifyDir)) {
    printf("Cannot create classify directory %s\n",classifyDir.c_str());
    return;
  }

  appendLine(joinPath(classifyDir,"all.txt"),pair);

  set<string> uniqueFiles;
  for(size_t i = 0; i < labels.size(); i++)
    uniqueFiles.insert(labels[i] + ".txt");

  for(set<string>::iterator it = uniqueFiles.begin(); it != uniqueFiles.end(); ++it)
    appendLine(joinPath(classifyDir,*it),pair);
}

void printConfigBanner(const TRON_VANITY_CONFIG &config,const char *modeName) {
  printf("TRON vanity mode (%s)\n",modeName);
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
  if(config.repeatTailLength > 0) {
    string classifyDir = config.classifyDir.length() > 0 ? config.classifyDir : DEFAULT_CLASSIFY_DIR;
    printf("Classify dir: %s\n",classifyDir.c_str());
  }
}

void printHit(TRON_VANITY_STATE *state,uint64_t foundIndex,const string &address,const string &hexAddress,const string &privateKeyHex) {

  string pair = address + "---" + privateKeyHex;
  vector<string> labels = getClassificationLabels(address,state->config.repeatTailLength);

  lock_guard<mutex> lock(state->outputMutex);
  printf("\nFOUND #%llu\n",(unsigned long long)foundIndex);
  printf("  TRON    : %s\n",address.c_str());
  printf("  TRONHEX : 0x%s\n",hexAddress.c_str());
  printf("  PRIV    : 0x%s\n",privateKeyHex.c_str());
  printf("  PAIR    : %s\n",pair.c_str());
  if(labels.size() > 0) {
    printf("  CLASS   : ");
    for(size_t i = 0; i < labels.size(); i++) {
      if(i > 0)
        printf(", ");
      printf("%s",labels[i].c_str());
    }
    printf("\n");
  }
  writeHit(state->config,pair);
  writeClassifiedHit(state->config,pair,labels);
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

void cpuWorker(TRON_VANITY_STATE *state,int threadId) {

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

bool runCPU(TRON_VANITY_STATE *state) {

  printConfigBanner(state->config,"CPU");

  vector<thread> workers;
  workers.reserve(state->config.nbThread);
  for(int i = 0; i < state->config.nbThread; i++)
    workers.push_back(thread(cpuWorker,state,i));

  double lastTick = Timer::get_tick();
  uint64_t lastCount = 0;
  while(!state->stop.load(memory_order_relaxed)) {
    Timer::SleepMillis(1000);
    double now = Timer::get_tick();
    uint64_t count = state->totalTried.load(memory_order_relaxed);
    double dt = now - lastTick;
    if(dt <= 0.0)
      dt = 1.0;
    double speed = (double)(count - lastCount) / dt;
    printf("\r[%.2f Addr/s][Total %" PRIu64 "][Found %" PRIu64 "]",
      speed,count,state->foundCount.load(memory_order_relaxed));
    fflush(stdout);
    lastTick = now;
    lastCount = count;
  }

  for(size_t i = 0; i < workers.size(); i++)
    workers[i].join();

  printf("\nDone. Tried %" PRIu64 " addresses, found %" PRIu64 "\n",
    state->totalTried.load(memory_order_relaxed),state->foundCount.load(memory_order_relaxed));

  return state->foundCount.load(memory_order_relaxed) > 0;

}

#ifdef WITHGPU

int getCudaDeviceCount() {
  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if(err != cudaSuccess)
    return 0;
  return deviceCount;
}

bool resolveGpuConfig(const TRON_VANITY_CONFIG &config,vector<int> &deviceIds,vector<int> &grids) {

  int deviceCount = getCudaDeviceCount();
  if(deviceCount <= 0)
    return false;

  if(config.gpuIdsProvided) {
    deviceIds = config.gpuIds;
  } else {
    deviceIds.clear();
    for(int i = 0; i < deviceCount; i++)
      deviceIds.push_back(i);
  }

  grids.clear();
  if(config.gridSize.size() == (deviceIds.size() * 2)) {
    grids = config.gridSize;
  } else {
    for(size_t i = 0; i < deviceIds.size(); i++) {
      int gx = 0;
      int gy = 0;
      if(!GPUEngine::GetGridSize(deviceIds[i],&gx,&gy))
        return false;
      grids.push_back(gx);
      grids.push_back(gy);
    }
  }

  return deviceIds.size() > 0;

}

bool runGPU(Secp256K1 *secp,TRON_VANITY_STATE *state) {

  vector<int> deviceIds;
  vector<int> grids;
  if(!resolveGpuConfig(state->config,deviceIds,grids))
    return false;

  atomic<int> activeWorkers((int)deviceIds.size());
  atomic<int> readyWorkers(0);
  vector<thread> workers;
  workers.reserve(deviceIds.size());

  printConfigBanner(state->config,"GPU");

  for(size_t i = 0; i < deviceIds.size(); i++) {
    int deviceId = deviceIds[i];
    int gx = grids[i * 2];
    int gy = grids[i * 2 + 1];
    workers.push_back(thread([&, deviceId, gx, gy, i]() {
      uint64_t seed = ((uint64_t)Timer::getPID() << 32) ^
                      (uint64_t)chrono::high_resolution_clock::now().time_since_epoch().count() ^
                      (uint64_t)(deviceId + 1) * 0xD2B74407B1CE6E93ULL;

      GPUVanityEngine engine(gx,gy,deviceId,(uint32_t)min<uint64_t>(state->config.maxFound,65535),
                             state->config.prefix,state->config.suffix,state->config.repeatTailLength);
      if(!engine.IsInitialised() || !engine.InitStates(secp,seed)) {
        lock_guard<mutex> lock(state->outputMutex);
        printf("GPU init failed on device %d, worker disabled\n",deviceId);
        activeWorkers.fetch_sub(1,memory_order_relaxed);
        if(activeWorkers.load(memory_order_relaxed) == 0)
          state->stop.store(true,memory_order_relaxed);
        return;
      }

      {
        lock_guard<mutex> lock(state->outputMutex);
        printf("GPU        : %s\n",engine.deviceName.c_str());
      }

      readyWorkers.fetch_add(1,memory_order_relaxed);

      while(!state->stop.load(memory_order_relaxed)) {
        vector<GPUVanityHit> hits;
        uint64_t processed = 0;
        if(!engine.Search(hits,&processed)) {
          lock_guard<mutex> lock(state->outputMutex);
          printf("GPU search failed on device %d, worker stopped\n",deviceId);
          break;
        }

        state->totalTried.fetch_add(processed,memory_order_relaxed);

        for(size_t hitIdx = 0; hitIdx < hits.size(); hitIdx++) {
          uint64_t foundIndex = state->foundCount.fetch_add(1,memory_order_relaxed) + 1;
          if(foundIndex <= state->config.maxFound) {
            printHit(state,foundIndex,string(hits[hitIdx].address),
                     bytesToHex(hits[hitIdx].rawAddress,VANITY_RAW_ADDRESS_LENGTH),
                     privWordsToHex(hits[hitIdx].priv));
          }
          if(foundIndex >= state->config.maxFound) {
            state->stop.store(true,memory_order_relaxed);
            break;
          }
        }
      }

      activeWorkers.fetch_sub(1,memory_order_relaxed);
      if(activeWorkers.load(memory_order_relaxed) == 0)
        state->stop.store(true,memory_order_relaxed);
    }));
  }

  double lastTick = Timer::get_tick();
  uint64_t lastCount = 0;
  while(!state->stop.load(memory_order_relaxed)) {
    Timer::SleepMillis(1000);
    double now = Timer::get_tick();
    uint64_t count = state->totalTried.load(memory_order_relaxed);
    double dt = now - lastTick;
    if(dt <= 0.0)
      dt = 1.0;
    double speed = (double)(count - lastCount) / dt;
    printf("\r[%.2f Addr/s][Total %" PRIu64 "][Found %" PRIu64 "][GPU %d/%d]",
      speed,count,state->foundCount.load(memory_order_relaxed),
      readyWorkers.load(memory_order_relaxed),(int)deviceIds.size());
    fflush(stdout);
    lastTick = now;
    lastCount = count;
  }

  for(size_t i = 0; i < workers.size(); i++)
    workers[i].join();

  printf("\nDone. Tried %" PRIu64 " addresses, found %" PRIu64 "\n",
    state->totalTried.load(memory_order_relaxed),state->foundCount.load(memory_order_relaxed));

  return readyWorkers.load(memory_order_relaxed) > 0 &&
         state->foundCount.load(memory_order_relaxed) > 0;

}

#endif

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

#ifdef WITHGPU
  if(runGPU(secp,&state))
    return true;

  state.totalTried.store(0);
  state.foundCount.store(0);
  state.stop.store(false);
  printf("Falling back to CPU search\n");
#endif

  return runCPU(&state);

}

}
