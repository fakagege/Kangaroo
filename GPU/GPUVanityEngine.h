#ifndef GPUVANITYENGINEH
#define GPUVANITYENGINEH

#include <stdint.h>
#include <string>
#include <vector>

#include "../SECPK1/SECP256k1.h"

#define VANITY_STATE_WORDS 12
#define VANITY_BATCH_SIZE 4
#define VANITY_THREAD_STATE_WORDS (VANITY_STATE_WORDS * VANITY_BATCH_SIZE)
#define VANITY_ADDRESS_LENGTH 34
#define VANITY_RAW_ADDRESS_LENGTH 21
#define VANITY_NB_RUN 64

typedef struct {
  uint64_t priv[4];
  unsigned char rawAddress[VANITY_RAW_ADDRESS_LENGTH];
  char address[VANITY_ADDRESS_LENGTH + 1];
} GPUVanityHit;

class GPUVanityEngine {

public:

  GPUVanityEngine(int nbThreadGroup,int nbThreadPerGroup,int gpuId,uint32_t maxFound,
                  const std::string &prefix,const std::string &suffix,int repeatTailLength);
  ~GPUVanityEngine();

  bool InitStates(Secp256K1 *secp,uint64_t seed);
  bool Search(std::vector<GPUVanityHit> &hits,uint64_t *processed);
  int GetNbThread() const;
  bool IsInitialised() const;
  const std::string &GetLastError() const;
  std::string deviceName;

private:

  int gpuId;
  int nbThread;
  int nbThreadPerGroup;
  uint32_t maxFound;
  bool initialised;
  std::string lastError;
  uint64_t *deviceStates;
  GPUVanityHit *deviceHits;
  uint32_t *deviceHitCount;
};

#endif // GPUVANITYENGINEH
