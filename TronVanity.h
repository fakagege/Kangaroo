#ifndef TRONVANITY_H
#define TRONVANITY_H

#include <stdint.h>
#include <string>
#include "SECPK1/SECP256k1.h"

typedef struct {
  int nbThread;
  int repeatTailLength;
  uint64_t maxFound;
  std::string prefix;
  std::string suffix;
  std::string outputFile;
  std::string classifyDir;
} TRON_VANITY_CONFIG;

namespace TronVanity {

bool IsValidBase58Text(const std::string &text);
bool Run(Secp256K1 *secp,const TRON_VANITY_CONFIG &config);

}

#endif // TRONVANITY_H
