#ifndef TRONADDRESS_H
#define TRONADDRESS_H

#include <string>
#include "SECPK1/Point.h"

namespace TronAddress {

std::string PublicKeyToHexAddress(const Point &pubKey);
std::string PublicKeyToBase58Address(const Point &pubKey);

}

#endif // TRONADDRESS_H
