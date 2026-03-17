/*
 * This file is part of the BSGS distribution (https://github.com/JeanLucPons/Kangaroo).
 * Copyright (c) 2020 Jean Luc PONS.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Kangaroo.h"
#include "Timer.h"
#include "SECPK1/SECP256k1.h"
#include "GPU/GPUEngine.h"
#include "TronAddress.h"
#include "TronVanity.h"
#include <fstream>
#include <string>
#include <string.h>
#include <stdexcept>
#include <cctype>
#include <stdint.h>

using namespace std;

#define CHECKARG(opt,n) if(a>=argc-1) {::printf(opt " missing argument #%d\n",n);exit(0);} else {a++;}

// ------------------------------------------------------------------------------------------

void printUsage() {

  printf("Kangaroo [-v] [-t nbThread] [-d dpBit] [gpu] [-check]\n");
  printf("         [-gpuId gpuId1[,gpuId2,...]] [-g g1x,g1y[,g2x,g2y,...]]\n");
  printf("         inFile\n");
  printf(" -v: Print version\n");
  printf(" -gpu: Enable gpu calculation\n");
  printf(" -gpuId gpuId1,gpuId2,...: List of GPU(s) to use, default is 0\n");
  printf(" -g g1x,g1y,g2x,g2y,...: Specify GPU(s) kernel gridsize, default is 2*(MP),2*(Core/MP)\n");
  printf(" -d: Specify number of leading zeros for the DP method (default is auto)\n");
  printf(" -t nbThread: Secify number of thread\n");
  printf(" -w workfile: Specify file to save work into (current processed key only)\n");
  printf(" -i workfile: Specify file to load work from (current processed key only)\n");
  printf(" -wi workInterval: Periodic interval (in seconds) for saving work\n");
  printf(" -ws: Save kangaroos in the work file\n");
  printf(" -wss: Save kangaroos via the server\n");
  printf(" -wsplit: Split work file of server and reset hashtable\n");
  printf(" -wm file1 file2 destfile: Merge work file\n");
  printf(" -wmdir dir destfile: Merge directory of work files\n");
  printf(" -wt timeout: Save work timeout in millisec (default is 3000ms)\n");
  printf(" -winfo file1: Work file info file\n");
  printf(" -wpartcreate name: Create empty partitioned work file (name is a directory)\n");
  printf(" -wcheck worfile: Check workfile integrity\n");
  printf(" -m maxStep: number of operations before give up the search (maxStep*expected operation)\n");
  printf(" -s: Start in server mode\n");
  printf(" -c server_ip: Start in client mode and connect to server server_ip\n");
  printf(" -sp port: Server port, default is 17403\n");
  printf(" -nt timeout: Network timeout in millisec (default is 3000ms)\n");
  printf(" -o fileName: output result to fileName\n");
  printf(" -l: List cuda enabled devices, or use -l N for repeated TRON tail chars\n");
  printf(" -check: Check GPU kernel vs CPU\n");
  printf(" -tronPriv privateKeyHex: Compute TRON address from a private key and exit\n");
  printf(" -tronPub publicKeyHex: Compute TRON address from a public key and exit\n");
  printf(" -lianghao N: Search TRON addresses ending with N identical chars\n");
  printf(" -tronPrefix text: Search TRON addresses starting with text\n");
  printf(" -tronSuffix text: Search TRON addresses ending with text\n");
  printf(" -tronCount n: Stop after n vanity hits (default 1)\n");
  printf(" -tronDir dir: Directory used for auto-classified repeated-tail hits\n");
  printf("   TRON vanity mode uses all CUDA GPUs automatically when available,\n");
  printf("   otherwise it falls back to CPU threads; -gpuId and -g still apply.\n");
  printf(" inFile: intput configuration file\n");
  exit(0);

}

// ------------------------------------------------------------------------------------------

int getInt(string name,char *v) {

  int r;

  try {

    r = std::stoi(string(v));

  } catch(std::invalid_argument&) {

    printf("Invalid %s argument, number expected\n",name.c_str());
    exit(-1);

  }

  return r;

}

double getDouble(string name,char *v) {

  double r;

  try {

    r = std::stod(string(v));

  } catch(std::invalid_argument&) {

    printf("Invalid %s argument, number expected\n",name.c_str());
    exit(-1);

  }

  return r;

}

// ------------------------------------------------------------------------------------------

void getInts(string name,vector<int> &tokens,const string &text,char sep) {

  size_t start = 0,end = 0;
  tokens.clear();
  int item;

  try {

    while((end = text.find(sep,start)) != string::npos) {
      item = std::stoi(text.substr(start,end - start));
      tokens.push_back(item);
      start = end + 1;
    }

    item = std::stoi(text.substr(start));
    tokens.push_back(item);

  }
  catch(std::invalid_argument &) {

    printf("Invalid %s argument, number expected\n",name.c_str());
    exit(-1);

  }

}
// ------------------------------------------------------------------------------------------

string normalizeHex(string value) {

  if(value.rfind("0x",0) == 0 || value.rfind("0X",0) == 0)
    value = value.substr(2);

  return value;

}

bool isHexString(const string &value) {

  if(value.length() == 0)
    return false;

  for(char c : value) {
    if(!isxdigit((unsigned char)c))
      return false;
  }

  return true;

}

bool isNumberString(const string &value) {

  if(value.length() == 0)
    return false;

  for(char c : value) {
    if(!isdigit((unsigned char)c))
      return false;
  }

  return true;

}

bool parsePrivateKeyHex(Secp256K1 *secp,const string &value,Int *privKey) {

  string hex = normalizeHex(value);
  if(hex.length() == 0 || hex.length() > 64 || !isHexString(hex)) {
    printf("Invalid private key, expected up to 64 hexadecimal digits\n");
    return false;
  }

  privKey->SetBase16((char *)hex.c_str());
  if(privKey->IsZero() || privKey->IsGreaterOrEqual(&secp->order)) {
    printf("Invalid private key, expected a value in the [1,n-1] secp256k1 range\n");
    return false;
  }

  return true;

}

void printTronFromPrivateKey(Secp256K1 *secp,const string &value) {

  Int privKey;
  if(!parsePrivateKeyHex(secp,value,&privKey))
    exit(-1);

  Point pubKey = secp->ComputePublicKey(&privKey);
  printf("Priv    : 0x%s\n",privKey.GetBase16().c_str());
  printf("Pub(C)  : 0x%s\n",secp->GetPublicKeyHex(true,pubKey).c_str());
  printf("Pub(U)  : 0x%s\n",secp->GetPublicKeyHex(false,pubKey).c_str());
  printf("TRONHEX : 0x%s\n",TronAddress::PublicKeyToHexAddress(pubKey).c_str());
  printf("TRON    : %s\n",TronAddress::PublicKeyToBase58Address(pubKey).c_str());
  exit(0);

}

void printTronFromPublicKey(Secp256K1 *secp,const string &value) {

  Point pubKey;
  bool isCompressed = false;
  string hex = normalizeHex(value);
  if(!secp->ParsePublicKeyHex(hex,pubKey,isCompressed))
    exit(-1);

  printf("Pub(C)  : 0x%s\n",secp->GetPublicKeyHex(true,pubKey).c_str());
  printf("Pub(U)  : 0x%s\n",secp->GetPublicKeyHex(false,pubKey).c_str());
  printf("TRONHEX : 0x%s\n",TronAddress::PublicKeyToHexAddress(pubKey).c_str());
  printf("TRON    : %s\n",TronAddress::PublicKeyToBase58Address(pubKey).c_str());
  exit(0);

}

// ------------------------------------------------------------------------------------------

// Default params
static int dp = -1;
static int nbCPUThread;
static bool cpuThreadsProvided = false;
static string configFile = "";
static bool checkFlag = false;
static bool gpuEnable = false;
static vector<int> gpuId = { 0 };
static bool gpuIdProvided = false;
static vector<int> gridSize;
static string workFile = "";
static string checkWorkFile = "";
static string iWorkFile = "";
static uint32_t savePeriod = 60;
static bool saveKangaroo = false;
static bool saveKangarooByServer = false;
static string merge1 = "";
static string merge2 = "";
static string mergeDest = "";
static string mergeDir = "";
static string infoFile = "";
static double maxStep = 0.0;
static int wtimeout = 3000;
static int ntimeout = 3000;
static int port = 17403;
static bool serverMode = false;
static string serverIP = "";
static string outputFile = "";
static bool splitWorkFile = false;
static string tronPrivKey = "";
static string tronPubKey = "";
static int tronLianghao = 0;
static string tronPrefix = "";
static string tronSuffix = "";
static uint64_t tronCount = 1;
static string tronClassifyDir = "";

int main(int argc, char* argv[]) {

#ifdef USE_SYMMETRY
  printf("Kangaroo v" RELEASE " (with symmetry)\n");
#else
  printf("Kangaroo v" RELEASE "\n");
#endif

  // Global Init
  Timer::Init();
  rseed(Timer::getSeed32());

  // Init SecpK1
  Secp256K1 *secp = new Secp256K1();
  secp->Init();

  int a = 1;
  nbCPUThread = Timer::getCoreNumber();

  while (a < argc) {

    if(strcmp(argv[a], "-t") == 0) {
      CHECKARG("-t",1);
      nbCPUThread = getInt("nbCPUThread",argv[a]);
      cpuThreadsProvided = true;
      a++;
    } else if(strcmp(argv[a],"-d") == 0) {
      CHECKARG("-d",1);
      dp = getInt("dpSize",argv[a]);
      a++;
    } else if (strcmp(argv[a], "-h") == 0) {
      printUsage();
    } else if(strcmp(argv[a],"-l") == 0) {
      if(a < argc - 1 && isNumberString(string(argv[a + 1]))) {
        a++;
        tronLianghao = getInt("lianghao",argv[a]);
        a++;
        continue;
      }

#ifdef WITHGPU
      GPUEngine::PrintCudaInfo();
#else
      printf("GPU code not compiled, use -DWITHGPU when compiling.\n");
#endif
      exit(0);

    } else if(strcmp(argv[a],"-w") == 0) {
      CHECKARG("-w",1);
      workFile = string(argv[a]);
      a++;
    } else if(strcmp(argv[a],"-i") == 0) {
      CHECKARG("-i",1);
      iWorkFile = string(argv[a]);
      a++;
    } else if(strcmp(argv[a],"-wm") == 0) {
      CHECKARG("-wm",1);
      merge1 = string(argv[a]);
      CHECKARG("-wm",2);
      merge2 = string(argv[a]);
      a++;
      if(a<argc) {
        // classic merge
        mergeDest = string(argv[a]);
        a++;
      }
    } else if(strcmp(argv[a],"-wmdir") == 0) {
      CHECKARG("-wmdir",1);
      mergeDir = string(argv[a]);
      CHECKARG("-wmdir",2);
      mergeDest = string(argv[a]);
      a++;
    }  else if(strcmp(argv[a],"-wcheck") == 0) {
      CHECKARG("-wcheck",1);
      checkWorkFile = string(argv[a]);
      a++;
    }  else if(strcmp(argv[a],"-winfo") == 0) {
      CHECKARG("-winfo",1);
      infoFile = string(argv[a]);
      a++;
    } else if(strcmp(argv[a],"-o") == 0) {
      CHECKARG("-o",1);
      outputFile = string(argv[a]);
      a++;
    } else if(strcmp(argv[a],"-wi") == 0) {
      CHECKARG("-wi",1);
      savePeriod = getInt("savePeriod",argv[a]);
      a++;
    } else if(strcmp(argv[a],"-wt") == 0) {
      CHECKARG("-wt",1);
      wtimeout = getInt("timeout",argv[a]);
      a++;
    } else if(strcmp(argv[a],"-nt") == 0) {
      CHECKARG("-nt",1);
      ntimeout = getInt("timeout",argv[a]);
      a++;
    } else if(strcmp(argv[a],"-m") == 0) {
      CHECKARG("-m",1);
      maxStep = getDouble("maxStep",argv[a]);
      a++;
    } else if(strcmp(argv[a],"-ws") == 0) {
      a++;
      saveKangaroo = true;
    } else if(strcmp(argv[a],"-wss") == 0) {
      a++;
      saveKangarooByServer = true;
    } else if(strcmp(argv[a],"-wsplit") == 0) {
      a++;
      splitWorkFile = true;
    } else if(strcmp(argv[a],"-wpartcreate") == 0) {
      CHECKARG("-wpartcreate",1);
      workFile = string(argv[a]);
      Kangaroo::CreateEmptyPartWork(workFile);
      exit(0);
    } else if(strcmp(argv[a],"-s") == 0) {
      a++;
      serverMode = true;
    } else if(strcmp(argv[a],"-c") == 0) {
      CHECKARG("-c",1);
      serverIP = string(argv[a]);
      a++;
    } else if(strcmp(argv[a],"-sp") == 0) {
      CHECKARG("-sp",1);
      port = getInt("serverPort",argv[a]);
      a++;
    } else if(strcmp(argv[a],"-gpu") == 0) {
      gpuEnable = true;
      a++;
    } else if(strcmp(argv[a],"-gpuId") == 0) {
      CHECKARG("-gpuId",1);
      getInts("gpuId",gpuId,string(argv[a]),',');
      gpuIdProvided = true;
      a++;
    } else if(strcmp(argv[a],"-g") == 0) {
      CHECKARG("-g",1);
      getInts("gridSize",gridSize,string(argv[a]),',');
      a++;
    } else if(strcmp(argv[a],"-v") == 0) {
      ::exit(0);
    } else if(strcmp(argv[a],"-check") == 0) {
      checkFlag = true;
      a++;
    } else if(strcmp(argv[a],"-tronPriv") == 0) {
      CHECKARG("-tronPriv",1);
      tronPrivKey = string(argv[a]);
      a++;
    } else if(strcmp(argv[a],"-tronPub") == 0) {
      CHECKARG("-tronPub",1);
      tronPubKey = string(argv[a]);
      a++;
    } else if(strcmp(argv[a],"-lianghao") == 0) {
      CHECKARG("-lianghao",1);
      tronLianghao = getInt("lianghao",argv[a]);
      a++;
    } else if(strcmp(argv[a],"-tronPrefix") == 0) {
      CHECKARG("-tronPrefix",1);
      tronPrefix = string(argv[a]);
      a++;
    } else if(strcmp(argv[a],"-tronSuffix") == 0) {
      CHECKARG("-tronSuffix",1);
      tronSuffix = string(argv[a]);
      a++;
    } else if(strcmp(argv[a],"-tronCount") == 0) {
      CHECKARG("-tronCount",1);
      tronCount = (uint64_t)getInt("tronCount",argv[a]);
      a++;
    } else if(strcmp(argv[a],"-tronDir") == 0) {
      CHECKARG("-tronDir",1);
      tronClassifyDir = string(argv[a]);
      a++;
    } else if(a == argc - 1) {
      configFile = string(argv[a]);
      a++;
    } else {
      printf("Unexpected %s argument\n",argv[a]);
      exit(-1);
    }

  }

  if(gridSize.size() == 0) {
    for(int i = 0; i < gpuId.size(); i++) {
      gridSize.push_back(0);
      gridSize.push_back(0);
    }
  } else if(gridSize.size() != gpuId.size() * 2) {
    printf("Invalid gridSize or gpuId argument, must have coherent size\n");
    exit(-1);
  }

  if(tronPrivKey.length() > 0 && tronPubKey.length() > 0) {
    printf("Use either -tronPriv or -tronPub, not both\n");
    exit(-1);
  }

  if((tronLianghao > 0 || tronPrefix.length() > 0 || tronSuffix.length() > 0) &&
     (tronPrivKey.length() > 0 || tronPubKey.length() > 0 || configFile.length() > 0 || iWorkFile.length() > 0)) {
    printf("TRON vanity mode cannot be combined with kangaroo input files or -tronPriv/-tronPub\n");
    exit(-1);
  }

  if(tronPrivKey.length() > 0)
    printTronFromPrivateKey(secp,tronPrivKey);
  if(tronPubKey.length() > 0)
    printTronFromPublicKey(secp,tronPubKey);

  if(tronLianghao > 0 || tronPrefix.length() > 0 || tronSuffix.length() > 0) {
    TRON_VANITY_CONFIG cfg;
    cfg.nbThread = nbCPUThread;
    cfg.cpuThreadsProvided = cpuThreadsProvided;
    cfg.repeatTailLength = tronLianghao;
    cfg.maxFound = tronCount;
    cfg.prefix = tronPrefix;
    cfg.suffix = tronSuffix;
    cfg.outputFile = outputFile;
    cfg.classifyDir = tronClassifyDir;
    cfg.gpuIdsProvided = gpuIdProvided;
    cfg.gpuIds = gpuId;
    cfg.gridSize = gridSize;
    if(!TronVanity::Run(secp,cfg))
      exit(-1);
    exit(0);
  }

  Kangaroo *v = new Kangaroo(secp,dp,gpuEnable,workFile,iWorkFile,savePeriod,saveKangaroo,saveKangarooByServer,
                             maxStep,wtimeout,port,ntimeout,serverIP,outputFile,splitWorkFile);
  if(checkFlag) {
    v->Check(gpuId,gridSize);  
    exit(0);
  } else {
    if(checkWorkFile.length() > 0) {
      v->CheckWorkFile(nbCPUThread,checkWorkFile);
      exit(0);
    } if(infoFile.length()>0) {
      v->WorkInfo(infoFile);
      exit(0);
    } else if(mergeDir.length() > 0) {
      v->MergeDir(mergeDir,mergeDest);
      exit(0);
    } else if(merge1.length()>0) {
      v->MergeWork(merge1,merge2,mergeDest);
      exit(0);
    } if(iWorkFile.length()>0) {
      if( !v->LoadWork(iWorkFile) )
        exit(-1);
    } else if(configFile.length()>0) {
      if( !v->ParseConfigFile(configFile) )
        exit(-1);
    } else {
      if(serverIP.length()==0) {
        ::printf("No input file to process\n");
        exit(-1);
      }
    }
    if(serverMode)
      v->RunServer();
    else
      v->Run(nbCPUThread,gpuId,gridSize);
  }

  return 0;

}
