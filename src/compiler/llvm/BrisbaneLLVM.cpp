#include "BrisbaneLLVM.h"
#include "polly/LinkAllPasses.h"
#include "polly/Options.h"
#include "polly/ScopBuilder.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Support/raw_ostream.h"

#define BRED      "\033[1;31m"
#define BGREEN    "\033[1;32m"
#define BYELLOW   "\033[1;33m"
#define BBLUE     "\033[1;34m"
#define BPURPLE   "\033[1;35m"
#define BCYAN     "\033[1;36m"
#define BGRAY     "\033[1;37m"
#define RESET     "\x1B[m"
#define CHECK_O   "\u2714 "
#define CHECK_X   "\u2716 "

using namespace llvm;
using namespace polly;

bool BrisbaneLLVM::runOnFunction(Function &F) {
  auto &SD = getAnalysis<ScopDetectionWrapperPass>().getSD();
  auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
  auto const &DL = F.getParent()->getDataLayout();
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  auto &AC = getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
  auto &ORE = getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE();

  SI.reset(new ScopInfo{DL, SD, SE, LI, AA, DT, AC, ORE});

  raw_string_ostream OS(S);

  StringRef Fname = F.getName();

  OS << "static int " << Fname << "(BRISBANE_POLY_KERNEL_ARGS) {\n";

  if ((*SI).empty()) {
    OS << "  return 0;\n}\n\n";
    printFunctionPolyAvailable(OS, Fname, 0);
    Ff.push_back(&F);
    return false;
  }

  for (auto &It : *SI) {
    if (!It.second) continue;
    Scop* S = It.second.get();
    for (ScopArrayInfo* Array : S->arrays()) {
      Type* T = Array->getElementType();
      std::string N = getArrayName(Array->getName());
      unsigned dim = Array->getNumberOfDimensions();
      if (dim == 0) continue;
      OS << "  BRISBANE_POLY_ARRAY_" << dim << "D(" << Fname << ", " <<  N << ", sizeof(" << *T << "), ";
      for (unsigned i = 0; i < dim; i++) {
        const SCEV* DS = Array->getDimensionSize(i);
        if (DS) {
          if (const SCEVUnknown *U = dyn_cast<SCEVUnknown>(DS)) {
            Value* V = U->getValue();
            StringRef N = V->getName();
            OS << N;
          }
        }
        else OS << "0";
        if (i != dim - 1) OS << ", ";
      }
      OS << ");\n";
    }

    for (const ScopStmt &Stmt : *S) {
      for (MemoryAccess* MA : Stmt) {
        if (MA->isScalarKind()) continue;
        OS << "  {\n";
        isl::id I = MA->getOriginalArrayId();
        isl::map M = MA->getLatestAccessRelation();
        isl::set D = MA->getStatement()->getDomain();

        std::string N = getArrayName(I.get_name());

        unsigned D_dim = D.n_dim();
        unsigned D_params = D.dim(isl::dim::param);
        std::string D_str = D.to_str();

        std::set<std::string>* params = printDomain(OS, D_dim, D_str);
        printMemoryAccess(OS, Fname, MA, params);
        OS << "  }\n";
      }
    }
  }
  OS << "  return 1;\n}\n\n";
  printFunctionPolyAvailable(OS, Fname, 1);
  Fs.push_back(&F);
  return false;
}

bool BrisbaneLLVM::doFinalization(Module &M) {
  std::string filename = M.getSourceFileName();
  filename.replace(filename.end() - 6, filename.end(), "poly.h");
  StringRef Filepath(filename);
  std::error_code EC;

  llvm::raw_fd_ostream OS(Filepath, EC, llvm::sys::fs::OF_None);
  if (EC) {
    llvm::errs() << EC.message() << "\n";
    return false;
  }
  OS << S;
  OS.close();

  outs() << BBLUE  CHECK_O "BrisbaneLLVM Pass generates [" << Filepath << "]" RESET "\n";

  if (!Fs.empty()) {
    outs() << BGREEN CHECK_O << Fs.size() << " functions [";
    for (auto I = Fs.begin(), E = Fs.end(); I != E; ++I) {
      outs() << (*I)->getName();
      if ((I + 1) != E) outs() << ", ";
    }
    outs() << "] are available." RESET "\n";
  }

  if (!Ff.empty()) {
    outs() << BRED   CHECK_X << Ff.size() << " functions [";
    for (auto I = Ff.begin(), E = Ff.end(); I != E; ++I) {
      outs() << (*I)->getName();
      if ((I + 1) != E) outs() << ", ";
    }
    outs() << "] are NOT available." RESET "\n";
  }

  return true;
}

void BrisbaneLLVM::printFunctionPolyAvailable(raw_ostream &OS, StringRef& Fname, int available) {
  OS << "static int " << Fname << "_poly_available() { return " << available << "; }\n\n";
}

//void BrisbaneLLVM::print(raw_ostream &OS, const Module *M) const { }

void BrisbaneLLVM::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addRequired<RegionInfoPass>();
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addRequiredTransitive<ScalarEvolutionWrapperPass>();
  AU.addRequiredTransitive<ScopDetectionWrapperPass>();
  AU.addRequired<AAResultsWrapperPass>();
  AU.addRequired<AssumptionCacheTracker>();
  AU.addRequired<OptimizationRemarkEmitterWrapperPass>();
  AU.setPreservesAll();
}  

std::string BrisbaneLLVM::getArrayName(std::string N) {
  return N.find("MemRef_") == 0 ? N.substr(7) : N;
}

std::set<std::string>* BrisbaneLLVM::printDomain(raw_string_ostream &OS, unsigned dim, std::string str) {
  std::set<std::string>* params = new std::set<std::string>;
  StringRef S = StringRef(str);
  size_t I = S.find(": ");
  if (I == StringRef::npos) return params;
  S = S.substr(I + 2);

  while (!S.empty()) {
    std::pair<StringRef, StringRef> Split = S.split(" and ");
    S = Split.second;
    if (!Split.first.contains("<=")) continue;
    std::pair<StringRef, StringRef> S1 = Split.first.split(" <= ");
    std::pair<StringRef, StringRef> S2 = S1.second.split(" < ");
    if (S2.second.endswith("}")) S2.second = S2.second.substr(0, S2.second.size() - 2);
    OS << "  BRISBANE_POLY_DOMAIN(" << S2.first << ", " << S1.first <<  ", " << S2.second << " - 1);\n";
    params->insert(S2.first.str());
  }
  return params;
}

void BrisbaneLLVM::printMemoryAccess(raw_string_ostream &OS, StringRef F, MemoryAccess* MA, std::set<std::string>* params) {
  isl::id I = MA->getOriginalArrayId();
  isl::map M = MA->getLatestAccessRelation();
  MemoryAccess::AccessType AT = MA->getType();
  const ScopArrayInfo* Array = MA->getLatestScopArrayInfo();
  unsigned dim = Array->getNumberOfDimensions();
  std::string N = getArrayName(I.get_name());
  std::string M_str = M.to_str();

  std::string MS = M_str.substr(M_str.find(I.get_name()) + I.get_name().size() + 1);
  MS = MS.substr(0, MS.size() - 3);

  std::pair<StringRef, StringRef> Split = StringRef(MS).split(", ");

  if (AT == MemoryAccess::READ) OS << "  BRISBANE_POLY_READ_" << dim << "D(";
  else if (AT == MemoryAccess::MUST_WRITE) OS << "  BRISBANE_POLY_MUWR_" << dim << "D(";
  else if (AT == MemoryAccess::MAY_WRITE) OS << "  BRISBANE_POLY_MAWR_" << dim << "D(";
  else OS << "  BRISBANE_POLY_XXXX_" << dim << "D(";
  OS << F << ", " << N << ", ";
  printRange(OS, Split.first, 0, params);
  if (dim == 1) {}
  else if (dim == 2) {
    OS << ", ";
    printRange(OS, Split.second, 0, params);
  } else {
    OS << ", ";
    printRange(OS, Split.second, 0, params);
  }
  OS << ");\n";

  if (AT == MemoryAccess::READ) OS << "  BRISBANE_POLY_READ_" << dim << "D(";
  else if (AT == MemoryAccess::MUST_WRITE) OS << "  BRISBANE_POLY_MUWR_" << dim << "D(";
  else if (AT == MemoryAccess::MAY_WRITE) OS << "  BRISBANE_POLY_MAWR_" << dim << "D(";
  else OS << "  BRISBANE_POLY_XXXX_" << dim << "D(";
  OS << F << ", " << N << ", ";
  printRange(OS, Split.first, 1, params);
  if (dim == 1) {}
  else if (dim == 2) {
    OS << ", ";
    printRange(OS, Split.second, 1, params);
  } else {
    OS << ", ";
    printRange(OS, Split.second, 1, params);
  }
  OS << ");\n";
}

void BrisbaneLLVM::printRange(raw_string_ostream &OS, StringRef S, int i, std::set<std::string>* params) {
  while (!S.empty()) {
    std::pair<StringRef, StringRef> Split = StringRef(S).split(" ");
    S = Split.first;
    bool found = false;
    for (std::string P : *params) {
      if (P == S) {
        OS << P << "[" << i << "]";
        found = true;
        continue;
      }
    }
    if (!found) OS << S;
    S = Split.second;
  }
}

char BrisbaneLLVM::ID = 0;

Pass *polly::createBrisbanePass() {
  return new BrisbaneLLVM();
}

static RegisterPass<BrisbaneLLVM> X("brisbane", "Brisbane Polyhedral Analyzer", false, false);

/*
INITIALIZE_PASS_BEGIN(BrisbaneLLVM, "brisbane", "Brisbane Polyhedral Analyzer", false, false);
INITIALIZE_PASS_DEPENDENCY(PollyCanonicalize)
INITIALIZE_PASS_DEPENDENCY(ScopInfoWrapperPass)
INITIALIZE_PASS_END(BrisbaneLLVM, "brisbane", "Brisbane Polyhedral Analyzer", false, false)
*/

