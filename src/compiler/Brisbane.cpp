#include "Brisbane.h"
#include "polly/LinkAllPasses.h"
#include "polly/Options.h"
#include "polly/ScopBuilder.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"

using namespace llvm;
using namespace polly;

bool Brisbane::runOnFunction(Function &F) {
  auto &SD = getAnalysis<ScopDetectionWrapperPass>().getSD();
  auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
  auto const &DL = F.getParent()->getDataLayout();
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  auto &AC = getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
  auto &ORE = getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE();

  SI.reset(new ScopInfo{DL, SD, SE, LI, AA, DT, AC, ORE});

  std::string Result;
  raw_string_ostream OS(Result);

  StringRef FnName = F.getName();

  OS << "static int " << FnName << "(BRISBANE_POLY_KERNEL_ARGS) {\n";

  if ((*SI).empty()) {
    OS << "  return 0;\n}\n\n";
    errs() << OS.str();
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
      OS << "  BRISBANE_POLY_ARRAY_" << dim << "D(" << FnName << ", " <<  N << ", sizeof(" << *T << "), ";
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

        //errs() << "DOMAIN:" << D_str << "\n";

        std::set<std::string>* params = printDomain(OS, D_dim, D_str);
        printMemoryAccess(OS, FnName, MA, params);
        OS << "  }\n";
      }
    }
  }
  OS << "  return 1;\n}\n\n";
  errs() << OS.str();
  return false;
}

void Brisbane::print(raw_ostream &OS, const Module *M) const {
  for (auto &It : *SI) {
    if (It.second)
      It.second->print(OS, true);
    else
      OS << "No Scop!\n";
  }
}

void Brisbane::getAnalysisUsage(AnalysisUsage &AU) const {
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

std::string Brisbane::getArrayName(std::string N) {
  return N.find("MemRef_") == 0 ? N.substr(7) : N;
}

std::set<std::string>* Brisbane::printDomain(raw_string_ostream &OS, unsigned dim, std::string str) {
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

void Brisbane::printMemoryAccess(raw_string_ostream &OS, StringRef F, MemoryAccess* MA, std::set<std::string>* params) {
  isl::id I = MA->getOriginalArrayId();
  isl::map M = MA->getLatestAccessRelation();
  MemoryAccess::AccessType AT = MA->getType();
  std::string N = getArrayName(I.get_name());
  std::string M_str = M.to_str();

  //errs() << "MEMORYACCESS:" << M_str << "\n";

  std::string MS = M_str.substr(M_str.find(I.get_name()) + I.get_name().size() + 1);
  MS = MS.substr(0, MS.size() - 3);

  std::pair<StringRef, StringRef> Split = StringRef(MS).split(", ");

  if (AT == MemoryAccess::READ) OS << "  BRISBANE_POLY_READ(";
  else if (AT == MemoryAccess::MUST_WRITE) OS << "  BRISBANE_POLY_MUWR(";
  else if (AT == MemoryAccess::MAY_WRITE) OS << "  BRISBANE_POLY_MAWR(";
  else OS << "  BRISBANE_POLY_XXXX(";
  OS << F << ", " << N << ", ";
  printRange(OS, Split.first, 0, params);
  OS << ", ";
  printRange(OS, Split.second, 0, params);
  OS << ");\n";

  if (AT == MemoryAccess::READ) OS << "  BRISBANE_POLY_READ(";
  else if (AT == MemoryAccess::MUST_WRITE) OS << "  BRISBANE_POLY_MUWR(";
  else if (AT == MemoryAccess::MAY_WRITE) OS << "  BRISBANE_POLY_MAWR(";
  else OS << "  BRISBANE_POLY_XXXX(";
  OS << F << ", " << N << ", ";
  printRange(OS, Split.first, 1, params);
  OS << ", ";
  printRange(OS, Split.second, 1, params);
  OS << ");\n";
}

void Brisbane::printRange(raw_string_ostream &OS, StringRef S, int i, std::set<std::string>* params) {
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

char Brisbane::ID = 0;

Pass *polly::createBrisbanePass() {
  return new Brisbane();
}

static RegisterPass<Brisbane> X("brisbane", "Brisbane Polyhedral Analyzer", false, false);

/*
INITIALIZE_PASS_BEGIN(Brisbane, "brisbane", "Brisbane Polyhedral Analyzer", false, false);
INITIALIZE_PASS_DEPENDENCY(PollyCanonicalize)
INITIALIZE_PASS_DEPENDENCY(ScopInfoWrapperPass)
INITIALIZE_PASS_END(Brisbane, "brisbane", "Brisbane Polyhedral Analyzer", false, false)
*/

