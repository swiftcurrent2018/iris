#include "Brisbane.h"
#include "polly/LinkAllPasses.h"
#include "polly/Options.h"
#include "polly/ScopBuilder.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"

using namespace llvm;
using namespace polly;

bool BrisbaneX::runOnFunction(Function &F) {
  auto &SD = getAnalysis<ScopDetectionWrapperPass>().getSD();
  auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
  auto const &DL = F.getParent()->getDataLayout();
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  auto &AC = getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
  auto &ORE = getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE();

  Result.reset(new ScopInfo{DL, SD, SE, LI, AA, DT, AC, ORE});

  StringRef FnName = F.getName();
  errs() << "--- FUNCTION ---\n" << FnName << "\n--- ARRAYS ---\n";

  bool empty = (*Result).empty();
  if (empty) errs() << "EMPTY Nonaffine Function\n";

  for (auto &It : *Result) {
    if (!It.second) {
      errs() << "Nonaffine Scop\n";
      continue;
    }
    Scop* S = It.second.get();
    for (ScopArrayInfo* Array : S->arrays()) {
      Type* T = Array->getElementType();
      unsigned dim = Array->getNumberOfDimensions();
      errs() << Array->getName() << ":" << *T << ":" << dim;
      if (dim == 0) continue;
      for (unsigned i = 0; i < dim; i++) {
        const SCEV* DS = Array->getDimensionSize(i);
        if (DS) {
          if (const SCEVUnknown *U = dyn_cast<SCEVUnknown>(DS)) {
            Value* V = U->getValue();
            StringRef N = V->getName();
            errs() << ":" << N;
          }
        }
        else errs() << ":*";
      }
      errs() << "\n";
    }
    errs() << "--- MEMACCESS ---\n";

    for (const ScopStmt &Stmt : *S) {
      for (MemoryAccess* MA : Stmt) {
        MemoryAccess::AccessType AT = MA->getType();
        bool scalar = MA->isScalarKind();
        if (scalar) continue;
        isl::id I = MA->getOriginalArrayId();
        isl::map M = MA->getLatestAccessRelation();
        isl::set D = MA->getStatement()->getDomain();

        std::string name = I.get_name();

        errs() << name << "\n";

        unsigned D_dim = D.n_dim();
        unsigned D_params = D.dim(isl::dim::param);

        errs() << "DOMAIN:" << D_params << ":" << D_dim << ":" << D.to_str() << "\n";

        if (AT == MemoryAccess::READ) errs() << "READ:";
        else if (AT == MemoryAccess::MUST_WRITE) errs() << "MUWR:";
        else if (AT == MemoryAccess::MAY_WRITE) errs() << "MAWR:";
        else errs() << "????:";

        errs() << M.to_str() << "\n";
      }
    }
  }
  errs() << "--- ENDOFFUCNTION ---\n";
  return false;
}

void BrisbaneX::print(raw_ostream &OS, const Module *M) const {
  for (auto &It : *Result) {
    if (It.second)
      It.second->print(OS, true);
    else
      OS << "No Scop!\n";
  }
}

void BrisbaneX::getAnalysisUsage(AnalysisUsage &AU) const {
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

char BrisbaneX::ID = 0;

Pass *polly::createBrisbaneXPass() {
  return new BrisbaneX();
}

static RegisterPass<BrisbaneX> X("brisbane-x", "BrisbaneX Polyhedral Analyzer", false, false);

/*
INITIALIZE_PASS_BEGIN(Brisbane, "brisbane", "Brisbane Polyhedral Analyzer", false, false);
INITIALIZE_PASS_DEPENDENCY(PollyCanonicalize)
INITIALIZE_PASS_DEPENDENCY(ScopInfoWrapperPass)
INITIALIZE_PASS_END(Brisbane, "brisbane", "Brisbane Polyhedral Analyzer", false, false)
*/

