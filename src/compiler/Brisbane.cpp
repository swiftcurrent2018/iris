#include "polly/Brisbane.h"
#include "polly/LinkAllPasses.h"
#include "polly/Options.h"
#include "polly/ScopBuilder.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"

using namespace llvm;
using namespace polly;

static cl::opt<bool> BrisbanePrintInstructions(
    "brisbane-print-instructions", cl::desc("Brisbane output instructions per ScopStmt"),
    cl::Hidden, cl::Optional, cl::init(false), cl::cat(PollyCategory));

bool Brisbane::runOnFunction(Function &F) {
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
  errs() << FnName << "\n";

  bool empty = (*Result).empty();
  if (empty) errs() << "EMPTY Nonaffine Function\n";

  for (auto &It : *Result) {
    if (!It.second) {
      errs() << "Nonaffine Scop\n";
      continue;
    }
    Scop* S = It.second.get();
    /*
       size_t nparams = S->getNumParams();
       for (const llvm::SCEV *Parameter : S->parameters()) {
       llvm::Type* T = Parameter->getType();
       Parameter->dump();
       }
    */

    for (ScopArrayInfo* Array : S->arrays()) {
      Type* T = Array->getElementType();
      errs() << *T << " " << Array->getName() << "\n";
      unsigned dim = Array->getNumberOfDimensions();
      if (dim == 0) continue;
      for (unsigned i = 0; i < dim; i++) {
        const SCEV* DS = Array->getDimensionSize(i);
        if (DS) DS->dump();
        else errs() << "*" << "\n";
      }
    }

    for (const ScopStmt &Stmt : *S) {
      for (MemoryAccess* MA : Stmt) {
        MemoryAccess::AccessType AT = MA->getType();
        bool scalar = MA->isScalarKind();
        if (scalar) continue;
        isl::map AccessRelation = MA->getLatestAccessRelation();
        isl::set StmtDomain = MA->getStatement()->getDomain();

        isl_map* M = AccessRelation.get();
        isl_set* D = StmtDomain.get();

        char* tmp = isl_map_to_str(M);
        errs() << "MemoryAccess:" << tmp << "\n";
        free(tmp);

        tmp = isl_set_to_str(D);
        errs() << "Domain:" << tmp << "\n";
        free(tmp);

        if (AT == MemoryAccess::READ) {
          errs() << "READ!\n";
        } else if (AT == MemoryAccess::MUST_WRITE) {
          errs() << "MUST WRITE!\n";
        } else if (AT == MemoryAccess::MAY_WRITE) {
          errs() << "MAY WRITE!\n";
        } else {
          errs() << "FUCK!\n";
        }
      }
    }
  }

  return false;
}

void Brisbane::print(raw_ostream &OS, const Module *M) const {
  for (auto &It : *Result) {
    if (It.second)
      It.second->print(OS, BrisbanePrintInstructions);
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

char Brisbane::ID = 0;

Pass *polly::createBrisbanePass() {
  return new Brisbane();
}

//static RegisterPass<Brisbane> X("brisbane", "Brisbane Polyhedral Analyzer", false, false);

INITIALIZE_PASS_BEGIN(Brisbane, "brisbane", "Brisbane Polyhedral Analyzer", false, false);
INITIALIZE_PASS_DEPENDENCY(PollyCanonicalize)
INITIALIZE_PASS_DEPENDENCY(ScopInfoWrapperPass)
INITIALIZE_PASS_END(Brisbane, "brisbane", "Brisbane Polyhedral Analyzer", false, false)

