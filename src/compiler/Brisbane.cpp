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
  return false;
}

void Brisbane::print(raw_ostream &OS, const Module *M) const {
  for (auto &It : *Result) {
    if (It.second)
      It.second->print(OS, BrisbanePrintInstructions);
    else
      OS << "Fuck Scop!\n";
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

//static RegisterPass<Brisbane> X("brisbane", "Brisbane Polyhedral Analyzer", false, false);
char Brisbane::ID = 0;

Pass *polly::createBrisbanePass() {
  return new Brisbane();
}

INITIALIZE_PASS_BEGIN(Brisbane, "brisbane", "Brisbane Polyhedral Analyzer", false, false);
INITIALIZE_PASS_DEPENDENCY(PollyCanonicalize)
INITIALIZE_PASS_DEPENDENCY(ScopInfoWrapperPass)
INITIALIZE_PASS_END(Brisbane, "brisbane", "Brisbane Polyhedral Analyzer", false, false)

