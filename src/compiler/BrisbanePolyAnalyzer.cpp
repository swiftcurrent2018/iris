#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

using namespace llvm;

namespace {

  struct BrisbanePolyAnalyzer : public ModulePass {
    static char ID;

    BrisbanePolyAnalyzer();

    bool runOnModule(Module &M) override;
    bool runOnFunction(Function &F);

  };

}

char BrisbanePolyAnalyzer::ID = 0;

BrisbanePolyAnalyzer::BrisbanePolyAnalyzer() : ModulePass (ID) {
}

bool BrisbanePolyAnalyzer::runOnModule(Module &M) {
  errs().write_escaped(M.getSourceFileName()) << '\n';
  for (auto &F : M) runOnFunction(F);
  return true;
}

bool BrisbanePolyAnalyzer::runOnFunction(Function &F) {
  StringRef Name = F.getName();
  errs().write_escaped(Name) << '\n';
  return true;
}

static RegisterPass<BrisbanePolyAnalyzer> X("brisbane-poly", "Brisbane Polyhedral Analyzer", false, false);

