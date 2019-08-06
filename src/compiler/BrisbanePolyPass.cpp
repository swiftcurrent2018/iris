#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
struct BrisbanePolyPass : public FunctionPass {
  static char ID;
  BrisbanePolyPass() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    errs() << "BrisbanePolyPass: ";
    errs().write_escaped(F.getName()) << '\n';
    return false;
  }
};
} // end of anonymous namespace

char BrisbanePolyPass::ID = 0;
static RegisterPass<BrisbanePolyPass> X("brisbane-poly", "Brisbane Polyhedral Analyzer", false, false);

