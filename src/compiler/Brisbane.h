#ifndef POLY_BRISBANE_H
#define POLY_BRISBANE_H

#include "polly/ScopInfo.h"
#include "llvm/Pass.h"

using namespace llvm;

namespace llvm {
void initializeBrisbaneXPass(PassRegistry &);
}

namespace polly {
Pass *createBrisbaneXPass();
}

namespace polly {

class BrisbaneX : public FunctionPass {
  std::unique_ptr<ScopInfo> Result;

public:
  static char ID;

  BrisbaneX() : FunctionPass(ID) {}
  ~BrisbaneX() override = default;

  ScopInfo *getSI() { return Result.get(); }
  const ScopInfo *getSI() const { return Result.get(); }

  bool runOnFunction(Function &F) override;

  void releaseMemory() override { Result.reset(); }

  void print(raw_ostream &O, const Module *M = nullptr) const override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

}

#endif /* POLY_BRISBANE_H */

