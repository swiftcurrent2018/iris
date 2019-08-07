#ifndef POLY_BRISBANE_H
#define POLY_BRISBANE_H

#include "polly/ScopInfo.h"
#include "llvm/Pass.h"

using namespace llvm;

namespace llvm {
void initializeBrisbanePass(PassRegistry &);
}

namespace polly {

class Brisbane : public FunctionPass {
  std::unique_ptr<ScopInfo> Result;

public:
  static char ID;

  Brisbane() : FunctionPass(ID) {}
  ~Brisbane() override = default;

  ScopInfo *getSI() { return Result.get(); }
  const ScopInfo *getSI() const { return Result.get(); }

  bool runOnFunction(Function &F) override;

  void releaseMemory() override { Result.reset(); }

  void print(raw_ostream &O, const Module *M = nullptr) const override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;
};
}

#endif /* POLY_BRISBANE_H */
