#ifndef POLY_BRISBANE_H
#define POLY_BRISBANE_H

#include "polly/ScopInfo.h"
#include "llvm/Pass.h"

using namespace llvm;

namespace llvm {
void initializeBrisbanePass(PassRegistry &);
}

namespace polly {
Pass *createBrisbanePass();
}

namespace polly {

class BrisbaneLLVM : public FunctionPass {
  std::unique_ptr<ScopInfo> SI;

public:
  static char ID;

  BrisbaneLLVM() : FunctionPass(ID) {}
  ~BrisbaneLLVM() override = default;

  ScopInfo *getSI() { return SI.get(); }
  const ScopInfo *getSI() const { return SI.get(); }

  bool runOnFunction(Function &F) override;

  void releaseMemory() override { SI.reset(); }

  void print(raw_ostream &O, const Module *M = nullptr) const override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  std::string getArrayName(std::string N);
  std::set<std::string>* printDomain(raw_string_ostream &OS, unsigned dim, std::string str);
  void printMemoryAccess(raw_string_ostream &OS, StringRef F, MemoryAccess* MA, std::set<std::string>* params);
  void printRange(raw_string_ostream &OS, StringRef S, int i, std::set<std::string>* params);
};

}

#endif /* POLY_BRISBANE_H */

