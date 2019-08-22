#ifndef BRISBANE_SRC_CC_LLVM_BRISBANE_LLVM_H
#define BRISBANE_SRC_CC_LLVM_BRISBANE_LLVM_H

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
  bool doFinalization(Module &M) override;

  void releaseMemory() override { SI.reset(); }

//  void print(raw_ostream &O, const Module *M = nullptr) const override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  void printFunctionPolyAvailable(raw_ostream &OS, StringRef &Fname, int available);
  std::string getArrayName(std::string N);
  std::set<std::string>* printDomain(raw_string_ostream &OS, unsigned dim, std::string str);
  void printMemoryAccess(raw_string_ostream &OS, StringRef F, MemoryAccess* MA, std::set<std::string>* params);
  void printRange(raw_string_ostream &OS, StringRef S, int i, std::set<std::string>* params);

private:
  std::string S;
  SmallVector<Function*, 64> Fs;
  SmallVector<Function*, 64> Ff;
};

}

#endif /* BRISBANE_SRC_CC_LLVM_BRISBANE_LLVM_H */

