#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/raw_ostream.h"
#include <sstream>
#include <string>
#include <stdio.h>

#define BRED      "\033[1;31m"
#define BGREEN    "\033[1;32m"
#define BYELLOW   "\033[1;33m"
#define BBLUE     "\033[1;34m"
#define BPURPLE   "\033[1;35m"
#define BCYAN     "\033[1;36m"
#define BGRAY     "\033[1;37m"
#define RESET     "\x1B[m"
#define CHECK_O   "\u2714 "
#define CHECK_X   "\u2716 "

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::driver;
using namespace clang::tooling;
using namespace std;

static llvm::cl::OptionCategory BRISBANE_OPTION_CATEGORY("Brisbane Clang");

class BrisbanePOLYCallback : public MatchFinder::MatchCallback {
public:
  BrisbanePOLYCallback(Rewriter &Rewrite) : Rewrite(Rewrite) {}

  void run(const MatchFinder::MatchResult &Result) override {
    const FunctionDecl *F = Result.Nodes.getNodeAs<FunctionDecl>("kernel");
    Loc = F->getBody()->getBeginLoc();
    StringRef Fname = F->getName();
    const unsigned nparams = F->getNumParams();

    llvm::raw_string_ostream OS(S);
    OS << "typedef struct {\n";
    for (const ParmVarDecl *P : F->parameters()) {
      const QualType T = P->getType();
      if (T->isPointerType()) OS << "  brisbane_poly_mem";
      else OS << "  " << T.getAsString();
      OS << " " << P->getName() << ";\n";
    }
    OS << "} brisbane_poly_" << Fname << "_args;\n";
    OS << "brisbane_poly_" << Fname << "_args " << Fname << "_args;\n\n";

    OS << "int brisbane_poly_" << Fname << "_init() {\n";
    for (unsigned i = 0; i < nparams; i++) {
      const ParmVarDecl* P = F->getParamDecl(i);
      const QualType T = P->getType();
      if (!T->isPointerType()) continue;
      OS << "  brisbane_poly_mem_init(&" << Fname << "_args." << P->getName() << ");\n";
    }
    OS << "  return BRISBANE_OK;\n}\n\n";

    OS << "int brisbane_poly_" << Fname << "_setarg(int idx, size_t size, void* value) {\n";
    OS << "  switch (idx) {\n";
    for (unsigned i = 0; i < nparams; i++) {
      const ParmVarDecl* P = F->getParamDecl(i);
      const QualType T = P->getType();
      if (T->isPointerType()) continue;
      OS << "    case " << i << ": memcpy(&" << Fname << "_args." << P->getName() << ", value, size); break;\n";
    }
    OS << "    default: return BRISBANE_ERR;\n  }\n  return BRISBANE_OK;\n}\n\n";

    OS << "int brisbane_poly_" << Fname << "_getmem(int idx, brisbane_poly_mem* mem) {\n";
    OS << "  switch (idx) {\n";
    for (unsigned i = 0; i < nparams; i++) {
      const ParmVarDecl* P = F->getParamDecl(i);
      const QualType T = P->getType();
      if (!T->isPointerType()) continue;
      OS << "    case " << i << ": memcpy(mem, &" << Fname << "_args." << P->getName() << ", sizeof(brisbane_poly_mem)); break;\n";
    }
    OS << "    default: return BRISBANE_ERR;\n  }\n  return BRISBANE_OK;\n}\n";

    Kernels.push_back(F);
  }

  void onStartOfTranslationUnit() override {
  }

  void onEndOfTranslationUnit() override {
    SourceManager& SM = Rewrite.getSourceMgr();
    llvm::SmallString<128> Filepath(SM.getFilename(Loc));
    StringRef Filename = llvm::sys::path::filename(Filepath.str());
    Filepath.append(".poly.c");
    llvm::outs() << BBLUE CHECK_O "BrisbaneClang Pass generates [" << Filepath << "]" RESET "\n";
    std::error_code EC;
    llvm::raw_fd_ostream OS(Filepath, EC, llvm::sys::fs::OF_None);
    if (EC) {
      llvm::errs() << EC.message() << "\n";
      return;
    }
    OS << "#include <brisbane/brisbane.h>\n";
    OS << "#include <brisbane/brisbane_poly.h>\n\n";
    OS << "#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n";

    OS << S;

    OS << "\n#include \"" << Filename << ".poly.h\"\n\n";

    OS << "int brisbane_poly_kernel(const char* name) {\n";
    OS << "  brisbane_poly_lock();\n";
    for (unsigned i = 0; i < Kernels.size(); i++) {
      StringRef Fname = Kernels[i]->getName();
      OS << "  if (strcmp(name, \"" << Fname << "\") == 0) {\n";
      OS << "    brisbane_poly_kernel_idx = " << i << ";\n";
      OS << "    return " << Fname << "_poly_available();\n  }\n";
    }
    OS << "  return BRISBANE_ERR;\n}\n\n";

    OS << "int brisbane_poly_setarg(int idx, size_t size, void* value) {\n";
    OS << "  switch (brisbane_poly_kernel_idx) {\n";
    for (unsigned i = 0; i < Kernels.size(); i++) {
      StringRef Fname = Kernels[i]->getName();
      OS << "    case " << i << ": return brisbane_poly_" << Fname << "_setarg(idx, size, value);\n";
    }
    OS << "  }\n  return BRISBANE_ERR;\n}\n\n";

    OS << "int brisbane_poly_launch(int dim, size_t* wgo, size_t* wgs, size_t* gws, size_t* lws) {\n";
    OS << "  int ret = BRISBANE_OK;\n";
    OS << "  switch (brisbane_poly_kernel_idx) {\n";
    for (unsigned i = 0; i < Kernels.size(); i++) {
      StringRef Fname = Kernels[i]->getName();
      OS << "    case " << i << ": brisbane_poly_" << Fname << "_init();";
      OS << " ret = " << Fname << "(wgo[0], wgo[1], wgo[2], wgs[0], wgs[1], wgs[2], gws[0], gws[1], gws[2], lws[0], lws[1], lws[2]); break;\n";
    }
    OS << "  }\n  brisbane_poly_unlock();\n";
    OS << "  return ret;\n}\n\n";

    OS << "int brisbane_poly_getmem(int idx, brisbane_poly_mem* mem) {\n";
    OS << "  switch (brisbane_poly_kernel_idx) {\n";
    for (unsigned i = 0; i < Kernels.size(); i++) {
      StringRef Fname = Kernels[i]->getName();
      OS << "    case " << i << ": return brisbane_poly_" << Fname << "_getmem(idx, mem);\n";
    }
    OS << "  }\n  return BRISBANE_ERR;\n}\n\n";

    OS << "#ifdef __cplusplus\n} /* end of extern \"C\" */\n#endif\n\n";
    OS.close();
  }

private:
  Rewriter &Rewrite;
  SourceLocation Loc;
  SmallVector<const FunctionDecl*, 32> Kernels;
  std::string S;
};

class BrisbaneLLVMCallback : public MatchFinder::MatchCallback {
public:
  BrisbaneLLVMCallback(Rewriter &Rewrite) : Rewrite(Rewrite) {}

  void run(const MatchFinder::MatchResult &Result) override {
    const FunctionDecl *F = Result.Nodes.getNodeAs<FunctionDecl>("kernel");
    const unsigned nparams = F->getNumParams();
    const ParmVarDecl* PVD = F->getParamDecl(nparams - 1);
    Loc = PVD->getEndLoc().getLocWithOffset(PVD->getName().size());
    Rewrite.InsertText(Loc, ", BRISBANE_LLVM_KERNEL_ARGS", true, true);

    Loc = F->getBody()->getBeginLoc();
    Rewrite.InsertTextAfterToken(Loc, "\n  BRISBANE_LLVM_KERNEL_BEGIN;");
    Loc = F->getBody()->getEndLoc();
    Rewrite.InsertText(Loc, "  BRISBANE_LLVM_KERNEL_END;\n", true, true);
  }

  void onStartOfTranslationUnit() override {
  }

  void onEndOfTranslationUnit() override {
    SourceManager& SM = Rewrite.getSourceMgr();
    llvm::SmallString<128> FilePath(SM.getFilename(Loc));
    FilePath.append(".llvm.c");
    llvm::outs() << BBLUE CHECK_O "BrisbaneClang Pass generates [" << FilePath << "]" RESET "\n";
    std::error_code EC;
    llvm::raw_fd_ostream OS(FilePath, EC, llvm::sys::fs::OF_None);
    if (EC) {
      llvm::errs() << EC.message() << "\n";
      return;
    }
    OS << "#include <brisbane/brisbane_llvm.h>\n\n";
    Rewrite.getEditBuffer(Rewrite.getSourceMgr().getMainFileID()).write(OS);
    OS.close();
  }

private:
  Rewriter &Rewrite;
  SourceLocation Loc;
};

class BrisbaneASTConsumer : public ASTConsumer {
public:
  BrisbaneASTConsumer(Rewriter &RWLLVM, Rewriter &RWPOLY) : CallbackLLVM(RWLLVM), CallbackPOLY(RWPOLY) {
    Finder.addMatcher(functionDecl(hasAttr(attr::OpenCLKernel)).bind("kernel"), &CallbackLLVM);
    Finder.addMatcher(functionDecl(hasAttr(attr::OpenCLKernel)).bind("kernel"), &CallbackPOLY);
  }

  void HandleTranslationUnit(ASTContext &Context) override {
    Finder.matchAST(Context);
  }

private:
  BrisbaneLLVMCallback CallbackLLVM;
  BrisbanePOLYCallback CallbackPOLY;
  MatchFinder Finder;
};

class BrisbaneFrontEndAction : public ASTFrontendAction {
public:
  BrisbaneFrontEndAction() {}

  bool BeginSourceFileAction(CompilerInstance &CI) override {
    /*
    auto PO = std::make_shared<PreprocessorOptions>();
    *PO = CI.getPreprocessorOpts();
    PO->Includes.push_back("opencl-c.h");
    */
    return true;
  }

  void EndSourceFileAction() override {
  }

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef file) override {
    RWLLVM.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    RWPOLY.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    return llvm::make_unique<BrisbaneASTConsumer>(RWLLVM, RWPOLY);
  }

private:
  Rewriter RWLLVM;
  Rewriter RWPOLY;
};

int main(int argc, const char **argv) {
  CommonOptionsParser op(argc, argv, BRISBANE_OPTION_CATEGORY);
  ClangTool Tool(op.getCompilations(), op.getSourcePathList());
  return Tool.run(newFrontendActionFactory<BrisbaneFrontEndAction>().get());
}
