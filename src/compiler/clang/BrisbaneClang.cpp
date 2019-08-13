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

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::driver;
using namespace clang::tooling;
using namespace std;

const string type_format_specifier[][2] = {
    {"int", "%d"},   {"long", "%ld"}, {"long long", "%lld"}, {"double", "%lf"},
    {"float", "%f"}, {"char", "%c"},  {"string", "%s"}};

static llvm::cl::OptionCategory BRISBANE_OPTION_CATEGORY("Brisbane Clang");

static llvm::cl::opt<bool>
    wFlag("wrap", llvm::cl::desc("Do you want to wrap a function?"),
          llvm::cl::Optional, llvm::cl::cat(BRISBANE_OPTION_CATEGORY));
//          llvm::cl::Required, llvm::cl::cat(BRISBANE_OPTION_CATEGORY));
static llvm::cl::opt<std::string>
    wrapPrefix("wrap-prefix",
               llvm::cl::desc("Select the prefix of the wrapper."),
               llvm::cl::Optional, llvm::cl::cat(BRISBANE_OPTION_CATEGORY));
static llvm::cl::opt<std::string>
    targetMethod("wrap-target", llvm::cl::desc("Name the function to wrap."),
                 llvm::cl::Optional, llvm::cl::cat(BRISBANE_OPTION_CATEGORY));

static llvm::cl::extrahelp MoreHelp("\nA Clang Libtool to create a wrapper for "
                                    "a function to show its input values\n");

class BrisbaneWrapper : public MatchFinder::MatchCallback {
private:
  string getFormatSpecifier(string index) {
    for (unsigned int i = 0;
         i < sizeof(type_format_specifier) / sizeof(type_format_specifier[0]);
         i++) {
      if (type_format_specifier[i][0] == index) {
        return type_format_specifier[i][1];
      }
    }
    return "%p";
  }

public:
  BrisbaneWrapper(Rewriter &Rewrite) : Rewrite(Rewrite) {}

  // AST matcher match the function declaration with target function
  virtual void run(const MatchFinder::MatchResult &Result) {
    // you can use sourceManager to print debug information about sourceLocation
    // SourceManager &sourceManager = Result.Context->getSourceManager();

    // retrieve the matched function declaration
    const FunctionDecl *func =
        Result.Nodes.getNodeAs<clang::FunctionDecl>("wrapFunc");

    // if function has a body
    if (func->hasBody()) {
      // collect the function return type
      string retType = func->getReturnType().getAsString();
      // collect number of params in the function
      unsigned int paramNum = func->getNumParams();

      // we create text for function body and signature
      string funcParamSignature = "";
      string funcBody = "";
      // we create text for call the target function from wrap function
      string argString = "";

      for (unsigned int i = 0; i < paramNum; i++) {
        // param_type param_name
        funcParamSignature += func->getParamDecl(i)->getType().getAsString() +
                              " " + func->getParamDecl(i)->getName().str();
        // argument_name
        argString += func->getParamDecl(i)->getName().str();
        if (i < paramNum - 1) {
          funcParamSignature += ", ";
          argString += ", ";
        }

        // creates a printf for every param to print their value
        string format_specifier =
            getFormatSpecifier(func->getParamDecl(i)->getType().getAsString());
        string pName;
        if (format_specifier == "%p") {
          // if not specified in format specifier array, then will print the
          // address
          pName = "&" + func->getParamDecl(i)->getName().str();
        } else {
          pName = func->getParamDecl(i)->getName().str();
        }
        funcBody +=
            "    printf(\"" + format_specifier + "\\n\", " + pName + ");\n";
      }
      // at the end of function body, we call the target function with return
      // statement
      funcBody += "    return " + targetMethod + "(" + argString + ");";

      // the target function end point is the '}', so we ask for +1 offset
      SourceLocation TARGET_END = func->getEndLoc().getLocWithOffset(1);
      std::stringstream wrapFunction;
      string wrapFunctionName = wrapPrefix + "_" + targetMethod;
      // we create the entire wrap function text
      wrapFunction << "\n" + retType + " " + wrapFunctionName + +"(" +
                          funcParamSignature + ")\n{\n" + funcBody + "\n}";
      // let's insert the wrap function at the end of target function
      Rewrite.InsertText(TARGET_END, wrapFunction.str(), true, true);
    }
  }

private:
  Rewriter &Rewrite;
};

class BrisbanePollyCallback : public MatchFinder::MatchCallback {
public:
  BrisbanePollyCallback(Rewriter &Rewrite) : Rewrite(Rewrite) {}

  void run(const MatchFinder::MatchResult &Result) override {
    SourceManager &SM = *Result.SourceManager;
    const LangOptions &Opts = Result.Context->getLangOpts();
    const FunctionDecl *F = Result.Nodes.getNodeAs<FunctionDecl>("kernel");
    const unsigned nparams = F->getNumParams();
    const ParmVarDecl* PVD = F->getParamDecl(nparams - 1);
//    SourceLocation Loc = PVD->getEndLoc().getLocWithOffset(PVD->getName().size());
    Loc = PVD->getEndLoc().getLocWithOffset(PVD->getName().size());
    Rewrite.InsertText(Loc, ", BRISBANE_POLY_KERNEL_ARGS", true, true);

    Loc = F->getBody()->getBeginLoc();
    Rewrite.InsertTextAfterToken(Loc, "\n  BRISBANE_POLY_KERNEL_BEGIN;");
    Loc = F->getBody()->getEndLoc();
    Rewrite.InsertText(Loc, "  BRISBANE_POLY_KERNEL_END;\n", true, true);
  }

  void onStartOfTranslationUnit() override {
  }

  void onEndOfTranslationUnit() override {
    SourceManager& SM = Rewrite.getSourceMgr();
    llvm::SmallString<128> FilePath(SM.getFilename(Loc));
    FilePath.append(".poly.c");
    llvm::outs() << "Generated [" << FilePath << "] for Brisbane Poly.\n";
    std::error_code EC;
    llvm::raw_fd_ostream OS(FilePath, EC, llvm::sys::fs::OF_None);
    if (EC) {
      llvm::errs() << EC.message() << "\n";
      return;
    }
    OS << "#include \"brisbane_poly_kernel.h\"\n\n";
    Rewrite.getEditBuffer(Rewrite.getSourceMgr().getMainFileID()).write(OS);
    OS.close();
  }

private:
  Rewriter &Rewrite;
  SourceLocation Loc;
};

class BrisbaneASTConsumer : public ASTConsumer {
public:
  BrisbaneASTConsumer(Rewriter &R) : handleWrapper(R), CallbackPolly(R) {
    //Finder.addMatcher(functionDecl(hasName(targetMethod)).bind("wrapFunc"), &handleWrapper);
    Finder.addMatcher(functionDecl(hasAttr(attr::OpenCLKernel)).bind("kernel"), &CallbackPolly);
  }

  void HandleTranslationUnit(ASTContext &Context) override {
    Finder.matchAST(Context);
  }

private:
  BrisbaneWrapper handleWrapper;
  BrisbanePollyCallback CallbackPolly;
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
//    TheRewriter.getEditBuffer(TheRewriter.getSourceMgr().getMainFileID()).write(llvm::outs());
  }

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef file) override {
    TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    return llvm::make_unique<BrisbaneASTConsumer>(TheRewriter);
  }

private:
  Rewriter TheRewriter;
};

int main(int argc, const char **argv) {
  CommonOptionsParser op(argc, argv, BRISBANE_OPTION_CATEGORY);
  ClangTool Tool(op.getCompilations(), op.getSourcePathList());

  /*
  if (wFlag) {
    if (targetMethod.length()) {
      llvm::errs() << "The target wrap function: " << targetMethod << "\n";
      if (wrapPrefix.length()) {
        llvm::errs() << "Prefix (User): " << wrapPrefix << "\n";
      } else {
        wrapPrefix = "syssec";
        llvm::errs() << "Prefix (Default): " << wrapPrefix << "\n";
      }
    } else {
      llvm::errs() << "Please, input a target function name.\n";
    }
  }
  */

  return Tool.run(newFrontendActionFactory<BrisbaneFrontEndAction>().get());
}
