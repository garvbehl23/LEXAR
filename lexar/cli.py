"""
LEXAR Command Line Interface

Provides basic CLI commands for LEXAR operations.
"""

import argparse
import sys
from lexar import __version__, LexarPipeline


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="lexar",
        description="LEXAR: Legal EXplainable Augmented Reasoner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  lexar --version              Show version information
  lexar query "What is IPC Section 302?"   Ask a legal question
  
For more information, visit: https://github.com/yourusername/legalrag
        """
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"LEXAR v{__version__}"
    )
    
    parser.add_argument(
        "command",
        nargs="?",
        choices=["query", "info"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "args",
        nargs="*",
        help="Arguments for the command"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with provenance information"
    )
    
    args = parser.parse_args()
    
    if args.command == "query":
        if not args.args:
            print("Error: query command requires a question", file=sys.stderr)
            return 1
        
        query = " ".join(args.args)
        pipeline = LexarPipeline()
        
        print(f"Query: {query}\n")
        result = pipeline.answer(query, debug_mode=args.debug)
        
        if result.get("refused"):
            print(f"[REFUSED] {result['refusal']['reason']}")
            print(f"Suggestion: {result['refusal'].get('suggestion', 'N/A')}")
            return 2
        
        print(f"Answer: {result['answer']}\n")
        
        if args.debug and "debug" in result:
            print("=" * 60)
            print("DEBUG INFORMATION")
            print("=" * 60)
            print(result["debug"].get("attention_visualization", "No visualization available"))
        
        return 0
    
    elif args.command == "info":
        print(f"LEXAR v{__version__}")
        print("Legal EXplainable Augmented Reasoner")
        print("\nA retrieval-augmented generation system for legal QA")
        print("with strict evidence grounding and explainable provenance.")
        return 0
    
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
