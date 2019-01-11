using SourceGraphExtractionUtils;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataExtraction.Utils
{
    public static class RoslynUtils
    {
        /// <summary>
        /// Get the symbol of a given node.
        /// </summary>
        /// <param name="node"></param>
        /// <param name="semanticModel"></param>
        /// <returns></returns>
        public static ISymbol GetReferenceSymbol(SyntaxNode node, SemanticModel semanticModel)
        {
            ISymbol identifierSymbol = semanticModel.GetSymbolInfo(node).Symbol;
            if (identifierSymbol == null)
            {
                identifierSymbol = semanticModel.GetDeclaredSymbol(node);
            }
            return identifierSymbol;
        }

        public static bool IsVariableLike(IdentifierNameSyntax node, SemanticModel semanticModel, out ISymbol nodeSymbol)
        {
            nodeSymbol = semanticModel.GetSymbolInfo(node).Symbol;
            if (nodeSymbol is IFieldSymbol || nodeSymbol is IPropertySymbol)
            {
                // We need to check if the field/property is in an LHR on an ObjectInitializerExpression
                return !(node.Parent is AssignmentExpressionSyntax assignmentExpression
                            && assignmentExpression.Parent is InitializerExpressionSyntax initializer)
                       || !assignmentExpression.Left.Equals(node);
            }
            return nodeSymbol is ILocalSymbol || nodeSymbol is IParameterSymbol ||
                   nodeSymbol is IEventSymbol;
        }

        public static IEnumerable<SyntaxToken> GetAllVariableSymbolsInSyntaxTree(SyntaxNode node, SemanticModel semanticModel)
        {
            foreach (var descendant in node.DescendantNodesAndSelf())
            {
                var idNode = descendant as IdentifierNameSyntax;
                if (idNode == null) continue;
                if (IsVariableLike(idNode, semanticModel, out var _))
                {
                    yield return idNode.Identifier;
                }
            }
        }

        /// <summary>
        /// Find all the symbols used in a given tree.
        /// </summary>
        private class VariableSymbolFinder : CSharpSyntaxWalker
        {
            private readonly HashSet<ISymbol> relevantSymbols;
            private readonly SemanticModel semanticModel;
            private readonly bool onlyLocalFileVariables;

            public VariableSymbolFinder(HashSet<ISymbol> relevantSymbols, SemanticModel semanticModel, bool onlyLocalFileVariables = true)
            {
                this.relevantSymbols = relevantSymbols;
                this.semanticModel = semanticModel;
                this.onlyLocalFileVariables = onlyLocalFileVariables;
            }

            public override void Visit(SyntaxNode node)
            {
                ISymbol symbol = GetReferenceSymbol(node, semanticModel);
                if (symbol != null
                    && !(symbol is IMethodSymbol) && !(symbol is INamespaceOrTypeSymbol)
                    && !(symbol is IPreprocessingSymbol) && !(symbol is ITypeSymbol)
                    && !(symbol is ILabelSymbol))
                {
                    if (symbol.OriginalDefinition != null && symbol.Locations.Length > 0 && (!onlyLocalFileVariables || symbol.Locations.First().SourceTree == node.SyntaxTree))
                    {
                        relevantSymbols.Add(symbol.OriginalDefinition);
                    }
                }
                base.Visit(node);
            }
        }

        private class MethodSymbolFinder : CSharpSyntaxWalker
        {
            HashSet<IMethodSymbol> relevantSymbols;
            SemanticModel semanticModel;

            public MethodSymbolFinder(HashSet<IMethodSymbol> relevantSymbols, SemanticModel semanticModel)
            {
                this.relevantSymbols = relevantSymbols;
                this.semanticModel = semanticModel;
            }

            public override void Visit(SyntaxNode node)
            {
                ISymbol symbol = GetReferenceSymbol(node, semanticModel);
                if (symbol != null && (symbol is IMethodSymbol))
                {
                    var methodSymbol = symbol as IMethodSymbol;
                    relevantSymbols.Add(methodSymbol.OriginalDefinition);                    
                }
                base.Visit(node);
            }
        }

        private class MethodDeclarationFinder: CSharpSyntaxWalker
        {
            public readonly List<MethodDeclarationSyntax> Methods = new List<MethodDeclarationSyntax>();
            public override void VisitMethodDeclaration(MethodDeclarationSyntax node)
            {
                Methods.Add(node);
            }
        }

        public static IEnumerable<MethodDeclarationSyntax> GetMethodDeclarationsInNode(SyntaxNode node)
        {
            var df = new MethodDeclarationFinder();
            df.Visit(node);
            return df.Methods;
        }

        public static InvocationExpressionSyntax GetInvocation(SyntaxToken token, IMethodSymbol methodSymbol, SemanticModel semanticModel)
        {
            var node = token.Parent;
            while (node != null && !node.IsKind(SyntaxKind.InvocationExpression))
            {
                node = node.Parent;
            }
            if (node == null)
            {
                return null;
            }
            return node as InvocationExpressionSyntax;
        }

        /// <summary>
        /// Get all the symbols in a tree.
        /// </summary>
        /// <param name="semanticModel"></param>
        /// <param name="currentNode"></param>
        /// <returns></returns>
        public static ISet<ISymbol> GetUsedVariableSymbols(SemanticModel semanticModel, SyntaxNode currentNode, bool onlyLocalFileVariables = true)
        {
            HashSet<ISymbol> usedSymbols = new HashSet<ISymbol>();
            VariableSymbolFinder sf = new VariableSymbolFinder(usedSymbols, semanticModel, onlyLocalFileVariables);
            sf.Visit(currentNode);
            return usedSymbols;
        }

        public static ISet<IMethodSymbol> GetUsedMethodSymbols(SemanticModel semanticModel, SyntaxNode currentNode)
        {
            var usedSymbols = new HashSet<IMethodSymbol>();
            MethodSymbolFinder sf = new MethodSymbolFinder(usedSymbols, semanticModel);
            sf.Visit(currentNode);
            return usedSymbols;
        }

        public static Dictionary<SyntaxNode, SyntaxToken> CreateNodeToTokenMap(SemanticModel semanticModel,
            Dictionary<SyntaxToken, ISymbol> variableTokens,
            IEnumerable<SyntaxNode> nodes)
        {
            var nodesToTokens = new Dictionary<SyntaxNode, SyntaxToken>();
            foreach (var syntaxNode in nodes)
            {
                SyntaxToken token;
                switch (syntaxNode)
                {
                    case PropertyDeclarationSyntax propDecl:
                        token = propDecl.Identifier;
                        break;
                    case VariableDeclaratorSyntax varDecl:
                        token = varDecl.Identifier;
                        break;
                    case ParameterSyntax parSyn:
                        token = parSyn.Identifier;
                        break;
                    case MethodDeclarationSyntax methodDecl:
                        continue;
                    case AssignmentExpressionSyntax assignExpr:
                        SyntaxNode baseExpr = assignExpr.Left;
                        bool visit = true;
                        while (visit)
                        {
                            visit = false;
                            if (baseExpr is ElementAccessExpressionSyntax access)
                            {
                                baseExpr = access.Expression;
                                visit = true;
                            }
                            else if (baseExpr is PrefixUnaryExpressionSyntax prefSyn)
                            {
                                baseExpr = prefSyn.Operand;
                                visit = true;
                            }
                            else if (baseExpr is CastExpressionSyntax castExpr)
                            {
                                baseExpr = castExpr.Expression;
                                visit = true;
                            }
                        }
                        var symbol = semanticModel.GetSymbolInfo(baseExpr).Symbol;
                        var tokens = baseExpr.DescendantTokens()
                            .Where(t => variableTokens.ContainsKey(t) && (symbol == null || t.Text.Equals(symbol.Name))).ToArray();
                        if (tokens.Length == 1) token = tokens[0];
                        else continue;
                        break;
                    case AccessorDeclarationSyntax accessorSyn:
                        var parent = accessorSyn.Parent.Parent;
                        switch (parent)
                        {
                            case PropertyDeclarationSyntax parentProperty:
                                token = parentProperty.Identifier;
                                break;
                            case IndexerDeclarationSyntax indexerProperty:
                                token = indexerProperty.ThisKeyword;
                                break;
                            case EventDeclarationSyntax eventParent:
                                token = eventParent.Identifier;
                                break;
                            default:
                                throw new Exception("Should not happen: " + parent.GetType());
                        }
                        break;
                    case ForEachStatementSyntax foreachSyn:
                        token = foreachSyn.Identifier;
                        break;
                    default:
                        var nodeTokens = syntaxNode.DescendantTokens().Where(t => variableTokens.ContainsKey(t)).ToList();
                        if (nodeTokens.Count == 1)
                        {
                            token = nodeTokens[0];
                        }
                        else if (nodeTokens.Count == 0)
                        {
                            continue;
                        }
                        else
                        {
                            throw new Exception("This shouldn't happen");
                        }
                        break;
                }
                nodesToTokens.Add(syntaxNode, token);
            }

            return nodesToTokens;
        }

        public static ISymbol GetTokenSymbolReference(SyntaxToken token, SemanticModel semanticModel)
        {
            if (!token.IsKind(SyntaxKind.IdentifierToken) && !token.IsKind(SyntaxKind.ThisKeyword)) return null;

            SyntaxNode node = token.Parent;
            while (node.Parent != null)
            {
                ISymbol nodeSymbol = RoslynUtils.GetReferenceSymbol(node, semanticModel);
                if (nodeSymbol == null)
                {
                    node = node.Parent;
                    continue;
                }
                if (token.Text.StartsWith("@"))
                {
                    if (token.Text.Substring(1) != nodeSymbol.Name) break;
                }
                else if (nodeSymbol.ToDisplayString() != token.Text && nodeSymbol.Name != token.Text)
                {
                    break;
                }

                return nodeSymbol;
            }
            return null;
        }

        public static IEnumerable<ISymbol> GetAvailableValueSymbols(SemanticModel semanticModel, SyntaxToken token)
        {
            var allSymbols = semanticModel.LookupSymbols(token.SpanStart);
            foreach (var symbol in allSymbols)
            {
                if (symbol.Kind == SymbolKind.Local)
                {
                    ILocalSymbol localSymbol = (ILocalSymbol)symbol;
                    var declarationSyntax = localSymbol.DeclaringSyntaxReferences[0].GetSyntax();
                    int declarationEnd;
                    switch (declarationSyntax)
                    {
                        case ForEachStatementSyntax foreachSyntax:
                            declarationEnd = foreachSyntax.CloseParenToken.SpanStart;
                            break;
                        default:
                            declarationEnd = declarationSyntax.Span.End;
                            break;
                    }
                    if (declarationEnd < token.SpanStart)
                    {
                        yield return localSymbol;
                    }
                }
                else
                {
                    if (symbol.Kind == SymbolKind.Field || symbol.Kind == SymbolKind.Property || symbol.Kind == SymbolKind.Parameter) 
                    {
                        yield return symbol;
                    }
                }
            }
        }

        public static List<ISymbol> ComputeVariablesToConsider(ISet<ISymbol> variableCandidates,
            ICollection<ISymbol> knownTokens)
        {
            var variableSymbolsToUse = new List<ISymbol>();
            foreach (var symbol in variableCandidates)
            {
                if (symbol.IsImplicitlyDeclared) continue;
                if (symbol is IAliasSymbol || symbol is IRangeVariableSymbol)
                {
                    continue;
                }
                if (!knownTokens.Contains(symbol)) continue;

                variableSymbolsToUse.Add(symbol);
            }

            return variableSymbolsToUse;
        }

        public static bool GetTypeSymbol(ISymbol symbol, out ITypeSymbol result)
        {
            if (symbol != null
                && !(symbol is IMethodSymbol) && !(symbol is INamespaceOrTypeSymbol)
                && !(symbol is IPreprocessingSymbol) && !(symbol is ITypeSymbol)
                && !(symbol is ILabelSymbol)
                && TypeHierarchy.ComputeTypeForSymbol(symbol, out var typeSym))
            {
                result = typeSym;
                return true;
            }
            else if (symbol is IMethodSymbol methodSymbol)
            {
                result = methodSymbol.ReturnType; // methods' symbols are their return types, for now.
                return true;
            }
            result = null;
            return false;
        }

        public static bool IsSimpleLiteral(SyntaxNodeOrToken node)
            => node.IsKind(SyntaxKind.FalseKeyword)
               || node.IsKind(SyntaxKind.TrueKeyword)
               || node.IsKind(SyntaxKind.NullKeyword)
               || node.IsKind(SyntaxKind.NumericLiteralExpression)
               || node.IsKind(SyntaxKind.CharacterLiteralExpression)
               || node.IsKind(SyntaxKind.StringLiteralExpression);
    }
}
